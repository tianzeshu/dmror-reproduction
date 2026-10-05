"""Independent discrete cascade simulator, explicitly synthetic, never real records.

Labels come from delayed Bernoulli transmissions and accumulating inventory
exhaustion; the simulator does not use the DM-ROR neural equations or parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import audit_dataset, hash_text, write_json

DOMAINS = {
    "SIM-Auto": {"seed": 1201, "buffer": .58, "coupling": .88, "noise": .16, "shock": "component delivery"},
    "SIM-Semi": {"seed": 1202, "buffer": .38, "coupling": 1.10, "noise": .20, "shock": "wafer production"},
    "SIM-Energy": {"seed": 1203, "buffer": .68, "coupling": 1.00, "noise": .24, "shock": "fuel transport"},
}


def build(output, name, snapshots=84, signal_dim=64, retrieval_size=5):
    cfg = DOMAINS[name]
    rng = np.random.default_rng(cfg["seed"])
    n = 112
    node_type = np.array([0] * 80 + [1] * 12 + [2] * 10 + [3] * 5 + [4] * 5, dtype=np.int64)
    edges = set()
    # Firm layers enforce economic direction; auxiliary heterogeneous links
    # test whether the same implementation supports all paper entity types.
    for u in range(60):
        layer = u // 20
        for v in rng.choice(np.arange((layer + 1) * 20, (layer + 2) * 20), size=3, replace=False):
            edges.add((u, int(v), 0))
    for u in range(80):
        edges.add((u, 80 + u % 12, 1))
        edges.add((u, 102 + u % 5, 3))
        edges.add((u, 107 + u % 5, 4))
    for u in range(92, 102):
        for v in rng.choice(np.arange(80, 92), size=3, replace=False):
            edges.add((u, int(v), 2))
    triples = np.array(sorted(edges), dtype=np.int64)
    src, dst, edge_type = triples.T
    e = len(src)
    dependency = rng.uniform(.3, .95, e).astype(np.float32)
    edge_features = rng.uniform(.05, .75, (e, 4)).astype(np.float32)
    # Stable observed capacities are independently sampled, never label-derived.
    capacity = np.clip(rng.normal(cfg["buffer"], .17, (n, 5)), .05, .95).astype(np.float32)
    node_features = np.column_stack([np.eye(5)[node_type], rng.normal(size=(n, 3))]).astype(np.float32)
    days = np.arange(snapshots, dtype=np.int64) * 10 + 100
    train_boundary, val_boundary = days[int(snapshots * .6)], days[int(snapshots * .8)]
    split = np.where(days < train_boundary, 0, np.where(days < val_boundary, 1, 2)).astype(np.int64)
    split[((split == 0) & (days + 30 >= train_boundary)) | ((split == 1) & (days + 30 >= val_boundary))] = -1
    signals = np.zeros((snapshots, n, retrieval_size, signal_dim), np.float32)
    signal_mask = np.zeros((snapshots, n, retrieval_size), bool)
    delta = np.zeros((snapshots, n, retrieval_size), np.float32)
    confidence = np.zeros_like(delta)
    memory_indices = np.full((snapshots, n, retrieval_size), -1, np.int64)
    labels = np.zeros((snapshots, n), np.float32)
    edge_labels = np.zeros((snapshots, e), np.float32)
    source_mask = np.zeros((snapshots, n), bool)
    resilience = np.repeat(capacity[None], snapshots, axis=0)
    texts, by_text, observations = [], {}, []

    def get_signal(text):
        if text not in by_text:
            by_text[text] = len(texts)
            texts.append({"signal_id": len(texts), "text": text, "source_kind": "simulated_template"})
        return by_text[text]

    for ti, day in enumerate(days):
        # Independent future disturbance sequence. The last observed warning
        # precedes the query and noisily predicts a source's impending failure.
        sources = rng.choice(np.r_[np.arange(60), np.arange(92, 102)], size=int(rng.integers(2, 5)), replace=False)
        severity = np.zeros(n)
        severity[sources] = rng.uniform(.65, 1.3, len(sources))
        source_mask[ti, sources] = True
        false_alarms = rng.choice(np.setdiff1d(np.arange(n), sources), size=8, replace=False)
        observed = np.zeros(n)
        observed[sources] = np.clip(severity[sources] + rng.normal(0, cfg["noise"], len(sources)), .1, 1.4)
        observed[false_alarms] = rng.uniform(.1, 1.1, len(false_alarms))
        for v in range(n):
            # Some routine updates provide neutral context, with explicit noise.
            if observed[v] == 0 and rng.random() > .30:
                continue
            strength = "severe" if observed[v] > .9 else "moderate" if observed[v] > .45 else "minor" if observed[v] > .05 else "no"
            uncertainty = "confirmed" if v in sources and rng.random() > .15 else "uncertain"
            type_name = ["firm", "product", "material", "industry", "region"][node_type[v]]
            text = f"A {type_name} reports {strength} {cfg['shock']} disruption. Evidence is {uncertainty}. Delivery and production may be affected."
            sid = get_signal(text)
            age = float(rng.integers(1, 8))
            signals[ti, v, 0] = hash_text(text, signal_dim)
            signal_mask[ti, v, 0] = True
            delta[ti, v, 0] = age
            confidence[ti, v, 0] = .9 if uncertainty == "confirmed" else .55
            memory_indices[ti, v, 0] = sid
            observations.append({"prediction_day": int(day), "entity_id": int(v), "signal_id": sid, "timestamp_day": int(day - age), "confidence": float(confidence[ti, v, 0]), "label_source": "not_a_label"})
            # Contradictory older routine signals test temporal weighting.
            if rng.random() < .55:
                old = f"A {type_name} reports stable operations and adequate inventory. No disruption was observed."
                oid = get_signal(old)
                signals[ti, v, 1] = hash_text(old, signal_dim)
                signal_mask[ti, v, 1] = True
                delta[ti, v, 1] = float(rng.integers(12, 30))
                confidence[ti, v, 1] = .8
                memory_indices[ti, v, 1] = oid
                observations.append({"prediction_day": int(day), "entity_id": int(v), "signal_id": oid, "timestamp_day": int(day - delta[ti, v, 1]), "confidence": .8, "label_source": "not_a_label"})

        # Ground truth: three 10-day cascade stages, delayed stochastic
        # transmission, inventory exhaustion, and hidden operational noise.
        failed = severity > (capacity.mean(1) + .30)
        labels[ti, failed] = 1
        stock = capacity.mean(1) * .65 + rng.uniform(.04, .12, n)
        frontier = failed.copy()
        for _ in range(3):
            active_edges = frontier[src]
            chance = np.clip(cfg["coupling"] * dependency * (1 - edge_features.mean(1)), 0, .95)
            transmitted = active_edges & (rng.random(e) < chance)
            increment = np.zeros(n)
            np.add.at(increment, dst[transmitted], rng.uniform(.5, 1.2, transmitted.sum()))
            stock -= increment
            new_failed = (stock < 0) & ~failed
            # Only edges delivering a load to an actually failed receiver are
            # recorded as causal successful transmissions in this simulator.
            successful = transmitted & new_failed[dst]
            edge_labels[ti, successful] = 1
            failed |= new_failed
            labels[ti, new_failed] = 1
            frontier = new_failed

    output = Path(output) / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", node_features=node_features, node_type=node_type, src=src, dst=dst, edge_type=edge_type, edge_dependency=dependency, edge_features=edge_features, resilience=resilience, signals=signals, signal_mask=signal_mask, delta=delta, confidence=confidence, memory_indices=memory_indices, labels=labels, edge_labels=edge_labels, source_mask=source_mask, split=split, prediction_days=days, target_days=days + 30)
    for filename, records in [("signals.jsonl", texts), ("signal_observations.jsonl", observations)]:
        with open(output / filename, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {"name": name, "data_kind": "synthetic", "label_kind": "independent_delayed_stochastic_cascade", "is_real_world": False, "seed": cfg["seed"], "generator": "dmror.build_simulated", "generator_settings": cfg, "forecast_horizon_days": 30, "query_step_days": 10, "temporal_split": "60/20/20 with purged boundary targets", "text_encoder": "sha256_signed_hash_64_diagnostic_only", "resilience_source": "simulated_observed_capacity", "path_ground_truth": "simulator_successful_transmission_edges", "scope_ground_truth": "simulator_failed_nodes", "limitations": ["Simulated graphs, texts, capacities and labels are not the paper SC-Auto/SC-Semi/SC-Energy datasets.", "Synthetic evidence cannot establish real supply-chain performance.", "Overlapping 30-day horizons yield correlated snapshots; seed standard deviation is not population confidence."], "statistics": audit_dataset(output / "dataset.npz")}
    write_json(output / "metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--snapshots", type=int, default=84)
    args = parser.parse_args()
    for domain in DOMAINS:
        print(json.dumps(build(args.output, domain, snapshots=args.snapshots), ensure_ascii=False))
