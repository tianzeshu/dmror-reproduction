"""Table-1-sized, explicitly synthetic supply-chain datasets.

The paper's original observations are unavailable. These generated records match
the requested counts, not their empirical distribution or reported performance.
Ground truth comes from an independent delayed inventory-exhaustion simulator;
DM-ROR equations and trained predictions are never used to manufacture labels.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from .data import hash_text, sha256, write_json


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    seed: int
    firms: int
    products: int
    materials: int
    industries: int
    regions: int
    relations: int
    texts: int
    risk_signals: int
    risk_labels: int
    buffer: float
    coupling: float
    topic: str

    @property
    def nodes(self):
        return sum((self.firms, self.products, self.materials, self.industries, self.regions))


SPECS = {
    "SC-Auto-Synthetic": DatasetSpec("SC-Auto-Synthetic", 864201, 6482, 1126, 438, 76, 42,
                                    58734, 124680, 38912, 8426, .54, .98, "vehicle component"),
    "SC-Semi-Synthetic": DatasetSpec("SC-Semi-Synthetic", 864202, 4935, 842, 316, 54, 37,
                                    46218, 96440, 31705, 6913, .43, 1.04, "semiconductor wafer"),
    "SC-Energy-Synthetic": DatasetSpec("SC-Energy-Synthetic", 864203, 5714, 973, 512, 68, 51,
                                      52906, 108375, 34286, 7584, .63, .95, "energy equipment"),
}
ENTITY_TYPES = ("firm", "product", "material", "industry", "region")
RELATION_TYPES = ("supplies", "produces", "input_to", "member_of", "located_in")
EPOCH = date(1970, 1, 1)
FIRST = (date(2018, 1, 1) - EPOCH).days
LAST = (date(2025, 12, 31) - EPOCH).days


def iso_day(day):
    return (EPOCH + timedelta(days=int(day))).isoformat()


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def scaled_spec(name, scale):
    """A test fixture only: full CLI construction always uses the unscaled spec."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    original = SPECS[name]
    if scale == 1:
        return original
    counts = {key: max(2, round(getattr(original, key) * scale))
              for key in ("firms", "products", "materials", "industries", "regions")}
    auxiliary = 2 * counts["firms"] + 2 * counts["products"] + 2 * counts["materials"]
    count_relations = max(round(original.relations * scale), auxiliary + counts["firms"])
    total_nodes = sum(counts.values())
    count_signals = max(95, round(original.risk_signals * scale))
    return replace(original, **counts, relations=count_relations,
                   risk_signals=count_signals,
                   texts=max(round(original.texts * scale), count_signals + total_nodes),
                   risk_labels=max(total_nodes, round(original.risk_labels * scale)))


def build_graph(spec, rng):
    counts = [spec.firms, spec.products, spec.materials, spec.industries, spec.regions]
    offsets = np.r_[0, np.cumsum(counts)].astype(int)
    node_type = np.repeat(np.arange(5, dtype=np.int64), counts)
    triples = set()
    # Membership and location are descriptive context, with complete firm and
    # auxiliary-node coverage. They do not cause inventory exhaustion.
    for firm in range(spec.firms):
        triples.add((firm, int(offsets[3] + firm % spec.industries), 3))
        triples.add((firm, int(offsets[4] + firm % spec.regions), 4))
    # Each product has two distinct manufacturers; every material has two users.
    for product in range(offsets[1], offsets[2]):
        for firm in rng.choice(spec.firms, size=2, replace=False):
            triples.add((int(firm), int(product), 1))
    for material in range(offsets[2], offsets[3]):
        for product in rng.choice(np.arange(offsets[1], offsets[2]), size=2, replace=False):
            triples.add((int(material), int(product), 2))
    remaining = spec.relations - len(triples)
    if remaining <= 0:
        raise ValueError("Relation budget does not leave room for firm supply links")
    tiers = np.minimum(4, np.arange(spec.firms) * 5 // spec.firms)
    tier_nodes = [np.flatnonzero(tiers == tier) for tier in range(5)]
    possible = sum(len(tier_nodes[a]) * len(tier_nodes[b]) for a in range(5) for b in range(a + 1, 5))
    if remaining > possible:
        raise ValueError("More firm supply links requested than unique forward-tier pairs")
    target = spec.relations
    while len(triples) < target:
        first_tier = int(rng.integers(0, 4))
        later = np.arange(first_tier + 1, 5)
        weights = np.exp(-.9 * (later - first_tier - 1))
        second_tier = int(rng.choice(later, p=weights / weights.sum()))
        src = int(rng.choice(tier_nodes[first_tier]))
        dst = int(rng.choice(tier_nodes[second_tier]))
        triples.add((src, dst, 0))
    src, dst, edge_type = np.asarray(sorted(triples), dtype=np.int64).T
    dependency = rng.uniform(.30, .95, spec.relations).astype(np.float32)
    edge_features = rng.uniform(.05, .75, (spec.relations, 4)).astype(np.float32)
    capacity = np.clip(rng.normal(spec.buffer, .14, (spec.nodes, 5)), .12, .94).astype(np.float32)
    node_features = np.column_stack((np.eye(5)[node_type], rng.normal(size=(spec.nodes, 3)))).astype(np.float32)
    return node_type, offsets, tiers, src, dst, edge_type, dependency, edge_features, capacity, node_features


def simulate_cascade(spec, rng, src, dst, edge_type, dependency, edge_features,
                     capacity, sources, severity):
    """Three ten-day Bernoulli delivery stages, independent of model equations."""
    n, e = len(capacity), len(src)
    failed = np.zeros(n, dtype=bool)
    failed[sources] = severity > capacity[sources].mean(1) + .25
    frontier = failed.copy()
    stock = capacity.mean(1).astype(np.float64) * .75 + rng.uniform(.03, .10, n)
    causal = np.zeros(e, dtype=np.int8)
    events = []
    supply_relation = edge_type <= 2
    chance = np.clip(spec.coupling * dependency * (1 - .45 * edge_features.mean(1)), .02, .85)
    for stage in range(1, 4):
        transmitted = frontier[src] & supply_relation & (rng.random(e) < chance)
        amount = rng.uniform(.35, .85, int(transmitted.sum()))
        load = np.zeros(n)
        np.add.at(load, dst[transmitted], amount)
        stock -= load
        newly_failed = (stock < 0) & ~failed
        successful = transmitted & newly_failed[dst]
        causal[successful] = 1
        for edge_id in np.flatnonzero(successful):
            events.append((int(edge_id), stage))
        failed |= newly_failed
        frontier = newly_failed
    return failed, causal, events


def build(output, name, *, seed=None, scale=1.0, signal_dim=64, retrieval_size=5):
    spec = scaled_spec(name, scale)
    actual_seed = spec.seed if seed is None else int(seed)
    # Independent random streams prevent annotation selection from depending
    # on graph outcomes or the text-generation sampling sequence.
    graph_rng, event_rng, text_rng, annotation_rng = [np.random.default_rng(s)
        for s in np.random.SeedSequence(actual_seed).spawn(4)]
    (node_type, offsets, tiers, src, dst, edge_type, dependency,
     edge_features, capacity, node_features) = build_graph(spec, graph_rng)
    output = Path(output) / name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset.npz").exists():
        raise FileExistsError(f"Refusing to replace an existing dataset: {output}")
    days = np.asarray([(date(year, month, 1) - EPOCH).days
                       for year in range(2018, 2026) for month in range(1, 13)], dtype=np.int64)
    t, n, e = len(days), spec.nodes, spec.relations
    train_boundary, val_boundary = days[int(t * .6)], days[int(t * .8)]
    split = np.where(days < train_boundary, 0, np.where(days < val_boundary, 1, 2)).astype(np.int64)
    split[((split == 0) & (days + 30 >= train_boundary)) |
          ((split == 1) & (days + 30 >= val_boundary))] = -1
    # Nearly all dense memory slots are zero; compression removes that padding.
    # Entity/relationship/text counts are counts of genuine distinct records.
    signals = np.zeros((t, n, retrieval_size, signal_dim), dtype=np.float32)
    signal_mask = np.zeros((t, n, retrieval_size), dtype=bool)
    delta = np.zeros((t, n, retrieval_size), dtype=np.float32)
    confidence = np.zeros_like(delta)
    memory_indices = np.full((t, n, retrieval_size), -1, dtype=np.int32)
    labels = np.full((t, n), -1, dtype=np.int8)
    truth = np.zeros((t, n), dtype=np.int8)
    edge_labels = np.zeros((t, e), dtype=np.int8)
    source_mask = np.zeros((t, n), dtype=bool)
    risk_records, risk_texts, shocks, causal_events = [], [], [], []
    signal_vectors = []
    count_by_time = np.full(t - 1, spec.risk_signals // (t - 1), dtype=int)
    count_by_time[:spec.risk_signals % (t - 1)] += 1
    material_nodes = np.arange(offsets[2], offsets[3])
    upstream = np.flatnonzero(tiers <= 1)
    for ti, day in enumerate(days):
        if ti == 0:
            continue  # No invented 2017 evidence outside the requested span.
        budget = int(count_by_time[ti - 1])
        number_sources = min(6, max(1, budget // 12), len(upstream) + len(material_nodes))
        number_material = min(2, number_sources // 3, len(material_nodes))
        sources = np.r_[event_rng.choice(upstream, size=number_sources - number_material, replace=False),
                        event_rng.choice(material_nodes, size=number_material, replace=False)].astype(int)
        severity = event_rng.uniform(.80, 1.45, len(sources))
        failed, causal, events = simulate_cascade(spec, event_rng, src, dst, edge_type,
                                                 dependency, edge_features, capacity, sources, severity)
        truth[ti] = failed
        edge_labels[ti] = causal
        for source, strength in zip(sources, severity):
            shocks.append({"snapshot": ti, "entity_id": int(source), "event_day": int(day + 1),
                           "severity": float(strength), "synthetic": True})
        for edge_id, stage in events:
            causal_events.append({"snapshot": ti, "relation_id": edge_id, "stage": stage,
                                  "event_day": int(day + stage * 10), "synthetic": True})
        # Imperfect pre-event reporting includes many false alarms. Its strength
        # and confidence do not expose future failures or successful graph edges.
        other = np.setdiff1d(np.arange(n), sources)
        false_count = budget - len(sources)
        weights = np.asarray([1.0, .55, .80, .25, .25])[node_type[other]]
        false_nodes = text_rng.choice(other, size=false_count, replace=false_count > len(other),
                                     p=weights / weights.sum())
        observed_nodes = np.r_[sources, false_nodes]
        candidates = []
        for position, entity in enumerate(observed_nodes):
            entity = int(entity)
            is_warning = position < len(sources)
            age = int(text_rng.integers(1, min(30, int(day - FIRST)) + 1))
            timestamp = int(day - age)
            if is_warning:
                grade = "severe" if severity[position] > 1.05 else "moderate"
                certainty = float(text_rng.uniform(.68, .97))
            else:
                grade = str(text_rng.choice(["minor", "moderate", "severe"], p=[.43, .45, .12]))
                certainty = float(text_rng.uniform(.28, .88))
            kind = str(text_rng.choice(["delivery_delay", "production_interruption", "inventory_shortage",
                                       "transport_constraint", "capacity_warning"]))
            signal_id = len(risk_records)
            text_id = len(risk_texts)
            text = (f"Synthetic bulletin {text_id}: {name}/{ENTITY_TYPES[node_type[entity]]}-{entity} "
                    f"reports {grade} {kind.replace('_', ' ')} for {spec.topic} operations on "
                    f"{iso_day(timestamp)}. This warning is unverified; alternative sourcing is under review.")
            record = {"signal_id": signal_id, "text_id": text_id, "entity_id": entity,
                      "timestamp_day": timestamp, "date": iso_day(timestamp), "confidence": certainty,
                      "risk_kind": kind, "severity": grade, "synthetic": True}
            risk_records.append(record)
            risk_texts.append({"text_id": text_id, "entity_id": entity, "timestamp_day": timestamp,
                               "date": iso_day(timestamp), "text": text,
                               "source_kind": "original_synthetic_template", "synthetic": True})
            signal_vectors.append(hash_text(text, signal_dim))
            if node_type[entity] in (0, 2):
                strength = {"minor": .25, "moderate": .60, "severe": 1.0}[grade]
                candidates.append((strength * certainty, entity))
        # This is an input-side selection; hidden simulator sources are not
        # passed to the model or the path extractor.
        selected = []
        for _, entity in sorted(candidates, reverse=True):
            if entity not in selected:
                selected.append(entity)
            if len(selected) == 6:
                break
        source_mask[ti, selected] = True

    signal_vectors = np.asarray(signal_vectors, dtype=np.float32)
    by_entity = [[] for _ in range(n)]
    for record in risk_records:
        by_entity[record["entity_id"]].append((record["timestamp_day"], record["signal_id"]))
    for entity, history in enumerate(by_entity):
        if not history:
            continue
        history.sort()
        timestamps = np.asarray([pair[0] for pair in history], dtype=np.int64)
        identifiers = np.asarray([pair[1] for pair in history], dtype=np.int64)
        for ti, day in enumerate(days):
            left = np.searchsorted(timestamps, day - 30, side="left")
            right = np.searchsorted(timestamps, day, side="left")
            selected = identifiers[max(left, right - retrieval_size):right][::-1]
            for slot, identifier in enumerate(selected):
                record = risk_records[int(identifier)]
                signals[ti, entity, slot] = signal_vectors[identifier]
                signal_mask[ti, entity, slot] = True
                delta[ti, entity, slot] = int(day - record["timestamp_day"])
                confidence[ti, entity, slot] = record["confidence"]
                memory_indices[ti, entity, slot] = int(identifier)

    eligible = np.flatnonzero(split >= 0)
    population = len(eligible) * n
    if spec.risk_labels > population:
        raise ValueError("More annotations requested than eligible node/query pairs")
    # Cover every entity type and entity once, then fill the remaining budget
    # with distinct uniformly selected node/query pairs. Neither step inspects
    # simulator outcomes, so rare positives are not artificially oversampled.
    chosen = {(int(annotation_rng.choice(eligible)), entity) for entity in range(n)}
    while len(chosen) < spec.risk_labels:
        chosen.add((int(annotation_rng.choice(eligible)), int(annotation_rng.integers(n))))
    annotations = []
    for label_id, (ti, entity) in enumerate(sorted(chosen)):
        labels[ti, entity] = truth[ti, entity]
        annotations.append({"label_id": label_id, "snapshot": ti, "entity_id": entity,
                            "prediction_day": int(days[ti]), "target_day": int(days[ti] + 30),
                            "label": int(truth[ti, entity]),
                            "label_kind": "independent_delayed_discrete_cascade", "synthetic": True})

    def entity_records():
        for entity, typ in enumerate(node_type):
            yield {"entity_id": entity, "entity_type": ENTITY_TYPES[int(typ)],
                   "name": f"{name}/{ENTITY_TYPES[int(typ)]}-{entity}", "synthetic": True}

    def relation_records():
        for relation_id, (u, v, typ) in enumerate(zip(src, dst, edge_type)):
            yield {"relation_id": relation_id, "src": int(u), "dst": int(v), "edge_type": int(typ),
                   "relation_type": RELATION_TYPES[int(typ)], "dependency": float(dependency[relation_id]),
                   "features": edge_features[relation_id].tolist(), "synthetic": True}

    def text_records():
        yield from risk_texts
        additional = spec.texts - len(risk_texts)
        if additional < n:
            raise ValueError("Routine-text budget must cover every entity")
        subjects = ("contract review", "inventory count", "transport planning", "supplier qualification",
                    "maintenance schedule", "production forecast", "capacity inspection", "sourcing review")
        for index in range(additional):
            text_id = len(risk_texts) + index
            entity = index if index < n else int(text_rng.integers(n))
            timestamp = FIRST if index == 0 else LAST if index == 1 else int(text_rng.integers(FIRST, LAST + 1))
            subject = subjects[index % len(subjects)]
            text = (f"Synthetic update {text_id}: {name}/{ENTITY_TYPES[node_type[entity]]}-{entity} "
                    f"records a {subject} on {iso_day(timestamp)} for {spec.topic} operations. "
                    "The report describes routine activity and does not confirm a disruption.")
            yield {"text_id": text_id, "entity_id": entity, "timestamp_day": timestamp,
                   "date": iso_day(timestamp), "text": text,
                   "source_kind": "original_synthetic_template", "synthetic": True}

    write_jsonl(output / "entities.jsonl", entity_records())
    write_jsonl(output / "relations.jsonl", relation_records())
    write_jsonl(output / "texts.jsonl", text_records())
    write_jsonl(output / "signals.jsonl", risk_records)
    write_jsonl(output / "risk_labels.jsonl", annotations)
    write_jsonl(output / "simulation_shocks.jsonl", shocks)
    write_jsonl(output / "simulation_events.jsonl", causal_events)
    write_jsonl(output / "queries.jsonl", ({"snapshot": ti, "prediction_day": int(day),
                "target_day": int(day + 30), "date": iso_day(day), "split": int(split[ti])}
                for ti, day in enumerate(days)))
    np.savez_compressed(output / "dataset.npz", node_features=node_features, node_type=node_type,
        src=src, dst=dst, edge_type=edge_type, edge_dependency=dependency, edge_features=edge_features,
        resilience=np.broadcast_to(capacity[None], (t, n, 5)), signals=signals, signal_mask=signal_mask,
        delta=delta, confidence=confidence, memory_indices=memory_indices, labels=labels,
        edge_labels=edge_labels, source_mask=source_mask, path_source_mask=source_mask,
        simulation_labels=truth, split=split, prediction_days=days, target_days=days + 30)
    stats = {"firms": spec.firms, "products": spec.products, "materials": spec.materials,
             "industries": spec.industries, "regions": spec.regions, "nodes": n,
             "relations": e, "texts": spec.texts, "risk_signals": len(risk_records),
             "risk_labels": len(annotations), "time_span": "2018--2025"}
    metadata = {"name": name, "data_kind": "synthetic", "is_real_world": False,
        "table1_size_matched": scale == 1, "scale": scale, "seed": actual_seed,
        "generator": "dmror.build_table_scale", "label_kind": "independent_delayed_discrete_cascade",
        "statistics": stats, "entity_types": dict(enumerate(ENTITY_TYPES)),
        "relation_types": dict(enumerate(RELATION_TYPES)), "query_count": t,
        "temporal_split": "60/20/20 chronology, cross-boundary 30-day targets purged",
        "forecast_horizon_days": 30, "memory_window_days": 30, "retrieval_size": retrieval_size,
        "signal_dim": signal_dim, "text_encoder": f"sha256_signed_hash_{signal_dim}_diagnostic_only",
        "memory_indices_reference": "signals.jsonl: signal_id, joined to texts.jsonl via text_id",
        "risk_label_count_definition": "known 0/1 entity-query annotations, not positive-only labels",
        "label_sampling": "one independent uniformly selected non-purged query per entity, plus distinct uniform entity-query samples to the budget; independent RNG stream",
        "unknown_labels": "-1 for every unannotated entity-query pair",
        "scope_ground_truth": "independent simulator failures, evaluated only at available annotations",
        "path_ground_truth": "simulator successful delivery edges; membership/location never cause delivery failure",
        "resilience_source": "independently sampled observed inventory/operational capacity",
        "source_mask_rule": "up to six firm/material warnings ranked by observed severity times confidence; no hidden shock IDs",
        "initial_month": "2018-01-01 has no pre-2018 evidence or simulated source disturbances",
        "known_label_positive": int((labels == 1).sum()), "known_label_negative": int((labels == 0).sum()),
        "annotation_statistics_by_split": {str(k): {
            "known": int((labels[split == k] >= 0).sum()),
            "positive": int((labels[split == k] == 1).sum()),
            "negative": int((labels[split == k] == 0).sum())} for k in (0, 1, 2)},
        "simulator_failed_counts": truth.sum(1).astype(int).tolist(),
        "simulator_causal_edge_counts": edge_labels.sum(1).astype(int).tolist(),
        "observed_source_counts": source_mask.sum(1).astype(int).tolist(),
        "split_counts": {str(k): int((split == k).sum()) for k in (-1, 0, 1, 2)},
        "limitations": ["Original SC-Auto/SC-Semi/SC-Energy observations and original table results were not recovered.",
                         "Matching aggregate sizes does not reproduce original topology, distributions, provenance or performance.",
                         "All names, texts, dates, capacities, events and labels are generated; none identify actual companies.",
                         "Hash text embeddings are diagnostic features, not embeddings from the paper's 8B LLM.",
                         "The simulator uses independent monthly scenarios, not a continuous historical replay."],
        "file_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--dataset", choices=tuple(SPECS) + ("SC-Auto", "SC-Semi", "SC-Energy"))
    parser.add_argument("--output", default="data/table_scale")
    parser.add_argument("--seed", type=int, help="Override the deterministic per-domain seed")
    args = parser.parse_args()
    names = tuple(SPECS) if args.all else (args.dataset if args.dataset.endswith("-Synthetic")
                                           else args.dataset + "-Synthetic",)
    for name in names:
        result = build(args.output, name, seed=args.seed)
        print(json.dumps({"dataset": name, "statistics": result["statistics"],
                          "known_label_positive": result["known_label_positive"],
                          "npz_sha256": result["file_sha256"]["dataset.npz"]}), flush=True)


if __name__ == "__main__":
    main()
