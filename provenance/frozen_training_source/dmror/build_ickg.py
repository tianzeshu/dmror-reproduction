"""Build a retrospective, weakly labelled ICKG publication-forecast dataset.

This is NOT a disruption gold standard. The labels mean that a linked public
text snippet in the next 30 days contains a fixed risk-related expression.
The graph and the entity canonicalisation are retrospective source artefacts;
their provenance limitations are recorded in metadata, not silently hidden.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import re

import numpy as np

from .data import audit_dataset, hash_text, sha256, write_json


SOURCE_FILES = {
    "canonical_entities_merged.jsonl": "/root/ICKG/phase03c_entity_canonical_consolidation/output/canonical_entities_merged.jsonl",
    "kg_edges_high_precision.jsonl": "/root/ICKG/phase05_build_icr_snapshot/output/kg_edges_high_precision.jsonl",
    "contexts.jsonl": "/root/ICKG/phase06_context_repository/output/contexts.jsonl",
    "context_entity_links.jsonl": "/root/ICKG/phase06_context_repository/output/context_entity_links.jsonl",
}
EXPECTED_SHA256 = {
    "kg_edges_high_precision.jsonl": "3ad550641dd590b490c47c194cd243db027be843aced1e25d358af1da8269cbf",
    "contexts.jsonl": "4109e5df303e1edb9562d74fc518bdc6580463f3a0c5bf4735f97d25c3af0d91",
    "context_entity_links.jsonl": "f64b5466ff4ee007028f2f72091d69f74e6725f4bbdb99f9859df0fbcfb29732",
    "canonical_entities_merged.jsonl": "c246f1afbb8be6e6c956c1ed6b97467ba7d420b8ea022fc7af4da0483d5aa8a4",
}
RELATION_TYPES = ["Supplies", "Cooperates", "InvestsIn", "Holding", "Acquires", "Competes", "JointVenture"]
ENTITY_TYPES = ["Company", "Product", "Material", "Industry", "Region"]
# A fixed, intentionally literal publication rule. Negation, hypothetical
# discussion and which mentioned company actually experienced the event are
# NOT adjudicated; these are documented weak-label limitations.
RISK_TERMS = [
    "断供", "供应中断", "停产", "减产", "破产", "违约", "短缺", "缺货", "制裁",
    "出口管制", "出口限制", "禁运", "裁员", "火灾", "罢工", "供应链中断",
    "交付延迟", "供应不足", "产能不足", "暂停生产", "关闭工厂", "生产中断", "限电",
    "chip shortage", "supply disruption", "export control", "sanction", "bankrupt",
    "layoff", "production halt",
]
RULE_VERSION = "ickg-risk-publication-literal-v1"
RISK_PATTERN = re.compile("|".join(re.escape(term) for term in RISK_TERMS), re.IGNORECASE)
EPOCH = date(1970, 1, 1)


def parse_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def day_number(value):
    return (value - EPOCH).days


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_extraction_demo(output_root, entities, node_ids, dedup_links):
    """A bounded, deterministic local-LLM extraction demonstration, not labels."""
    demo = Path(output_root) / "llm_extraction_example"
    demo.mkdir(parents=True, exist_ok=True)
    node_ids = set(node_ids)
    grouped = {}
    for (entity_id, context_id), (timestamp, context) in dedup_links.items():
        if entity_id not in node_ids or timestamp.year != 2022:
            continue
        if context_id not in grouped:
            grouped[context_id] = {"document_id": context_id, "source_document_id": context["doc_id"],
                                   "text": context["text"], "timestamp": timestamp.isoformat(),
                                   "url": context.get("url", ""), "entities": []}
        grouped[context_id]["entities"].append({"id": entity_id, "name": entities[entity_id]["canonical_name"]})
    records = sorted(grouped.values(), key=lambda row: (row["timestamp"], row["document_id"]))
    risk = [row for row in records if RISK_PATTERN.search(row["text"])]
    neutral = [row for row in records if not RISK_PATTERN.search(row["text"])]
    # Sampling categories never go into the model's extraction input. They
    # are a reproducible sampling rule, not an adjudicated extraction target.
    risk = risk[:16]
    indices = np.linspace(0, len(neutral) - 1, min(16, len(neutral)), dtype=int)
    neutral = [neutral[int(index)] for index in indices]
    sample = risk + neutral
    for row in sample:
        row["entities"].sort(key=lambda entity: entity["id"])
    write_jsonl(demo / "input.jsonl", sample)
    write_json(demo / "manifest.json", {
        "purpose": "bounded frozen-local-LLM structured extraction demonstration on train-period snippets",
        "input_count": len(sample), "literal_risk_term_snippets": len(risk), "no_literal_risk_term_snippets": len(neutral),
        "selection_dates": "2022 only", "selection_rule": "up to 16 first risk matching contexts plus 16 evenly spaced nonmatching contexts",
        "source": "ICKG contexts linked to selected frozen-graph entities", "has_adjudicated_gold_labels": False,
        "used_as_training_future_labels": False, "risk_term_rule_version": RULE_VERSION,
        "limitations": ["Nonmatching text is not an operational safety determination.", "Risk-matching text may be hypothetical or negated.", "Extraction outputs are current textual evidence; never reuse them as future outcomes."],
        "sha256": sha256(demo / "input.jsonl"),
    })
    return len(sample)


def build(raw, output, retrieval_size=5, lookback_days=30, horizon_days=30, step_days=10):
    if min(retrieval_size, lookback_days, horizon_days, step_days) <= 0:
        raise ValueError("Retrieval size and all time windows/steps must be positive")
    raw, output = Path(raw), Path(output) / "ICKG-Weak"
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for filename, server_path in SOURCE_FILES.items():
        digest = sha256(raw / filename)
        if digest != EXPECTED_SHA256[filename]:
            raise ValueError(f"Source SHA-256 mismatch for {filename}; audit a new source before building")
        provenance.append({"filename": filename, "server_path": server_path, "sha256": digest})
    entities = {row["entity_id"]: row for row in read_jsonl(raw / "canonical_entities_merged.jsonl")}
    raw_edges = read_jsonl(raw / "kg_edges_high_precision.jsonl")
    contexts = {row["context_id"]: row for row in read_jsonl(raw / "contexts.jsonl")}
    links = read_jsonl(raw / "context_entity_links.jsonl")
    first_query = date(2022, 1, 1)
    train_boundary, validation_boundary = date(2023, 1, 1), date(2024, 1, 1)
    source_end = date(2025, 8, 30)

    # Only first_event_date is consulted; never use last_event_date, evidence
    # counts, aggregated confidence, country, sector or final node statistics.
    frozen_edges = [row for row in raw_edges if
                    parse_date(row.get("first_event_date")) is not None and
                    parse_date(row["first_event_date"]) < first_query]
    graph_entities = {row[field] for row in frozen_edges for field in ("head_entity_id", "tail_entity_id")}
    eligible = set()
    dedup_links = {}
    rejected_date_links = 0
    for link in links:
        context = contexts.get(link["context_id"])
        if context is None or link["entity_id"] not in graph_entities:
            continue
        timestamp = parse_date(context.get("context_date"))
        if timestamp is None or timestamp != parse_date(link.get("context_date")):
            rejected_date_links += 1
            continue
        if timestamp < first_query:
            eligible.add(link["entity_id"])
        if timestamp <= source_end and context.get("text", "").strip():
            dedup_links[(link["entity_id"], context["context_id"])] = (timestamp, context)
    # Eligibility uses pre-query availability only, not future positive labels.
    node_ids = sorted(graph_entities & eligible)
    if not 100 <= len(node_ids) <= 250:
        raise ValueError(f"Expected an auditable 100-250 node frozen graph; got {len(node_ids)}")
    node_index = {entity_id: index for index, entity_id in enumerate(node_ids)}
    frozen_edges = sorted((row for row in frozen_edges if row["head_entity_id"] in node_index and row["tail_entity_id"] in node_index), key=lambda row: row["edge_id"])
    n, e = len(node_ids), len(frozen_edges)
    src = np.array([node_index[row["head_entity_id"]] for row in frozen_edges], np.int64)
    dst = np.array([node_index[row["tail_entity_id"]] for row in frozen_edges], np.int64)
    edge_type = np.array([RELATION_TYPES.index(row["relation_type"]) for row in frozen_edges], np.int64)
    node_type = np.array([ENTITY_TYPES.index(entities[entity_id]["entity_type"]) for entity_id in node_ids], np.int64)
    node_features = np.zeros((n, 8), np.float32)
    node_features[np.arange(n), node_type] = 1
    # Missing real-world attributes are explicit placeholders, not estimators
    # disguised as observations. Their observed masks are all false.
    edge_features = np.zeros((e, 4), np.float32)
    edge_dependency = np.ones(e, np.float32)

    by_node = defaultdict(list)
    events_by_node = defaultdict(list)
    for (entity_id, _), (timestamp, context) in dedup_links.items():
        if entity_id not in node_index:
            continue
        by_node[entity_id].append((timestamp, context))
        matches = list(RISK_PATTERN.finditer(context["text"]))
        if matches:
            events_by_node[entity_id].append((timestamp, context, matches))
    for records in by_node.values():
        records.sort(key=lambda value: (value[0], value[1]["context_id"]))

    queries = []
    query = first_query
    while query + timedelta(days=horizon_days) <= source_end:
        queries.append(query)
        query += timedelta(days=step_days)
    days = np.array([day_number(value) for value in queries], np.int64)
    t = len(queries)
    split = np.array([0 if value < train_boundary else 1 if value < validation_boundary else 2 for value in queries], np.int64)
    for index, value in enumerate(queries):
        boundary = train_boundary if split[index] == 0 else validation_boundary if split[index] == 1 else None
        if boundary is not None and value + timedelta(days=horizon_days) >= boundary:
            split[index] = -1
    signals = np.zeros((t, n, retrieval_size, 64), np.float32)
    signal_mask = np.zeros((t, n, retrieval_size), bool)
    delta = np.zeros((t, n, retrieval_size), np.float32)
    confidence = np.zeros_like(delta)
    memory_indices = np.full((t, n, retrieval_size), -1, np.int64)
    labels = np.zeros((t, n), np.float32)
    text_records, observations, evidence = [], [], []
    signal_lookup = {}

    for ti, query in enumerate(queries):
        target = query + timedelta(days=horizon_days)
        for ni, entity_id in enumerate(node_ids):
            # Open past interval (t-lookback,t); publication dates only, and no
            # label/evidence aggregates enter ranking or the encoder input.
            candidates = [(timestamp, context) for timestamp, context in by_node[entity_id]
                          if query - timedelta(days=lookback_days) < timestamp < query]
            candidates.sort(key=lambda value: (value[0], value[1]["context_id"]), reverse=True)
            for ki, (timestamp, context) in enumerate(candidates[:retrieval_size]):
                key = (entity_id, context["context_id"])
                if key not in signal_lookup:
                    sid = len(text_records)
                    signal_lookup[key] = sid
                    text_records.append({
                        "signal_id": sid, "text": context["text"], "tau": timestamp.isoformat(),
                        "timestamp_day": day_number(timestamp), "entity_id": entity_id,
                        "node_index": ni, "context_id": context["context_id"], "doc_id": context["doc_id"],
                        "url": context.get("url", ""), "title": context.get("title", ""),
                        "source_kind": "retrospective_publication_context", "confidence": 0.8,
                    })
                sid = signal_lookup[key]
                memory_indices[ti, ni, ki] = sid
                signals[ti, ni, ki] = hash_text(context["text"], 64)
                signal_mask[ti, ni, ki] = True
                delta[ti, ni, ki] = (query - timestamp).days
                confidence[ti, ni, ki] = 0.8
                observations.append({"prediction_day": day_number(query), "entity_id": entity_id, "node_index": ni,
                                     "signal_id": sid, "timestamp_day": day_number(timestamp), "confidence": 0.8})
            # Future interval [t,t+horizon); no recorded matching snippet is 0
            # for this publication rule, and never means operational safety.
            seen_evidence = set()
            for timestamp, context, matches in events_by_node[entity_id]:
                if not query <= timestamp < target:
                    continue
                labels[ti, ni] = 1
                # Several windows from one document may quote the same span;
                # labels are binary, while supporting evidence is deduplicated.
                for match in matches:
                    key = (context["doc_id"], int(context.get("char_start", 0)) + match.start(), match.group(0))
                    if key in seen_evidence:
                        continue
                    seen_evidence.add(key)
                    evidence.append({
                        "prediction_day": day_number(query), "prediction_date": query.isoformat(),
                        "target_end_exclusive": target.isoformat(), "split": int(split[ti]),
                        "node_index": ni, "entity_id": entity_id, "date": timestamp.isoformat(),
                        "doc_id": context["doc_id"], "context_id": context["context_id"],
                        "url": context.get("url", ""), "title": context.get("title", ""),
                        "matched_phrase": match.group(0), "context_span": [match.start(), match.end()],
                        "document_span": [int(context.get("char_start", 0)) + match.start(), int(context.get("char_start", 0)) + match.end()],
                        "context_text": context["text"], "rule_version": RULE_VERSION,
                        "weak_evidence_only": True,
                    })

    np.savez_compressed(output / "dataset.npz", node_features=node_features, node_type=node_type,
                        src=src, dst=dst, edge_type=edge_type, edge_dependency=edge_dependency,
                        edge_features=edge_features, resilience=np.zeros((t, n, 5), np.float32),
                        resilience_observed_mask=np.zeros((t, n, 5), bool),
                        edge_features_observed_mask=np.zeros((e, 4), bool),
                        edge_dependency_observed_mask=np.zeros(e, bool),
                        signals=signals, signal_mask=signal_mask, delta=delta, confidence=confidence,
                        memory_indices=memory_indices, labels=labels, edge_labels=np.full((t, e), -1, np.float32),
                        source_mask=np.zeros((t, n), bool), split=split, prediction_days=days,
                        target_days=days + horizon_days)
    write_jsonl(output / "signals.jsonl", text_records)
    write_jsonl(output / "signal_observations.jsonl", observations)
    write_jsonl(output / "label_evidence.jsonl", evidence)
    write_jsonl(output / "nodes.jsonl", [{"node_index": index, "entity_id": entity_id,
                                         "display_name": entities[entity_id]["canonical_name"],
                                         "entity_type": entities[entity_id]["entity_type"]}
                                        for index, entity_id in enumerate(node_ids)])
    write_jsonl(output / "edges.jsonl", [{"edge_index": index, "edge_id": row["edge_id"],
                                         "src": int(src[index]), "dst": int(dst[index]),
                                         "relation_type": row["relation_type"],
                                         "first_event_date": row["first_event_date"]}
                                        for index, row in enumerate(frozen_edges)])
    statistics = audit_dataset(output / "dataset.npz")
    statistics["label_counts_by_split"] = {str(value): {"snapshots": int((split == value).sum()),
                                                         "positive": int(labels[split == value].sum()),
                                                         "negative": int((labels[split == value] == 0).sum())}
                                            for value in (-1, 0, 1, 2)}
    statistics["retrieved_signals"] = len(text_records)
    statistics["retrieval_observations"] = len(observations)
    statistics["label_evidence_records"] = len(evidence)
    statistics["rejected_inconsistent_date_links"] = rejected_date_links
    # Independently compare serialized-memory identities against dates and
    # positive-label evidence, beyond the generic tensor-shape audit.
    for record in observations:
        signal = text_records[record["signal_id"]]
        assert 0 < record["prediction_day"] - signal["timestamp_day"] < lookback_days
        assert record["entity_id"] == signal["entity_id"]
    positive_cells = {(int(row["prediction_day"]), int(row["node_index"])) for row in evidence}
    expected_cells = {(int(days[ti]), int(ni)) for ti, ni in zip(*np.where(labels == 1))}
    assert positive_cells == expected_cells, "Every positive weak label must have future snippet evidence"
    for row in evidence:
        assert parse_date(row["prediction_date"]) <= parse_date(row["date"]) < parse_date(row["target_end_exclusive"])
        assert row["entity_id"] == node_ids[row["node_index"]]
    assert all(parse_date(row["first_event_date"]) < first_query for row in frozen_edges)
    statistics["source_temporal_audit"] = {
        "all_retrieved_signals_strictly_before_query": True,
        "all_retrieved_signals_inside_open_lookback": True,
        "all_positive_labels_have_matching_future_evidence": True,
        "all_graph_first_event_dates_before_first_query": True,
        "cross_split_target_windows_purged": True,
        "retrospective_source_selection_bias_eliminated": False,
    }
    metadata = {
        "name": "ICKG-Weak", "data_kind": "real_source_weak_labels", "is_real_world": True,
        "is_disruption_gold_standard": False, "generator": "dmror.build_ickg", "source_provenance": provenance,
        "source_counts": {"entities": len(entities), "high_precision_edges": len(raw_edges),
                          "contexts": len(contexts), "links": len(links)},
        "graph_cutoff_exclusive": first_query.isoformat(), "node_selection": "linked context existed before first query",
        "first_query_date": queries[0].isoformat(), "last_query_date": queries[-1].isoformat(),
        "source_date_coverage_end": source_end.isoformat(), "forecast_horizon_days": horizon_days,
        "query_step_days": step_days, "retrieval_size": retrieval_size, "lookback_days": lookback_days,
        "retrieval_rule": "strict open past interval, constant 0.8 confidence, descending publication date; context_id tie break",
        "temporal_split": "2022 train, 2023 validation, 2024-2025 test; crossing or touching target boundary purged",
        "signal_timestamp_kind": "reported publication date; contemporaneous crawl availability not established",
        "text_encoder": "sha256_signed_hash_64_diagnostic_only; replace through dmror.encode_signals for frozen LLM runs",
        "node_feature_semantics": "type one-hot plus three zero placeholders; names/country/sector/final counts excluded",
        "relation_types": RELATION_TYPES, "entity_types": ENTITY_TYPES,
        "edge_dependency": "unobserved constant 1; observed mask false", "edge_features": "unobserved zero; observed mask false",
        "resilience_source": "unobserved zeros; resilience_observed_mask false",
        "label_kind": "future_30_day_linked_snippet_literal_risk_expression_publication",
        "label_rule": {"version": RULE_VERSION, "phrases": RISK_TERMS, "positive": "linked snippet dated in [query,query+30) matches a literal risk term",
                       "negative": "no such matched linked snippet recorded in this corpus; does not imply actual safety"},
        "path_ground_truth": "unknown; edge_labels all -1; source_mask all false; causal path precision unavailable",
        "scope_ground_truth": "publication-rule node set only; not measured physical affected entities",
        "statistics": statistics,
        "limitations": [
            "This is not the paper SC-Auto, SC-Semi or SC-Energy dataset and does not establish real disruption accuracy.",
            "High-precision edge inclusion and canonical entity consolidation were produced retrospectively using complete-source evidence; even date freezing cannot remove that selection bias.",
            "Publication dates are not proven historical ingestion/availability dates; this is retrospective evaluation, not a certified online backtest.",
            "Literal weak labels include hypothetical and negated risk discussion; entity linking and relation correctness are not human gold standards.",
            "No observed resilience, dependency, edge attributes, source-cause labels, affected-set gold truth or causal-path truth were found.",
            "Thirty-day target windows overlap across ten-day queries; snapshot samples are correlated and seed variability is not a population confidence interval.",
            "The corpus has collection/coverage bias and severe class imbalance; a zero label records absence of matching captured text only.",
            "Raw public webpage text redistribution licences were not independently verified; source text is not licensed as this project's original code.",
        ],
    }
    metadata["extraction_demo_input_count"] = build_extraction_demo(Path(output).parent.parent, entities, node_ids, dedup_links)
    write_json(output / "metadata.json", metadata)
    write_json(output / "audit.json", statistics)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/ickg")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--retrieval-size", type=int, default=5)
    args = parser.parse_args()
    result = build(args.raw, args.output, retrieval_size=args.retrieval_size)
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
