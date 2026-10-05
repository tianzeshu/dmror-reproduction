"""Build literal official recall facts; no supply relations or negative labels."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from acquire_nhtsa import ROOT, sha256

def identity(kind, value):
    return "nhtsa_" + kind + ":" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

def write_line(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

def build():
    normalized = ROOT / "normalized"
    nodes, edges = {}, {}
    for row in map(json.loads, (normalized / "recall_rows_2018_2025.jsonl").open(encoding="utf-8")):
        fields, evidence = row["source_fields"], row["evidence"]
        campaign_id = "nhtsa_recall:" + row["campaign_id"]
        reporter_id = identity("recall_mfr", fields["MFGNAME"])
        manufacturer_id = identity("listed_manufacturer", fields["MFGTXT"])
        product_key = (fields["RCLTYPECD"].strip(), fields["MAKETXT"], fields["MODELTXT"], fields["YEARTXT"])
        product_id = "nhtsa_product:" + hashlib.sha256(json.dumps(product_key, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        component_id = identity("component_descriptor", fields["COMPNAME"])
        values = [
            (reporter_id, "recall_reporting_manufacturer", fields["MFGNAME"], {"source_field": "MFGNAME"}),
            (manufacturer_id, "listed_recalled_product_manufacturer", fields["MFGTXT"], {"source_field": "MFGTXT"}),
            (campaign_id, "official_recall_campaign", row["campaign_id"], {"event_date": row["report_received_date"]}),
            (product_id, "recalled_make_model_year_product", fields["MAKETXT"] + "/" + fields["MODELTXT"] + "/" + fields["YEARTXT"],
             {"make": fields["MAKETXT"], "model": fields["MODELTXT"], "model_year_raw": fields["YEARTXT"], "recall_type_code": fields["RCLTYPECD"]}),
            (component_id, "official_component_descriptor", fields["COMPNAME"], {"source_field": "COMPNAME"}),
        ]
        for node_id, typ, name, attributes in values:
            if node_id not in nodes:
                nodes[node_id] = {"node_id": node_id, "node_type": typ, "name": name, **attributes,
                                  "evidence": evidence, "is_synthetic": False}
        relations = [(reporter_id, "filed_official_recall", campaign_id),
                     (campaign_id, "recalls_listed_product", product_id),
                     (campaign_id, "identifies_component_descriptor", component_id),
                     (manufacturer_id, "listed_manufacturer_for_recalled_product", product_id)]
        for src, relation, dst in relations:
            key = src, relation, dst
            if key not in edges:
                edges[key] = {"source": src, "relation": relation, "target": dst, "source_record_ids": [],
                              "supporting_campaign_ids": [], "evidence": evidence, "is_synthetic": False}
            edges[key]["source_record_ids"].append(row["record_id"])
            if row["campaign_id"] not in edges[key]["supporting_campaign_ids"]:
                edges[key]["supporting_campaign_ids"].append(row["campaign_id"])
    with (normalized / "recall_fact_nodes.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for _, value in sorted(nodes.items()):
            write_line(out, value)
    with (normalized / "recall_fact_edges.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for _, value in sorted(edges.items()):
            write_line(out, value)
    report = {"data_kind": "official_recall_fact_graph", "synthetic_nodes": 0, "synthetic_edges": 0,
              "node_count": len(nodes), "edge_count": len(edges), "node_types": dict(Counter(n["node_type"] for n in nodes.values())),
              "relation_types": dict(Counter(e["relation"] for e in edges.values())),
              "product_definition": "distinct literal (recall type, make, model, model year); 9999 denotes unknown/not-applicable year",
              "manufacturer_identity": "literal official filer MFGNAME and listed manufacturer MFGTXT retained as separate source roles; no inferred aliases or parents",
              "component_definition": "literal official COMPNAME descriptor; not an inferred material or physical supplier",
              "edge_semantics": "only source-listed recall/product/manufacturer/component associations; no supply dependencies or risk-propagation truth",
              "historical_limitation": "2026 source snapshot; report received dates 2018-2025; not guaranteed contemporaneous point-in-time text",
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in normalized.glob("recall_fact_*.jsonl")}}
    (ROOT / "recall_fact_graph_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    build()
