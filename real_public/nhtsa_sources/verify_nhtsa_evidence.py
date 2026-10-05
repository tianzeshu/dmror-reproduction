"""Full normalized-to-official-byte evidence audit, including literal fact edges."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from acquire_nhtsa import ROOT, sha256
from normalize_nhtsa import FIELDS, dated
from build_nhtsa_fact_graph import identity

def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)

def product_id(fields):
    key = (fields["RCLTYPECD"].strip(), fields["MAKETXT"], fields["MODELTXT"], fields["YEARTXT"])
    return "nhtsa_product:" + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]

def verify():
    normalized = ROOT / "normalized"
    summary = json.loads((ROOT / "collection_summary.json").read_text(encoding="utf-8"))
    checked_files = {}
    for entry in json.loads((ROOT / "download_manifest.json").read_text()) ["sources"]:
        assert entry["http_status"] == 200
        assert sha256(ROOT / entry["local_file"]) == entry["sha256"]
        checked_files[entry["local_file"]] = entry["sha256"]
    raw_path = ROOT / "raw/FLAT_RCL_POST_2010.txt"
    assert sha256(raw_path) == summary["raw_text_sha256"]
    expected_raw_selected_ids = set()
    with raw_path.open("rb") as raw:
        for line in raw:
            values = line.decode("utf-8").rstrip("\r\n").split("\t")
            assert len(values) == 29
            date = dated(values[15])
            if date is not None and "2018-01-01" <= date <= "2025-12-31":
                expected_raw_selected_ids.add(values[0].strip())
    row_data, campaign_records, campaign_products = {}, defaultdict(list), defaultdict(set)
    checked_rows = 0
    with raw_path.open("rb") as raw:
        for row in records(normalized / "recall_rows_2018_2025.jsonl"):
            evidence = row["evidence"]
            raw.seek(evidence["source_byte_offset"])
            source_line = raw.read(evidence["source_line_bytes"])
            assert hashlib.sha256(source_line).hexdigest() == evidence["source_line_sha256"]
            source_fields = dict(zip(FIELDS, source_line.decode(evidence["source_encoding"]).rstrip("\r\n").split("\t")))
            assert source_fields == row["source_fields"]
            assert row["campaign_id"] == source_fields["CAMPNO"].strip()
            assert row["report_received_date"] == dated(source_fields["RCDATE"])
            assert "2018-01-01" <= row["report_received_date"] <= "2025-12-31"
            record_id = row["record_id"]
            assert record_id not in row_data
            row_data[record_id] = source_fields
            campaign_records[row["campaign_id"]].append(record_id)
            campaign_products[row["campaign_id"]].add(product_id(source_fields))
            checked_rows += 1
    assert set(row_data) == expected_raw_selected_ids
    event_count, campaigns = 0, set()
    required_field_map = {"campaign_id": "CAMPNO", "manufacturer": "MFGNAME", "component": "COMPNAME",
                          "summary": "DESC_DEFECT", "consequence": "CONEQUENCE_DEFECT", "remedy": "CORRECTIVE_ACTION"}
    for event in records(normalized / "recall_events_2018_2025.jsonl"):
        campaign = event["campaign_id"]
        assert campaign not in campaigns
        campaigns.add(campaign)
        assert event["source_record_ids"] == campaign_records[campaign]
        fields = row_data[event["source_record_ids"][0]]
        for key, source_field in required_field_map.items():
            assert event[key].strip() and event[key].strip() == fields[source_field].strip()
        assert event["event_date"] == dated(fields["RCDATE"])
        assert set(event["affected_products"]) == campaign_products[campaign]
        assert event["label_policy"] == "observed positive recall event only; absence is unknown"
        event_count += 1
    assert campaigns == set(campaign_records) and event_count == summary["unique_recall_campaigns"]
    nodes = {node["node_id"]: node for node in records(normalized / "recall_fact_nodes.jsonl")}
    edge_count, relation_counts = 0, Counter()
    for edge in records(normalized / "recall_fact_edges.jsonl"):
        assert edge["source"] in nodes and edge["target"] in nodes
        assert edge["is_synthetic"] is False
        for record_id in edge["source_record_ids"]:
            fields = row_data[record_id]
            relation = edge["relation"]
            campaign_id = "nhtsa_recall:" + fields["CAMPNO"].strip()
            expected = {
                "filed_official_recall": (identity("recall_mfr", fields["MFGNAME"]), campaign_id),
                "recalls_listed_product": (campaign_id, product_id(fields)),
                "identifies_component_descriptor": (campaign_id, identity("component_descriptor", fields["COMPNAME"])),
                "listed_manufacturer_for_recalled_product": (identity("listed_manufacturer", fields["MFGTXT"]), product_id(fields)),
            }[relation]
            assert expected == (edge["source"], edge["target"])
        edge_count += 1
        relation_counts[edge["relation"]] += 1
    vpic_checked = 0
    vpic_path = ROOT / "vpic_collection_manifest.json"
    if vpic_path.exists():
        vpic = json.loads(vpic_path.read_text())
        for page in vpic["pages"]:
            assert sha256(ROOT / page["local_file"]) == page["sha256"]
            response = json.loads((ROOT / page["local_file"]).read_bytes())
            assert len(response["Results"]) == response["Count"] == page["result_count"]
        for manufacturer in records(normalized / "vpic_manufacturers_current.jsonl"):
            ev = manufacturer["evidence"]
            response = json.loads((ROOT / ev["source_file"]).read_bytes())
            assert response["Results"][ev["source_result_index"]] == manufacturer["source_fields"]
            assert manufacturer["recall_label"] is None
            vpic_checked += 1
        assert vpic_checked == vpic["unique_manufacturers"]
    report = {"status": "passed", "official_raw_download_hashes_checked": checked_files,
              "selected_rows_checked_against_original_byte_offsets_and_all_29_fields": checked_rows,
              "selected_rows_complete_for_official_date_filter": checked_rows == len(expected_raw_selected_ids),
              "events_checked_against_actual_source_text": event_count,
              "fact_nodes": len(nodes), "fact_edges_all_supporting_records_checked": edge_count,
              "relation_counts": dict(relation_counts), "vpic_manufacturers_checked_against_official_page_results": vpic_checked,
              "synthetic_events": 0, "generated_summaries": 0, "invented_negatives": 0, "supply_or_causal_propagation_edges": 0,
              "scope": "official positive safety-recall facts; not a labelled supply-chain interruption benchmark"}
    (ROOT / "evidence_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    verify()
