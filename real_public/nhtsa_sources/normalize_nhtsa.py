"""Normalize official recall records; no synthetic events, labels or relationships."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from acquire_nhtsa import ROOT, sha256

FIELDS = ["RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME",
          "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA",
          "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID",
          "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE"]
TYPE_LABELS = {"V": "vehicle", "E": "equipment", "T": "tire", "C": "child_restraint"}
DATA_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip"

def dated(value):
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date().isoformat()
    except ValueError:
        return None

def decode_line(raw):
    for encoding in ("utf-8", "cp1252", "iso8859-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise AssertionError("lossless single-byte fallback must decode")

def number(value):
    value = value.strip()
    return int(value) if value.isdigit() else None

def jsonline(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

def normalize():
    output = ROOT / "normalized"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "raw/FLAT_RCL_POST_2010.txt"
    archive_hash = sha256(ROOT / "raw/FLAT_RCL_POST_2010.zip")
    source_hash = sha256(source)
    events, manufacturers, products, event_product_evidence = {}, defaultdict(list), {}, {}
    raw_rows = selected_rows = byte_offset = 0
    raw_dates, years, encodings, types = Counter(), Counter(), Counter(), Counter()
    invalid_records = []
    source_record_ids, row_missing = set(), Counter()
    campaign_values = defaultdict(lambda: defaultdict(set))
    per_year_campaigns = defaultdict(set)
    rows_path = output / "recall_rows_2018_2025.jsonl"
    csv_path = output / "recall_rows_2018_2025.csv"
    with source.open("rb") as stream, rows_path.open("w", encoding="utf-8", newline="\n") as rows_out, \
            csv_path.open("w", encoding="utf-8", newline="") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=["source_line_number", "source_byte_offset", "source_line_sha256"] + FIELDS)
        writer.writeheader()
        for raw_rows, raw_line in enumerate(stream, start=1):
            offset = byte_offset
            byte_offset += len(raw_line)
            text, encoding = decode_line(raw_line)
            fields = text.rstrip("\r\n").split("\t")
            if len(fields) != len(FIELDS):
                invalid_records.append({"source_line_number": raw_rows, "source_byte_offset": offset,
                                        "field_count": len(fields), "reason": "field_count_not_29"})
                continue
            record = dict(zip(FIELDS, fields))
            received = dated(record["RCDATE"])
            if received is None:
                invalid_records.append({"source_line_number": raw_rows, "source_byte_offset": offset,
                                        "RCDATE": record["RCDATE"], "reason": "invalid_or_missing_report_received_date"})
                continue
            raw_dates[received[:4]] += 1
            if not "2018-01-01" <= received <= "2025-12-31":
                continue
            selected_rows += 1
            encodings[encoding] += 1
            campaign = record["CAMPNO"].strip()
            typ = record["RCLTYPECD"].strip()
            row_hash = hashlib.sha256(raw_line).hexdigest()
            evidence = {"source_url": DATA_URL, "raw_zip_sha256": archive_hash,
                        "raw_text_file": "raw/FLAT_RCL_POST_2010.txt", "raw_text_sha256": source_hash,
                        "source_line_number": raw_rows, "source_byte_offset": offset, "source_line_bytes": len(raw_line),
                        "source_line_sha256": row_hash, "source_encoding": encoding}
            row = {"source": "NHTSA_ODI_recall_flatfile", "record_id": record["RECORD_ID"].strip(),
                   "campaign_id": campaign, "report_received_date": received,
                   "recall_type_code": typ, "recall_type": TYPE_LABELS.get(typ, "unknown_code"),
                   "manufacturer": record["MFGNAME"].strip(), "make": record["MAKETXT"].strip(),
                   "model": record["MODELTXT"].strip(), "model_year": (None if record["YEARTXT"].strip() == "9999" else number(record["YEARTXT"])),
                   "component": record["COMPNAME"].strip(), "source_fields": record, "evidence": evidence}
            jsonline(rows_out, row)
            writer.writerow({"source_line_number": raw_rows, "source_byte_offset": offset,
                             "source_line_sha256": row_hash, **record})
            source_record_ids.add(row["record_id"])
            years[received[:4]] += 1
            types[typ] += 1
            per_year_campaigns[received[:4]].add(campaign)
            for name in ("CAMPNO", "MFGNAME", "COMPNAME", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION"):
                if not record[name].strip():
                    row_missing[name] += 1
            for name in ("RCDATE", "RCLTYPECD", "MFGNAME", "MFGTXT", "COMPNAME", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION"):
                campaign_values[campaign][name].add(record[name])
            if campaign not in events:
                events[campaign] = {"event_id": "nhtsa_recall:" + campaign, "campaign_id": campaign,
                      "event_type": "official_safety_recall", "event_date": received,
                      "event_date_definition": "official Part 573 report received date (RCDATE)",
                      "manufacturer": record["MFGNAME"].strip(),
                      "listed_manufacturer": record["MFGTXT"].strip(),
                      "component": record["COMPNAME"].strip(),
                      "summary": record["DESC_DEFECT"], "consequence": record["CONEQUENCE_DEFECT"],
                      "remedy": record["CORRECTIVE_ACTION"], "notes": record["NOTES"],
                      "recall_type_code": typ, "recall_type": TYPE_LABELS.get(typ, "unknown_code"),
                      "potential_units_affected": number(record["POTAFF"]),
                      "owner_notification_date": dated(record["ODATE"]),
                      "record_creation_date": dated(record["DATEA"]),
                      "manufacturer_campaign_number": record["MFGCAMPNO"],
                      "official_campaign_url": "https://www.nhtsa.gov/recalls?nhtsaId=" + campaign,
                      "official_campaign_api_url": "https://api.nhtsa.gov/recalls/campaignNumber?campaignNumber=" + campaign,
                      "canonical_fields_policy": "first source row; all observed field variants and row IDs retained",
                      "source_record_ids": [], "source_line_numbers": [], "affected_products": [],
                      "evidence": evidence, "label_policy": "observed positive recall event only; absence is unknown"}
            event = events[campaign]
            event["source_record_ids"].append(row["record_id"])
            event["source_line_numbers"].append(raw_rows)
            product_key = (typ, record["MAKETXT"], record["MODELTXT"], record["YEARTXT"])
            product_id = "nhtsa_product:" + hashlib.sha256(json.dumps(product_key, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
            if product_id not in products:
                products[product_id] = {"product_id": product_id, "recall_type_code": typ,
                     "make": record["MAKETXT"], "model": record["MODELTXT"], "model_year_raw": record["YEARTXT"],
                     "source_record_ids": [], "evidence": evidence}
            products[product_id]["source_record_ids"].append(row["record_id"])
            if product_id not in event["affected_products"]:
                event["affected_products"].append(product_id)
            event_product_evidence.setdefault((campaign, product_id), (row["record_id"], evidence))
            manufacturers[record["MFGNAME"]].append((campaign, row["record_id"], evidence))
    with (output / "recall_events_2018_2025.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for campaign, event in sorted(events.items()):
            event["source_field_variants"] = {key: sorted(value) for key, value in campaign_values[campaign].items() if len(value) > 1}
            jsonline(target, event)
    with (output / "affected_products_2018_2025.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for _, product in sorted(products.items()):
            jsonline(target, product)
    with (output / "recall_manufacturers_2018_2025.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for name, values in sorted(manufacturers.items()):
            jsonline(target, {"manufacturer_id": "nhtsa_recall_mfr:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24],
                  "manufacturer_name": name, "name_policy": "literal official MFGNAME; no entity-resolution inference",
                  "campaign_ids": sorted({value[0] for value in values}), "source_record_ids": [value[1] for value in values],
                  "evidence": values[0][2]})
    with (output / "observed_relations_2018_2025.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for name, values in sorted(manufacturers.items()):
            manufacturer_id = "nhtsa_recall_mfr:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
            seen = set()
            for campaign, record_id, evidence in values:
                if campaign not in seen:
                    jsonline(target, {"source": manufacturer_id, "relation": "filed_official_recall", "target": "nhtsa_recall:" + campaign,
                                      "campaign_id": campaign, "source_record_id": record_id, "evidence": evidence})
                    seen.add(campaign)
        for _, event in sorted(events.items()):
            for product_id in event["affected_products"]:
                record_id, evidence = event_product_evidence[(event["campaign_id"], product_id)]
                jsonline(target, {"source": event["event_id"], "relation": "recalls_listed_product", "target": product_id,
                                  "campaign_id": event["campaign_id"], "source_record_id": record_id, "evidence": evidence})
    event_types = Counter(event["recall_type_code"] for event in events.values())
    missing_events = {key: sum(not event[key].strip() for event in events.values())
                      for key in ("campaign_id", "manufacturer", "component", "summary", "consequence", "remedy")}
    summary = {"source_agency": "NHTSA, U.S. Department of Transportation", "data_kind": "actual_official_recall_records",
          "synthetic_records": 0, "generated_summaries": 0, "collection_snapshot_utc": datetime.now(timezone.utc).isoformat(),
          "date_filter_field": "RCDATE (report received date), not vehicle model year", "date_range": ["2018-01-01", "2025-12-31"],
          "raw_rows_total": raw_rows, "raw_report_year_row_distribution": dict(sorted(raw_dates.items())),
          "selected_rows": selected_rows, "unique_source_record_ids": len(source_record_ids),
          "unique_recall_campaigns": len(events), "literal_manufacturer_names": len(manufacturers),
          "distinct_make_model_year_type_products": len(products),
          "selected_row_year_distribution": dict(sorted(years.items())),
          "campaign_year_distribution": {year: len(values) for year, values in sorted(per_year_campaigns.items())},
          "selected_row_recall_type_distribution": dict(types), "campaign_recall_type_distribution": dict(event_types),
          "recall_type_meanings": TYPE_LABELS, "source_text_encoding_counts": dict(encodings),
          "missing_required_fields_rows": dict(row_missing), "missing_required_fields_events": missing_events,
          "invalid_or_unparseable_raw_records": len(invalid_records),
          "dataset_scope": "U.S. safety recalls, mixed vehicles/equipment/tires/child restraints; not supply-chain transactions",
          "label_policy": "only official observed positive recall events; no invented negative labels or causal propagation edges",
          "temporal_limitation": "2026 retrieval snapshot with historical received dates; narrative fields may contain later updates and are not guaranteed as-of historical dates",
          "raw_zip_sha256": archive_hash, "raw_text_sha256": source_hash,
          "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.glob("*")) if p.is_file()}}
    (ROOT / "collection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "invalid_raw_records.json").write_text(json.dumps(invalid_records, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    normalize()
