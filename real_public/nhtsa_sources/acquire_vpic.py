"""Low-frequency official paginated manufacturer registry collection; no VIN calls."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from acquire_nhtsa import ROOT, sha256

URL = "https://vpic.nhtsa.dot.gov/api/vehicles/GetAllManufacturers?format=json&page={page}"

def acquire(max_pages=100):
    raw = ROOT / "raw"
    output = ROOT / "normalized"
    output.mkdir(exist_ok=True)
    entries, manufacturers, duplicate_ids = [], {}, []
    complete = False
    started = datetime.now(timezone.utc).isoformat()
    for page in range(1, max_pages + 1):
        path = raw / f"vpic_manufacturers_page_{page:04d}.json"
        metadata_path = raw / (path.name + ".download.json")
        if path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sha256(path) != metadata["sha256"]:
                raise ValueError("Existing VPIC raw hash mismatch")
            response = json.loads(path.read_bytes())
        else:
            began = time.perf_counter()
            request = urllib.request.Request(URL.format(page=page), headers={
                "User-Agent": "ResearchDataAcquisition/1.0 (official public manufacturer registry; no VIN lookups)",
                "Accept-Encoding": "identity"})
            last_error = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=90) as remote:
                        data = remote.read()
                        metadata = {"requested_url": URL.format(page=page), "final_url": remote.url,
                                    "http_status": remote.status, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                    "response_headers": dict(remote.headers.items()), "local_file": "raw/" + path.name,
                                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    response = json.loads(data)
                    if not isinstance(response.get("Results"), list):
                        raise ValueError("API response did not contain Results list")
                    path.write_bytes(data)
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 3:
                        (ROOT / "vpic_failure.json").write_text(json.dumps({"page": page, "url": URL.format(page=page),
                                 "error_type": type(error).__name__, "error": str(error)}), encoding="utf-8")
                        raise
                    time.sleep(min(60, 10 * 2 ** attempt))
            time.sleep(max(0, 1.5 - (time.perf_counter() - began)))
        records = response["Results"]
        if response.get("Count") != len(records):
            raise ValueError("VPIC Count disagrees with Results")
        entries.append({**metadata, "page": page, "result_count": len(records)})
        for index, record in enumerate(records):
            identity = record["Mfr_ID"]
            normalized = {"manufacturer_id": f"vpic_mfr:{identity}", "vpic_mfr_id": identity,
                          "manufacturer_name": record["Mfr_Name"], "common_name": record.get("Mfr_CommonName"),
                          "country": record.get("Country"), "vehicle_types": record.get("VehicleTypes", []),
                          "source_fields": record,
                          "evidence": {"source_url": URL.format(page=page), "source_file": "raw/" + path.name,
                                       "source_sha256": metadata["sha256"], "source_page": page, "source_result_index": index,
                                       "retrieved_utc": metadata["retrieved_utc"]},
                          "temporal_scope": "current 2026 registry snapshot; no invented historical operating dates",
                          "recall_label": None}
            if identity in manufacturers:
                duplicate_ids.append({"mfr_id": identity, "first_page": manufacturers[identity]["evidence"]["source_page"],
                                      "repeat_page": page, "identical_source_fields": manufacturers[identity]["source_fields"] == record})
            else:
                manufacturers[identity] = normalized
        manifest = {"source_agency": "NHTSA vPIC", "started_utc": started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(), "status": "collecting",
                    "endpoint": URL, "page_size": 100, "vin_lookup_requests": 0,
                    "rate_policy": "sequential requests, at least 1.5 seconds between request starts, bounded retries",
                    "page_count": len(entries), "unique_manufacturers": len(manufacturers), "pages": entries,
                    "duplicate_ids": duplicate_ids}
        (ROOT / "vpic_collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if page % 10 == 0 or not records:
            print(json.dumps({"page": page, "unique_manufacturers": len(manufacturers), "last_page_count": len(records)}), flush=True)
        if not records:
            complete = True
            break
    countries, types = {}, {}
    with (output / "vpic_manufacturers_current.jsonl").open("w", encoding="utf-8", newline="\n") as target, \
            (output / "vpic_observed_relations_current.jsonl").open("w", encoding="utf-8", newline="\n") as relations:
        for _, record in sorted(manufacturers.items()):
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["country"]:
                country = record["country"]
                countries[country] = countries.get(country, 0) + 1
                relations.write(json.dumps({"source": record["manufacturer_id"], "relation": "registered_country",
                         "target_name": country, "evidence": record["evidence"]}, ensure_ascii=False) + "\n")
            for typ in record["vehicle_types"]:
                name = typ["Name"]
                types[name] = types.get(name, 0) + 1
                relations.write(json.dumps({"source": record["manufacturer_id"], "relation": "registered_vehicle_type",
                         "target_name": name, "is_primary": typ.get("IsPrimary"), "evidence": record["evidence"]}, ensure_ascii=False) + "\n")
    manifest.update(status="complete_empty_terminal_page" if complete else "incomplete_page_safety_cap",
                    completed_utc=datetime.now(timezone.utc).isoformat(), country_distribution=countries,
                    vehicle_type_distribution=types, historical_date_limitation="No historical validity intervals provided by this endpoint",
                    files={p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output.glob("vpic_*.jsonl")})
    (ROOT / "vpic_collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manufacturers": len(manufacturers), "pages": len(entries)}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=100)
    arguments = parser.parse_args()
    acquire(arguments.max_pages)
