"""Download official NHTSA bulk sources, preserving response and byte provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "FLAT_RCL_POST_2010.zip": "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip",
    "RCL.txt": "https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt",
    "nhtsa_datasets_and_apis.html": "https://www.nhtsa.gov/nhtsa-datasets-and-apis",
    "nhtsa_terms_use.html": "https://www.nhtsa.gov/about-nhtsa/terms-use",
    "data_gov_recall_metadata.html": "https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls-nhtsa-api-6e97f",
    "api_use_policy.html": "https://api.nhtsa.gov/",
}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def acquire():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename, url in SOURCES.items():
        path = raw / filename
        entry_path = raw / (filename + ".download.json")
        if path.exists() and entry_path.exists():
            record = json.loads(entry_path.read_text(encoding="utf-8"))
            if sha256(path) != record["sha256"]:
                raise ValueError("Existing raw file digest mismatch: " + filename)
            entries.append(record)
            print("retained " + filename, flush=True)
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "ResearchDataAcquisition/1.0 (official NHTSA bulk dataset)",
                                                       "Accept-Encoding": "identity"})
        started = datetime.now(timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = response.status
                headers = dict(response.headers.items())
                final_url = response.url
                temporary = path.with_name(path.name + ".partial")
                with temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            temporary.rename(path)
            record = {"source_agency": "NHTSA, U.S. Department of Transportation", "requested_url": url,
                      "final_url": final_url, "retrieved_started_utc": started,
                      "retrieved_completed_utc": datetime.now(timezone.utc).isoformat(),
                      "http_status": status, "response_headers": headers,
                      "local_file": "raw/" + filename, "bytes": path.stat().st_size,
                      "sha256": sha256(path)}
            entry_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            entries.append(record)
            print(json.dumps({"download": filename, "bytes": record["bytes"], "sha256": record["sha256"]}), flush=True)
        except Exception as error:
            record = {"requested_url": url, "retrieved_started_utc": started,
                      "error_type": type(error).__name__, "error": str(error)}
            (raw / (filename + ".failure.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
            if filename in ("FLAT_RCL_POST_2010.zip", "RCL.txt"):
                raise
    (ROOT / "download_manifest.json").write_text(json.dumps({"sources": entries}, indent=2), encoding="utf-8")
    archive = raw / "FLAT_RCL_POST_2010.zip"
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Downloaded NHTSA ZIP failed CRC")
        members = zipped.namelist()
        extracted = []
        for name in members:
            if name.lower().endswith((".txt", ".lst")):
                destination = raw / Path(name).name
                if not destination.exists():
                    with zipped.open(name) as source, destination.open("wb") as target:
                        while block := source.read(1024 * 1024):
                            target.write(block)
                extracted.append({"zip_member": name, "local_file": "raw/" + destination.name,
                                  "bytes": destination.stat().st_size, "sha256": sha256(destination)})
        (ROOT / "archive_manifest.json").write_text(json.dumps({"zip_file": "raw/FLAT_RCL_POST_2010.zip",
                   "zip_sha256": sha256(archive), "crc_check": "passed", "members": members,
                   "extracted": extracted}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    acquire()
