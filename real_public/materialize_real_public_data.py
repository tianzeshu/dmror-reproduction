"""Restore exact plain normalized data and omitted extracted archives when requested."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from verify_real_public_delivery import digest, resolved

def restore(source, destination, expected_size, expected_sha):
    if destination.exists():
        with destination.open("rb") as existing:
            size, sha = digest(existing)
        if (size, sha) != (expected_size, expected_sha):
            raise ValueError("Refusing to replace changed existing data: " + str(destination))
        return "retained"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".materializing.partial")
    hasher, size = hashlib.sha256(), 0
    with temporary.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
            hasher.update(block)
            size += len(block)
    if (size, hasher.hexdigest()) != (expected_size, expected_sha):
        raise ValueError("Materialized bytes fail original-data checksum")
    temporary.rename(destination)
    return "restored"

def materialize(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "public_file_manifest.json").read_text(encoding="utf-8"))
    results = []
    for entry in manifest["files"]:
        if entry["encoding"] == "gzip":
            destination = resolved(root, entry["decoded_path"])
            with gzip.open(resolved(root, entry["path"]), "rb") as source:
                status = restore(source, destination, entry["decoded_bytes"], entry["decoded_sha256"])
            results.append({"path": entry["decoded_path"], "status": status})
    for entry in manifest.get("reconstructable_archive_members", []):
        destination = resolved(root, entry["decoded_path"])
        with zipfile.ZipFile(resolved(root, entry["archive_path"])) as archive:
            with archive.open(entry["zip_member"]) as source:
                status = restore(source, destination, entry["decoded_bytes"], entry["decoded_sha256"])
        results.append({"path": entry["decoded_path"], "status": status})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps({"files": materialize(args.root)}, ensure_ascii=False, indent=2))
