"""Read-only audit of portable files and decoded gzip bytes; no source tree required."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

def digest(stream):
    value, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
        size += len(block)
    return size, value.hexdigest()

def resolved(root, relative):
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Manifest path escapes package root")
    return path

def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "public_file_manifest.json").read_text(encoding="utf-8"))
    errors, count, decoded, reconstructed = [], 0, 0, 0
    seen = set()
    for entry in manifest["files"]:
        path = resolved(root, entry["path"])
        try:
            if entry["path"] in seen:
                raise ValueError("Duplicate manifest path")
            seen.add(entry["path"])
            resolved(root, entry["decoded_path"])
            with path.open("rb") as stream:
                size, sha = digest(stream)
            if size != entry["stored_bytes"] or sha != entry["stored_sha256"]:
                raise ValueError("Stored file size/SHA-256 differs")
            if entry["encoding"] == "gzip":
                with path.open("rb") as stream:
                    header = stream.read(10)
                    if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
                        raise ValueError("Invalid gzip header")
                    if header[4:8] != b"\x00\x00\x00\x00":
                        raise ValueError("Gzip mtime is not zero")
                with gzip.open(path, "rb") as stream:
                    decoded_size, decoded_sha = digest(stream)
                if decoded_size != entry["decoded_bytes"] or decoded_sha != entry["decoded_sha256"]:
                    raise ValueError("Decoded gzip bytes differ from collected source")
                decoded += 1
            elif entry["encoding"] == "identity":
                if size != entry["decoded_bytes"] or sha != entry["decoded_sha256"]:
                    raise ValueError("Identity file decoded metadata differs")
            else:
                raise ValueError("Unknown manifest encoding")
            count += 1
        except Exception as error:
            errors.append({"path": entry["path"], "error_type": type(error).__name__, "error": str(error)})
    for entry in manifest.get("reconstructable_archive_members", []):
        try:
            archive = resolved(root, entry["archive_path"])
            if entry["archive_path"] not in seen:
                raise ValueError("Reconstructable archive is absent from manifest")
            resolved(root, entry["decoded_path"])
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open(entry["zip_member"]) as stream:
                    size, sha = digest(stream)
            if size != entry["decoded_bytes"] or sha != entry["decoded_sha256"]:
                raise ValueError("Archive member decoded bytes differ")
            reconstructed += 1
        except Exception as error:
            errors.append({"path": entry["decoded_path"], "error_type": type(error).__name__, "error": str(error)})
    result = {"status": "passed" if not errors else "failed", "files_verified": count,
              "expected_files": len(manifest["files"]), "gzip_decoded_files_verified": decoded,
              "reconstructable_archive_members_verified": reconstructed,
              "package_sources": manifest["included_sources"], "errors": errors,
              "checks": ["stored bytes and SHA-256", "gzip decoded original bytes and SHA-256", "gzip mtime zero", "portable relative paths", "reconstructable archive member original bytes and SHA-256"]}
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)
