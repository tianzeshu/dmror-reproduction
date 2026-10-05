"""Read-only validation of collection code and metadata in the lightweight Git subset."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(stream):
    size, h = 0, hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(block)
        h.update(block)
    return size, h.hexdigest()

def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "GIT_FILE_MANIFEST.json").read_text(encoding="utf-8"))
    paths, failures = set(), []
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if path == root or root not in path.parents or relative in paths:
            failures.append({"path": relative, "reason": "Unsafe or duplicated manifest path"})
            continue
        paths.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "Missing file"})
            continue
        with path.open("rb") as stream:
            stored = digest(stream)
        if stored != (entry["stored_bytes"], entry["stored_sha256"]):
            failures.append({"path": relative, "reason": "Stored bytes/hash mismatch"})
            continue
        if entry["encoding"] != "identity" or path.suffix == ".gz":
            failures.append({"path": relative, "reason": "Git subset must contain only identity code/metadata files"})
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.name != "GIT_FILE_MANIFEST.json"}
    for relative in sorted(actual - paths):
        failures.append({"path": relative, "reason": "File not listed in Git manifest"})
    return {"status": "passed" if not failures else "failed", "files": len(paths),
            "distribution": "code_and_metadata_only", "real_data_included": False,
            "full_data_release": "v1.2.0", "failures": failures}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "real_public")
    result = verify(parser.parse_args().root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(result["status"] != "passed")
