"""Verify the five fixed v1.0.0 parts and restore the unchanged release ZIP.

Uses only Python's standard library. Digests are pinned here; a downloaded
manifest cannot override them. An existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ZIP_NAME = "DMROR_Public_Delivery_20261005.zip"
ZIP_BYTES = 324985351
ZIP_SHA256 = "3cb88e7e65bc519a547af2ec7ec95c58834cce711de0f83fe53538c726a6ca78"
PARTS = (
    (ZIP_NAME + ".part01", 67108864, "2de149f0aec715b7153560d7d7f81ed03145703c5e2c43abb6fca26e541de148"),
    (ZIP_NAME + ".part02", 67108864, "d14671c8ac337499ee43ae05c07153a6ae2e80f8e610daf0385f59f63f078fb7"),
    (ZIP_NAME + ".part03", 67108864, "ed95a15bf8c6209aca3adf1f8bf5a5348dd4e9362fac7554c2f96038df988e6a"),
    (ZIP_NAME + ".part04", 67108864, "54c62752a03f26b7c72d228bb3edd0fb3b4c83dffef264e8c12c8dbfb0e58d12"),
    (ZIP_NAME + ".part05", 56549895, "9dfa0df7ed3a46c00b6857ee568e620f859b7f110a11fdf242acb1ec8487b158"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assemble(parts_dir: Path, output_dir: Path) -> Path:
    joined = hashlib.sha256()
    for name, size, expected in PARTS:
        path = parts_dir / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing part or incorrect byte count: {name} (expected {size})")
        part_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                part_hash.update(block)
                joined.update(block)
        if part_hash.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        print(f"Verified {name}")
    if joined.hexdigest() != ZIP_SHA256:
        raise ValueError("Combined parts do not match the original ZIP digest")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ZIP_NAME
    if output.exists():
        if output.is_file() and output.stat().st_size == ZIP_BYTES and digest(output) == ZIP_SHA256:
            print(f"Existing ZIP verified; unchanged: {output}")
            return output
        raise FileExistsError(f"Output exists with different content; refusing to overwrite: {output}")

    # Exclusive creation also protects against another process creating output
    # between the existence check and this open. Only our new partial file is
    # removed if copying fails; pre-existing files are never opened for writing.
    with output.open("xb") as destination:
        try:
            copied = hashlib.sha256()
            for name, _, _ in PARTS:
                with (parts_dir / name).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        copied.update(block)
            if destination.tell() != ZIP_BYTES or copied.hexdigest() != ZIP_SHA256:
                raise ValueError("Parts changed while copying; output is not the original ZIP")
        except BaseException:
            destination.close()
            output.unlink()
            raise
    print(f"Restored {output}\nSHA-256: {ZIP_SHA256}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, default=Path("."), help="Directory containing all five fixed-name parts")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for the restored fixed-name ZIP")
    args = parser.parse_args()
    try:
        assemble(args.parts_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Release assembly failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
