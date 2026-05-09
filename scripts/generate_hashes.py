"""
scripts/generate_hashes.py — Model Artefact SHA256 Generator
=============================================================
Walks the project for model files (.onnx, .pt, .pth, .tflite, SavedModel dirs)
and writes a SHA256 manifest to model_hashes.sha256.

Usage:
    py scripts/generate_hashes.py
    py scripts/generate_hashes.py --root .
    py scripts/generate_hashes.py --output model_hashes.sha256

Output format (one line per file):
    <hex_sha256>  <relative_path>

This format is compatible with `sha256sum --check` on Linux/macOS.
On Windows, use `py scripts/verify_hashes.py` instead.

Run this script:
  - After any model file is changed or newly added
  - After EACH model version release
  - NEVER without re-running golden tests afterward
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# File extensions to include in the hash manifest
_MODEL_EXTENSIONS = {".onnx", ".pt", ".pth", ".tflite"}

# Directory markers for TF SavedModel bundles
_SAVEDMODEL_MARKER = "saved_model.pb"


def sha256_file(path: Path) -> str:
    """Return hex SHA256 of a single file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path: Path) -> str:
    """Return hex SHA256 of a directory by hashing all files deterministically.

    Files are sorted by relative path to ensure cross-platform consistency.
    """
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            # Hash the relative path too — catches renames
            rel = child.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\x00")
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    return h.hexdigest()


def find_model_files(root: Path) -> list[tuple[str, Path]]:
    """Return (hash_type, path) for all model files under root."""
    results: list[tuple[str, Path]] = []

    # Skip these directories entirely
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules"}

    for path in sorted(root.rglob("*")):
        # Skip hidden / cache dirs
        if any(part in skip_dirs for part in path.parts):
            continue

        if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS:
            results.append(("file", path))
        elif path.is_dir() and (path / _SAVEDMODEL_MARKER).exists():
            results.append(("dir", path))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SHA256 hashes for all model artefacts."
    )
    parser.add_argument(
        "--root", "-r", default=".",
        help="Root directory to search (default: .)",
    )
    parser.add_argument(
        "--output", "-o", default="model_hashes.sha256",
        help="Output manifest file (default: model_hashes.sha256)",
    )
    args = parser.parse_args()

    root    = Path(args.root).resolve()
    entries = find_model_files(root)

    if not entries:
        print(f"[generate_hashes] No model files found under {root}")
        sys.exit(0)

    lines: list[str] = []
    for kind, path in entries:
        if kind == "file":
            digest = sha256_file(path)
        else:
            digest = sha256_directory(path)
        rel = path.relative_to(root).as_posix()
        lines.append(f"{digest}  {rel}")
        print(f"  {digest[:16]}…  {rel}")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[generate_hashes] Written {len(lines)} entries to: {out.resolve()}")


if __name__ == "__main__":
    main()
