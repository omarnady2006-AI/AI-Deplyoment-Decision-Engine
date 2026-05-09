"""
scripts/verify_hashes.py — Model Artefact Integrity Verifier
=============================================================
Reads model_hashes.sha256 and re-computes SHA256 for every listed file.
Exits non-zero if ANY hash does not match.

Usage:
    py scripts/verify_hashes.py
    py scripts/verify_hashes.py --manifest model_hashes.sha256
    py scripts/verify_hashes.py --manifest model_bundle_v1/model_hashes.sha256

Exit codes:
    0  — all hashes match
    1  — one or more hashes differ (HARD STOP)
    2  — manifest file not found

Called automatically by:
    - make verify-hashes
    - make reproduce
    - make ci
    - src/env_check.py at startup (if MODEL_HASH_VERIFY=1)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            rel = child.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\x00")
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify SHA256 integrity of all model artefacts."
    )
    parser.add_argument(
        "--manifest", "-m", default="model_hashes.sha256",
        help="Path to the SHA256 manifest (default: model_hashes.sha256)",
    )
    parser.add_argument(
        "--root", "-r", default=".",
        help="Root directory for resolving relative paths in manifest (default: .)",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[verify_hashes] ERROR: manifest not found: {manifest!r}")
        print(
            "[verify_hashes] Run `py scripts/generate_hashes.py` first "
            "to create the manifest."
        )
        sys.exit(2)

    root = Path(args.root).resolve()
    lines = [
        ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    if not lines:
        print("[verify_hashes] WARNING: manifest is empty — nothing to verify.")
        sys.exit(0)

    failures: list[str] = []
    ok_count = 0

    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            print(f"[verify_hashes] Skipping malformed line: {line!r}")
            continue

        expected_digest, rel_path = parts
        path = root / rel_path

        if not path.exists():
            failures.append(f"  MISSING   {rel_path}")
            continue

        if path.is_dir():
            actual_digest = sha256_directory(path)
        else:
            actual_digest = sha256_file(path)

        if actual_digest != expected_digest:
            failures.append(
                f"  MISMATCH  {rel_path}\n"
                f"            expected: {expected_digest}\n"
                f"            actual:   {actual_digest}"
            )
        else:
            ok_count += 1
            print(f"  OK        {rel_path}")

    print(f"\n[verify_hashes] {ok_count}/{len(lines)} artefacts verified.")

    if failures:
        print("\n[verify_hashes] FAILURES:")
        for f in failures:
            print(f)
        print(
            "\n[verify_hashes] Artefact integrity check FAILED.\n"
            "  Model files have changed since the manifest was generated.\n"
            "  Re-run `py scripts/generate_hashes.py` ONLY after explicit approval."
        )
        sys.exit(1)

    print("[verify_hashes] All artefact hashes OK.")


if __name__ == "__main__":
    main()
