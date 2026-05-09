"""
scripts/verify_fingerprint.py — Environment Drift Detector
===========================================================
Compares the current runtime environment against a stored fingerprint.
Called at container startup and in CI before any test or serving begins.

Usage:
    py scripts/verify_fingerprint.py                           # uses fingerprint.json
    py scripts/verify_fingerprint.py --expected fingerprint.json
    py scripts/verify_fingerprint.py --expected fingerprint.json --strict

Exit codes:
    0  — environment matches (within tolerance)
    1  — environment mismatch (HARD STOP)

Fields compared (strict):
    python, numpy, onnxruntime, torch, tensorflow,
    PYTHONHASHSEED, OMP_NUM_THREADS, MKL_NUM_THREADS,
    OPENBLAS_NUM_THREADS, CUDA_VISIBLE_DEVICES

Fields NOT compared (informational only):
    generated_at, git_commit, os_version, cpu, cpu_count
    (these legitimately differ across machines)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields that MUST match exactly — any drift = hard failure
_STRICT_FIELDS = [
    "python",
    "python_impl",
    "numpy",
    "onnxruntime",
    "torch",
    "tensorflow",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
]

# Fields compared but only warn on mismatch (can be overridden by --strict)
_WARN_FIELDS = [
    "os_name",
    "machine",
]


def load_fingerprint(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[verify_fingerprint] ERROR: fingerprint file not found: {path!r}")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def current_fingerprint() -> dict:
    """Import the generator and build current fingerprint without writing."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_fingerprint import build_fingerprint
    return build_fingerprint()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the current environment against a stored fingerprint."
    )
    parser.add_argument(
        "--expected", "-e",
        default="fingerprint.json",
        help="Path to the stored fingerprint JSON (default: fingerprint.json)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warn-fields as hard failures too",
    )
    args = parser.parse_args()

    expected = load_fingerprint(args.expected)
    current  = current_fingerprint()

    failures: list[str] = []
    warnings: list[str] = []

    for field in _STRICT_FIELDS:
        exp = expected.get(field, "MISSING")
        got = current.get(field, "MISSING")
        if exp != got:
            failures.append(
                f"  MISMATCH  {field:<30} expected={exp!r}  got={got!r}"
            )

    warn_fields = _STRICT_FIELDS if args.strict else _WARN_FIELDS
    if not args.strict:
        for field in _WARN_FIELDS:
            exp = expected.get(field, "MISSING")
            got = current.get(field, "MISSING")
            if exp != got:
                warnings.append(
                    f"  WARNING   {field:<30} expected={exp!r}  got={got!r}"
                )

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"[verify_fingerprint] Comparing against: {args.expected!r}")
    print(f"[verify_fingerprint] Strict mode: {'ON' if args.strict else 'OFF'}")

    if warnings:
        print("\n[verify_fingerprint] WARNINGS (non-blocking):")
        for w in warnings:
            print(w)

    if failures:
        print("\n[verify_fingerprint] FAILURES (blocking):")
        for f in failures:
            print(f)
        print(
            "\n[verify_fingerprint] ENVIRONMENT MISMATCH — execution rejected.\n"
            "  Reproduce this environment or update the fingerprint after\n"
            "  regenerating golden outputs and obtaining explicit approval."
        )
        sys.exit(1)

    print(f"\n[verify_fingerprint] OK — {len(_STRICT_FIELDS)} fields match.")


if __name__ == "__main__":
    main()
