#!/usr/bin/env bash
# scripts/verify_wheels.sh — Verify pre-frozen wheel archive integrity
# ====================================================================
# Implements STEP 6 (Verify Wheel Hashes).
# Called by the Dockerfile RUN layer and by `make verify-wheels`.
#
# Usage:
#   bash scripts/verify_wheels.sh
#   bash scripts/verify_wheels.sh --wheels-dir ./wheels --hash-file wheels.sha256
#
# Exit codes:
#   0 — all hashes match
#   1 — one or more hashes differ or files missing (HARD STOP)
#   2 — wheels/ directory or wheels.sha256 not found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
HASH_FILE="${REPO_ROOT}/wheels.sha256"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheels-dir) WHEELS_DIR="$2"; shift 2 ;;
    --hash-file)  HASH_FILE="$2";  shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "${WHEELS_DIR}" ]]; then
  echo "[verify_wheels] ERROR: wheels directory not found: ${WHEELS_DIR}" >&2
  echo "[verify_wheels] Run: bash scripts/freeze_wheels.sh" >&2
  exit 2
fi

if [[ ! -f "${HASH_FILE}" ]]; then
  echo "[verify_wheels] ERROR: hash manifest not found: ${HASH_FILE}" >&2
  echo "[verify_wheels] Run: bash scripts/freeze_wheels.sh" >&2
  exit 2
fi

echo "[verify_wheels] Verifying wheels against: ${HASH_FILE}"
wheel_count=$(wc -l < "${HASH_FILE}" | tr -d ' ')
echo "[verify_wheels] Manifest contains ${wheel_count} entries."

(
  cd "${WHEELS_DIR}"
  sha256sum -c "${HASH_FILE}" --strict
) || {
  echo ""
  echo "[verify_wheels] FATAL: Wheel hash verification FAILED." >&2
  echo "[verify_wheels] One or more wheels have been tampered with or corrupted." >&2
  echo "[verify_wheels] Re-run: bash scripts/freeze_wheels.sh to refresh the archive." >&2
  exit 1
}

echo "[verify_wheels] OK — all ${wheel_count} wheels verified."
