#!/usr/bin/env bash
# scripts/freeze_wheels.sh — Pre-download and hash all dependency wheels
# ======================================================================
# Implements STEP 5 (Freeze Binary Wheels) and STEP 6 (Verify Wheel Hashes).
#
# MUST be run on the TARGET platform (linux/amd64, Python 3.10.13) to
# produce platform-correct wheel files.  The recommended way is:
#
#   docker run --rm \
#     -v "$(pwd)/wheels:/wheels_out" \
#     python:3.10.13-slim \
#     bash -c "pip download -r /tmp/req.lock -d /wheels_out --platform linux_x86_64 \
#              --python-version 310 --only-binary=:all:"
#
# After running, commit wheels/ and wheels.sha256 to the repo (or store in
# a locked artefact registry).  The Dockerfile installs ONLY from this
# pre-verified local archive.
#
# Usage:
#   bash scripts/freeze_wheels.sh                   # download + hash
#   bash scripts/freeze_wheels.sh --verify-only     # re-verify existing
#
# Exit codes:
#   0 — success
#   1 — download or hash failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"
HASH_FILE="${REPO_ROOT}/wheels.sha256"
LOCK_FILE="${REPO_ROOT}/requirements.lock"
VERIFY_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

echo "==> [freeze_wheels] REPO_ROOT: ${REPO_ROOT}"
echo "==> [freeze_wheels] WHEELS_DIR: ${WHEELS_DIR}"
echo "==> [freeze_wheels] HASH_FILE: ${HASH_FILE}"

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  # ── Download phase ────────────────────────────────────────────────────
  echo ""
  echo "==> [freeze_wheels] STEP 5: Downloading wheels from PyPI..."
  mkdir -p "${WHEELS_DIR}"

  pip download \
    -r "${LOCK_FILE}" \
    -d "${WHEELS_DIR}" \
    --platform linux_x86_64 \
    --python-version 310 \
    --implementation cp \
    --only-binary=:all: \
    --no-cache-dir

  echo "==> [freeze_wheels] Download complete. Files in ${WHEELS_DIR}:"
  ls -lh "${WHEELS_DIR}"

  # ── Hash generation phase ─────────────────────────────────────────────
  echo ""
  echo "==> [freeze_wheels] STEP 6: Generating wheel SHA256 manifest..."
  (
    cd "${WHEELS_DIR}"
    sha256sum ./*.whl > "${HASH_FILE}"
  )
  echo "==> [freeze_wheels] Manifest written to: ${HASH_FILE}"
fi

# ── Verification phase ────────────────────────────────────────────────────
echo ""
echo "==> [freeze_wheels] Verifying wheel hashes..."
(
  cd "${WHEELS_DIR}"
  sha256sum -c "${HASH_FILE}"
)

echo ""
echo "==> [freeze_wheels] All wheels verified OK."
echo "==> [freeze_wheels] Commit wheels/ and wheels.sha256 to the repository."
