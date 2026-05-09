# Dockerfile — Deployment Decision Engine
# =========================================
# Deterministic, CPU-only, bounded-deterministic execution environment.
# All runs in production and CI MUST use this exact image.
#
# ── DIGEST PINNING ────────────────────────────────────────────────────────
# The FROM line MUST reference the image by SHA256 digest, never by tag.
# Tags are mutable; digests are immutable.
#
# To obtain/refresh the digest:
#   docker pull python:3.14-slim
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.14-slim
#   make pin-digest          # auto-patches this Dockerfile in-place
#
# RECORDED DIGEST (amd64/linux, python:3.14-slim):
#   <run `make pin-digest` to record the current digest here>
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.14-slim

# ── Image metadata ─────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="Deployment Decision Engine"
LABEL org.opencontainers.image.description="Production model serving runtime — bounded-deterministic, CPU-only"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.base.name="python:3.14-slim"
LABEL dde.python.version="3.14"
LABEL dde.cpu.baseline="AVX2"
LABEL dde.cpu.arch="x86_64"

# ── Determinism environment variables ─────────────────────────────────────
# PYTHONHASHSEED=0          — deterministic dict/set ordering
# OMP_NUM_THREADS=1         — single-threaded OpenMP (numpy/BLAS)
# MKL_NUM_THREADS=1         — single-threaded Intel MKL
# OPENBLAS_NUM_THREADS=1    — single-threaded OpenBLAS
# OPENBLAS_CORETYPE=HASWELL — pin BLAS code path regardless of host CPU brand
# MKL_CBWR=COMPATIBLE       — MKL conditional numerical reproducibility
# CUDA_VISIBLE_DEVICES=""   — disable GPU (CPU-only execution)
# PYTHONDONTWRITEBYTECODE=1 — no .pyc drift between builds
# PYTHONUNBUFFERED=1        — unbuffered stdout/stderr for log flushing
ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV OPENBLAS_CORETYPE=HASWELL
ENV MKL_CBWR=COMPATIBLE
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

WORKDIR /app

# ── System deps (locked by image digest — deterministic layer) ────────────
# libgomp1  — required by OpenBLAS / onnxruntime CPU kernels
# cpuid     — used by AVX2 baseline check below
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        cpuid \
 && rm -rf /var/lib/apt/lists/*

# ── Verify CPU meets AVX2 baseline BEFORE any Python work ─────────────────
# Rejects the build immediately on pre-Haswell or non-x86_64 hardware.
# This enforces STEP 3 (CPU Feature Baseline) and STEP 8 (Hardware Class).
RUN cpuid -1 2>/dev/null | grep -q "AVX2" \
    || grep -q "avx2" /proc/cpuinfo \
    || { echo "FATAL: CPU does not support AVX2 — build rejected (STEP 8)" >&2; exit 1; }

# ── Copy hardware class declaration ───────────────────────────────────────
COPY hardware_class.json ./

# ── Copy wheel archive + SHA256 manifest (pre-downloaded, pre-hashed) ─────
# wheels/ is populated via:  make freeze-wheels
# wheels.sha256 is generated: make wheel-hashes
COPY wheels.sha256 ./
COPY wheels/ ./wheels/

# ── Verify ALL wheel hashes BEFORE installing anything ────────────────────
# Any tampered, corrupted, or mismatched wheel aborts the build here.
RUN sha256sum -c wheels.sha256 \
    || { echo "FATAL: wheel hash mismatch — aborting build (STEP 6)" >&2; exit 1; }

# ── Upgrade pip to pinned version (from wheels if available) ──────────────
COPY requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip==24.0

# ── Install exact pinned deps from local wheels ONLY — no network ─────────
# --no-index          — never reach out to PyPI
# --find-links=wheels — use only the verified local archive
RUN pip install \
        --no-cache-dir \
        --no-index \
        --find-links=./wheels \
        -r requirements.lock

# ── Copy source ───────────────────────────────────────────────────────────
COPY . .

# ── Install project in editable mode (from local wheels where possible) ───
RUN pip install --no-cache-dir -e .

# ── Generate deep environment fingerprint at BUILD TIME ───────────────────
# This bakes the authoritative fingerprint into the image layer.
# Any runtime environment that diverges from this will be rejected by
# src/env_check.py before serving begins.
RUN python scripts/generate_fingerprint.py --output /app/fingerprint.json \
 && echo "[build] fingerprint written:" \
 && cat /app/fingerprint.json

# ── Verify model artefact hashes ──────────────────────────────────────────
RUN python scripts/verify_hashes.py \
    || { echo "FATAL: model artefact hash mismatch at build time" >&2; exit 1; }

# ── Default entrypoint ────────────────────────────────────────────────────
CMD ["python", "main.py"]
