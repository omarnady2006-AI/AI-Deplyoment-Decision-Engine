"""
scripts/generate_fingerprint.py — Deep Environment Fingerprint Generator
========================================================================
Records the exact runtime environment to a JSON file, including:
  - Python version and implementation
  - OS / hardware platform
  - libc version (glibc variant and release)
  - CPU feature flags (parsed from /proc/cpuinfo or cpuid)
  - BLAS backend identity (OpenBLAS / MKL / ATLAS / Accelerate)
  - All framework library versions
  - Determinism env-var values

Usage:
    python scripts/generate_fingerprint.py
    python scripts/generate_fingerprint.py --output fingerprint.json
    python scripts/generate_fingerprint.py --output model_bundle_v1/fingerprint.json

The fingerprint.json is an immutable, auditable record of the environment
that produced the associated golden outputs or model artefacts.
verify_fingerprint.py rejects any runtime that diverges from this record.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── CPU flags considered part of the hardware baseline ───────────────────
# These are the flags we fingerprint and enforce.  Additional flags on the
# host are irrelevant; MISSING flags from this set are a hard rejection.
_BASELINE_CPU_FLAGS = {
    "sse", "sse2", "sse4_1", "sse4_2", "avx", "avx2", "fma",
}


def _safe_version(module_name: str) -> str:
    """Return `module.__version__` or 'NOT_INSTALLED' without raising."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "NOT_INSTALLED"


def _git_commit() -> str:
    """Return current HEAD SHA or 'UNKNOWN' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def _cpu_info() -> str:
    """Return a best-effort CPU identifier string."""
    try:
        import cpuinfo  # optional dep
        return cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except ImportError:
        return platform.processor() or "UNKNOWN"


def get_cpu_flags() -> list[str]:
    """Return sorted list of CPU feature flags for this machine.

    Strategy:
      1. /proc/cpuinfo (Linux — authoritative)
      2. subprocess cpuid (if installed)
      3. platform.machine() fallback — records arch only
    """
    # ── Linux: /proc/cpuinfo ──────────────────────────────────────────────
    proc_cpuinfo = Path("/proc/cpuinfo")
    if proc_cpuinfo.exists():
        try:
            text = proc_cpuinfo.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("flags") or line.startswith("Features"):
                    # "flags\t\t: fpu vme de pse ..."
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        flags = sorted(set(parts[1].split()))
                        return flags
        except OSError:
            pass

    # ── macOS: sysctl ────────────────────────────────────────────────────
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5,
            )
            raw = result.stdout.replace("\n", " ")
            flags = sorted(set(f.lower() for f in raw.split() if f))
            if flags:
                return flags
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    return [platform.machine().lower() or "unknown"]


def detect_blas_backend() -> str:
    """Return a string identifying the active BLAS backend.

    Checks in priority order:
      1. numpy.show_config() structured info (numpy ≥ 2.0)
      2. numpy.__config__ string heuristic
      3. 'UNKNOWN'
    """
    try:
        import numpy as np

        # numpy ≥ 2.0 exposes structured build config
        if hasattr(np, "show_config"):
            try:
                import io
                buf = io.StringIO()
                np.show_config(mode="dicts")  # raises on older numpy
                # If we get here, parse the dict
                cfg = np.__config__.blas_opt_info  # type: ignore[attr-defined]
                libs = " ".join(cfg.get("libraries", []))
                if "openblas" in libs.lower():
                    return "OpenBLAS"
                if "mkl" in libs.lower():
                    return "MKL"
                if "atlas" in libs.lower():
                    return "ATLAS"
                if "blis" in libs.lower():
                    return "BLIS"
                if "accelerate" in libs.lower():
                    return "Accelerate"
                return libs or "UNKNOWN"
            except (AttributeError, TypeError):
                pass

        # Older numpy: parse __config__ as string
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            config_text = buf.getvalue().lower()
            for backend in ("openblas", "mkl", "blis", "atlas", "accelerate"):
                if backend in config_text:
                    return backend.upper() if backend != "openblas" else "OpenBLAS"
        except Exception:
            pass

        # Last resort: inspect linked library names
        try:
            cfg = np.__config__  # type: ignore[attr-defined]
            blas_info = str(getattr(cfg, "blas_opt_info", "") or "").lower()
            for backend in ("openblas", "mkl", "blis", "atlas", "accelerate"):
                if backend in blas_info:
                    return backend.upper() if backend != "openblas" else "OpenBLAS"
        except AttributeError:
            pass

    except ImportError:
        return "numpy_NOT_INSTALLED"

    return "UNKNOWN"


def get_libc_version() -> list[str]:
    """Return [name, version] for the C library, e.g. ['glibc', '2.36'].

    Uses platform.libc_ver() which returns ('glibc', '2.x') on glibc systems
    and ('', '') on non-glibc (macOS, musl, Windows).
    Falls back to parsing ldd --version output.
    """
    name, version = platform.libc_ver()
    if name and version:
        return [name, version]

    # ldd --version fallback (catches musl branding as well)
    try:
        result = subprocess.run(
            ["ldd", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0]
        # "ldd (GNU libc) 2.36" or "musl libc (x86_64) Version 1.2.3"
        if "musl" in first_line.lower():
            parts = first_line.split()
            ver = parts[-1] if parts else "unknown"
            return ["musl", ver]
        tokens = first_line.split()
        ver = tokens[-1] if tokens else "unknown"
        return ["glibc", ver]
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, OSError):
        pass

    return [platform.system().lower() or "unknown", "unknown"]


def build_fingerprint() -> dict:
    """Collect and return the full deep environment fingerprint."""
    return {
        # ── Timestamps ──────────────────────────────────────────────────
        "generated_at":         datetime.now(timezone.utc).isoformat(),

        # ── Python runtime ───────────────────────────────────────────────
        "python":               platform.python_version(),
        "python_impl":          platform.python_implementation(),
        "python_executable":    sys.executable,

        # ── OS / hardware ────────────────────────────────────────────────
        "os":                   platform.platform(),
        "os_name":              platform.system(),
        "os_version":           platform.version(),
        "machine":              platform.machine(),
        "cpu":                  _cpu_info(),
        "cpu_count":            os.cpu_count(),

        # ── Deep environment fields (STEP 4) ─────────────────────────────
        "libc":                 get_libc_version(),
        "cpu_flags":            get_cpu_flags(),
        "blas":                 detect_blas_backend(),

        # ── Framework versions ───────────────────────────────────────────
        "numpy":                _safe_version("numpy"),
        "onnxruntime":          _safe_version("onnxruntime"),
        "torch":                _safe_version("torch"),
        "tensorflow":           _safe_version("tensorflow"),
        "onnx":                 _safe_version("onnx"),

        # ── Determinism env vars ─────────────────────────────────────────
        "PYTHONHASHSEED":           os.environ.get("PYTHONHASHSEED",       "NOT_SET"),
        "OMP_NUM_THREADS":          os.environ.get("OMP_NUM_THREADS",       "NOT_SET"),
        "MKL_NUM_THREADS":          os.environ.get("MKL_NUM_THREADS",       "NOT_SET"),
        "OPENBLAS_NUM_THREADS":     os.environ.get("OPENBLAS_NUM_THREADS",  "NOT_SET"),
        "OPENBLAS_CORETYPE":        os.environ.get("OPENBLAS_CORETYPE",     "NOT_SET"),
        "MKL_CBWR":                 os.environ.get("MKL_CBWR",              "NOT_SET"),
        "CUDA_VISIBLE_DEVICES":     os.environ.get("CUDA_VISIBLE_DEVICES",  "NOT_SET"),

        # ── Source control ───────────────────────────────────────────────
        "git_commit":           _git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deep environment fingerprint JSON file."
    )
    parser.add_argument(
        "--output", "-o",
        default="fingerprint.json",
        help="Output path for the fingerprint file (default: fingerprint.json)",
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print the JSON (default: True)",
    )
    args = parser.parse_args()

    fp = build_fingerprint()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(fp, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[fingerprint] Written to: {out_path.resolve()}")
    for key, val in fp.items():
        print(f"  {key:<30} {val}")


if __name__ == "__main__":
    main()
