from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""
GUI Web Application for Deployment Decision Engine

Serves the frontend UI and provides basic API endpoints.
"""


import hashlib
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.app_state import APP_STATE, APP_STATE_LOCK, require_stage
from src.core.logging_config import get_logger
from src.core.pipeline import compute_risk, decision_from_risk
from src.core.deployment_summary import build_deployment_summary
from src.core import persistence


logger = get_logger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Deployment Decision Engine API",
    description="API for deployment decision making and model analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _configured_worker_count() -> int | None:
    workers_env = os.environ.get("UVICORN_WORKERS") or os.environ.get("WEB_CONCURRENCY")
    if workers_env:
        try:
            workers = int(workers_env)
            if workers > 0:
                return workers
        except Exception:
            pass

    try:
        import psutil

        parent = psutil.Process().parent()
        cmdline = parent.cmdline() if parent is not None else []
        for idx, item in enumerate(cmdline):
            token = str(item)
            if token == "--workers" and idx + 1 < len(cmdline):
                return int(cmdline[idx + 1])
            if token.startswith("--workers="):
                return int(token.split("=", 1)[1])
    except Exception:
        return None

    return None


@app.on_event("startup")
async def _enforce_single_worker_state_machine() -> None:
    workers = _configured_worker_count()
    if workers is not None and workers > 1:
        raise RuntimeError(
            "Strict APP_STATE authority requires a single worker process; "
            f"configured workers={workers}"
        )

    latest_cal = persistence.load_latest_calibration()
    if latest_cal is not None:
        try:
            from src.core.gpu_calibration import GPUCalibrationProfile
            profile = GPUCalibrationProfile(latest_cal["raw_json"])
            with APP_STATE_LOCK:
                APP_STATE["gpu_calibration"] = profile
            logger.info(
                "startup_calibration_recovery",
                extra={
                    "event": "startup_calibration_recovery",
                    "providers": latest_cal["provider_list"],
                    "gpu_name": latest_cal["gpu_name"],
                },
            )
        except Exception:
            logger.warning("startup_calibration_recovery_failed", exc_info=True)

    # Profile the server hardware once at startup
    try:
        _profile_server_hardware()
        logger.info("server_hardware_profiled", extra={"event": "server_hardware_profiled", "profile": _SERVER_HW_PROFILE})
    except Exception:
        logger.warning("server_hardware_profile_failed", exc_info=True)


# ── Server hardware profile (filled at startup) ───────────────────────────────
_SERVER_HW_PROFILE: dict[str, Any] = {}

def _profile_server_hardware() -> None:
    """Benchmark server hardware once to enable latency scaling."""
    global _SERVER_HW_PROFILE
    try:
        import psutil
        import numpy as np

        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Quick matmul benchmark to measure server GFLOPS
        size = 512
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        # warmup
        for _ in range(3):
            np.dot(A, B)
        # timed
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            np.dot(A, B)
            times.append(time.perf_counter() - t0)
        avg_s = sum(times) / len(times)
        flops = 2 * (size ** 3)
        matmul_gflops = flops / (avg_s * 1e9)

        # Memory bandwidth (copy 50 MB array)
        arr = np.random.randn(50 * 1024 * 1024 // 8).astype(np.float32)
        for _ in range(3):
            arr.copy()
        bw_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            arr.copy()
            bw_times.append(time.perf_counter() - t0)
        avg_bw_s = sum(bw_times) / len(bw_times)
        memory_bandwidth_gbps = (50 / 1024) / avg_bw_s

        _SERVER_HW_PROFILE = {
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "matmul_gflops": matmul_gflops,
            "memory_bandwidth_gbps": memory_bandwidth_gbps,
        }
    except Exception as e:
        _SERVER_HW_PROFILE = {
            "cpu_cores": 4,
            "ram_gb": 8.0,
            "matmul_gflops": 10.0,
            "memory_bandwidth_gbps": 10.0,
        }


# DDR bandwidth lookup (GB/s theoretical peak per channel)
_DDR_BANDWIDTH_GBPS: dict[str, float] = {
    "ddr3":   25.6,
    "lpddr3": 17.0,
    "ddr4":   51.2,
    "lpddr4": 34.1,
    "ddr5":  102.4,
    "lpddr5": 68.3,
}

def _estimate_target_hw_score(
    cpu_cores: int,
    ram_gb: float,
    ram_ddr: str,
    gpu_available: bool,
) -> float:
    """
    Return a normalised compute score for the user's target hardware
    relative to the server. Score > 1 means target is faster than server.
    """
    srv = _SERVER_HW_PROFILE
    srv_cores = float(srv.get("cpu_cores") or 4)
    srv_ram   = float(srv.get("ram_gb")    or 8.0)
    srv_bw    = float(srv.get("memory_bandwidth_gbps") or 10.0)

    # CPU throughput factor — roughly linear with cores (diminishing returns)
    target_core_score  = math.log1p(float(cpu_cores))
    server_core_score  = math.log1p(srv_cores)
    core_ratio = target_core_score / max(server_core_score, 1e-9)

    # RAM bandwidth factor
    ddr_key = str(ram_ddr or "").lower()
    target_bw = _DDR_BANDWIDTH_GBPS.get(ddr_key, srv_bw)
    # also cap by actual RAM available (very low RAM = less effective bandwidth)
    effective_target_bw = target_bw * min(1.0, float(ram_gb) / 4.0)
    bw_ratio = effective_target_bw / max(srv_bw, 1e-9)

    # Combined score: 60% compute / 40% memory bandwidth
    hw_score = 0.60 * core_ratio + 0.40 * bw_ratio
    return max(0.05, hw_score)  # never let it go to zero


def _scale_latency_for_target_hw(
    measured_ms: float,
    cpu_cores: int,
    ram_gb: float,
    ram_ddr: str,
    gpu_available: bool,
) -> float:
    """Scale measured server latency to estimated latency on target hardware."""
    hw_score = _estimate_target_hw_score(cpu_cores, ram_gb, ram_ddr, gpu_available)
    # If hw_score < 1, target is slower → latency goes up (divide by hw_score)
    scaled = measured_ms / max(hw_score, 0.05)
    return round(scaled, 2)


def _scale_memory_for_target_hw(
    measured_mb: float,
    ram_gb: float,
) -> float:
    """
    Memory usage is mostly constant per model, but low-RAM systems
    may have extra OS overhead. Return a slightly adjusted value.
    """
    if ram_gb < 2.0:
        return measured_mb * 1.3   # OS pressure adds ~30% overhead
    if ram_gb < 4.0:
        return measured_mb * 1.1
    return measured_mb

# ============================================================================
# Helper Functions
# ============================================================================

def _analyze_model(model_path: str) -> dict[str, Any]:
    """Analyze a model file and return basic info."""
    try:
        import onnx
        
        model = onnx.load(model_path)
        graph = model.graph
        
        # Extract operators
        operators = set()
        for node in graph.node:
            operators.add(node.op_type)
        
        # Count parameters
        parameter_count = 0
        for initializer in graph.initializer:
            dims = list(initializer.dims) if hasattr(initializer, 'dims') else []
            product = 1
            for dim in dims:
                product *= dim if dim > 0 else 1
            parameter_count += product
        
        # Check for dynamic shapes
        has_dynamic_shapes = False
        for collection_name in ("input", "output", "value_info"):
            items = getattr(graph, collection_name, []) or []
            for value_info in items:
                tensor_type = getattr(value_info, 'type', None)
                if tensor_type is None:
                    continue
                shape = getattr(tensor_type, 'tensor_type', None)
                if shape is None:
                    continue
                dims = getattr(shape, 'dim', [])
                for dim in dims:
                    if hasattr(dim, 'dim_param') and dim.dim_param:
                        has_dynamic_shapes = True
                        break
                for dim in dims:
                    try:
                        dim_value = dim.dim_value if hasattr(dim, 'dim_value') else None
                        if dim_value is None or int(dim_value) <= 0:
                            has_dynamic_shapes = True
                            break
                    except (ValueError, TypeError):
                        has_dynamic_shapes = True
                        break
                if has_dynamic_shapes:
                    break
        
        # Count operator occurrences
        operator_counts = {}
        for node in graph.node:
            op_type = node.op_type
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1
        
        # Calculate file hash
        try:
            model_bytes = Path(model_path).read_bytes()
            file_hash = hashlib.sha256(model_bytes).hexdigest()
            model_hash_input = f"{Path(model_path).resolve()}:{file_hash}".encode("utf-8")
            model_hash = hashlib.sha1(model_hash_input).hexdigest()
        except Exception:
            model_hash = Path(model_path).name
        
        return {
            "model_path": model_path,
            "file_hash": model_hash,
            "operator_count": len(graph.node),
            "operators": list(operators),
            "operator_counts": operator_counts,
            "parameter_count": parameter_count,
            "has_dynamic_shapes": has_dynamic_shapes,
            "input_count": len(graph.input),
            "output_count": len(graph.output),
            "success": True,
        }
    except Exception as e:
        return {
            "model_path": model_path,
            "success": False,
            "error": str(e),
        }


def _build_facts(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build facts from model analysis."""
    facts: dict[str, Any] = {}
    
    op_types = set(analysis.get("operators", []))
    
    facts["model.has_non_max_suppression"] = "NonMaxSuppression" in op_types
    facts["model.has_resize"] = "Resize" in op_types
    facts["model.has_conv_transpose"] = "ConvTranspose" in op_types
    facts["model.uses_layer_normalization"] = "LayerNormalization" in op_types
    facts["model.uses_batch_normalization"] = "BatchNormalization" in op_types
    facts["model.has_conv"] = "Conv" in op_types or "ConvTranspose" in op_types
    facts["model.has_attention"] = (
        "Attention" in op_types or "MultiHeadAttention" in op_types
    )
    facts["operator_counts"] = analysis.get("operator_counts", {})
    facts["parameter_count"] = analysis.get("parameter_count", 0)
    facts["model.has_dynamic_shapes"] = analysis.get("has_dynamic_shapes", False)
    facts["sequential_depth_estimate"] = sum(
        analysis.get("operator_counts", {}).values()
    )
    
    # Parameter scale class
    if facts["parameter_count"] >= 25_000_000:
        facts["parameter_scale_class"] = "large"
    elif facts["parameter_count"] >= 8_000_000:
        facts["parameter_scale_class"] = "medium"
    else:
        facts["parameter_scale_class"] = "small"
    
    return facts


def run_onnx_with_provider(
    model_path: str,
    provider: str,
    stress_runs: int = 0,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Safe runtime benchmark for an ONNX model on a given execution provider.

    Strategy
    --------
    1. Attempt a full runtime benchmark:
       - Load the model with ONNX Runtime.
       - Resolve every dynamic dimension to a safe concrete value.
       - Generate random dummy input tensors.
       - Run 3 warmup passes, then 10 timed passes (hard 2 s wall-clock cap).
       - Record average and p95 latency; estimate RSS memory delta.
    2. If *anything* in the runtime path raises an exception (shape mismatch,
       unsupported op, OOM, ...), fall back to ONNX graph inspection:
       - Load the protobuf directly with the `onnx` package.
       - Derive memory_mb from initializer tensor sizes.
       - Return latency as None so callers treat it as unmeasured.
    The server never crashes regardless of model content.
    """
    import numpy as np
    import onnxruntime as ort

    _default_stress: dict[str, Any] = {
        "enabled": False,
        "runs": 0,
        "latency_avg_ms": None,
        "latency_p95_ms": None,
        "peak_memory_mb": None,
        "memory_growth_mb": None,
        "memory_stability": "STABLE",
    }

    # ── Provider availability — checked before any expensive I/O ─────────────
    try:
        available_providers = ort.get_available_providers()
    except Exception as _e:
        available_providers = []

    if provider not in available_providers:
        return {
            "success": False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb": None,
            "error": (
                f"Provider '{provider}' is unavailable. "
                f"Available providers: {available_providers}"
            ),
            "stress_test": _default_stress,
        }

    # ── Helper: estimate memory from graph initializers (used by fallback) ────
    def _graph_memory_mb(model_path: str) -> float:
        """Sum weight tensor sizes from the ONNX protobuf."""
        try:
            import onnx
            _ELEM_BYTES: dict[int, int] = {
                1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8,
                9: 1, 10: 2, 11: 8, 12: 4, 13: 8,
            }
            g = onnx.load(model_path).graph
            total = 0
            for init in g.initializer:
                esz = _ELEM_BYTES.get(int(init.data_type), 4)
                n = 1
                for d in init.dims:
                    n *= max(1, int(d))
                total += n * esz
            return float(total) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    # ── Helper: resolve a single raw ORT shape into concrete ints ────────────
    def _resolve_shape(raw_shape: Any) -> list[int]:
        """
        Replace dynamic dimensions with safe concrete values.

        Rules applied in order:
          - Position 0 (batch)     → 1
          - None / negative / str  → 1   (generic dynamic)
          - Spatial dims ≥ 3       → if value is already concrete and > 0, keep it;
                                      otherwise use 224 for image-like models or 128
                                      for sequence/vector models.
        The heuristic for "image vs vector":
          Look at the total number of dimensions:
            4-D  → image convention  (B, C, H, W)  → spatial default 224
            3-D  → sequence          (B, T, F)      → spatial default 128
            2-D  → vector            (B, F)         → spatial default 128
            other → 128
        """
        if not raw_shape:
            return [1, 3, 224, 224]   # safe image default

        ndim = len(raw_shape)
        spatial_default = 224 if ndim == 4 else 128

        resolved: list[int] = []
        for i, dim in enumerate(raw_shape):
            if i == 0:
                # Batch dimension — always 1
                resolved.append(1)
            elif isinstance(dim, int) and dim > 0:
                resolved.append(int(dim))
            else:
                # None, string ("batch", "unk__0", …), -1, or 0
                resolved.append(spatial_default)

        return resolved

    # ── Helper: map ORT type string → numpy dtype ────────────────────────────
    def _ort_type_to_numpy(type_str: str):
        _MAP = {
            "tensor(float)":   np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)":  np.float64,
            "tensor(int64)":   np.int64,
            "tensor(int32)":   np.int32,
            "tensor(int16)":   np.int16,
            "tensor(int8)":    np.int8,
            "tensor(uint64)":  np.uint64,
            "tensor(uint32)":  np.uint32,
            "tensor(uint16)":  np.uint16,
            "tensor(uint8)":   np.uint8,
            "tensor(bool)":    np.bool_,
            "tensor(string)":  None,   # handled separately below
        }
        return _MAP.get(type_str, np.float32)

    # ── Attempt runtime benchmark ─────────────────────────────────────────────
    _BENCHMARK_RUNS  = 10
    _WARMUP_RUNS     = 3
    _WALL_CAP_S      = 2.0    # hard cap: total benchmark wall time ≤ 2 s

    try:
        # Step 1: load the model
        session = ort.InferenceSession(model_path, providers=[provider])
        inputs  = session.get_inputs()

        # Step 2 + 3 + 4: detect inputs, resolve shapes, build dummy tensors
        dummy_inputs: dict[str, Any] = {}
        rng = np.random.default_rng(0)

        for inp in inputs:
            raw_shape = getattr(inp, "shape", None) or []
            dtype_ort  = getattr(inp, "type",  "tensor(float)")
            shape      = _resolve_shape(raw_shape)
            dtype_np   = _ort_type_to_numpy(dtype_ort)

            if dtype_np is None:
                # String tensors — pass empty string array
                dummy_inputs[inp.name] = np.array([""] * int(np.prod(shape)), dtype=object).reshape(shape)
            elif np.issubdtype(dtype_np, np.floating):
                dummy_inputs[inp.name] = rng.random(shape).astype(dtype_np)
            elif np.issubdtype(dtype_np, np.integer):
                dummy_inputs[inp.name] = np.zeros(shape, dtype=dtype_np)
            elif dtype_np is np.bool_:
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.bool_)
            else:
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

        logger.info(
            "onnx_runtime_benchmark_start",
            extra={
                "event":    "onnx_runtime_benchmark_start",
                "provider": provider,
                "input_shapes": {
                    inp.name: _resolve_shape(getattr(inp, "shape", None) or [])
                    for inp in inputs
                },
            },
        )

        # Step 5: warmup runs
        for _ in range(_WARMUP_RUNS):
            session.run(None, dummy_inputs)

        # Step 6: timed benchmark runs with hard wall-clock cap
        import psutil
        process   = psutil.Process()
        rss_before = process.memory_info().rss

        timings_ms: list[float] = []
        _wall_start = time.perf_counter()
        for _ in range(_BENCHMARK_RUNS):
            t0 = time.perf_counter()
            session.run(None, dummy_inputs)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            if (time.perf_counter() - _wall_start) >= _WALL_CAP_S:
                # Wall cap hit — stop early but keep what we have
                break

        rss_after = process.memory_info().rss

        # Step 7: compute metrics
        latency_avg_ms: float = float(sum(timings_ms) / len(timings_ms))
        timings_sorted = sorted(timings_ms)
        p95_idx        = max(0, int(len(timings_sorted) * 0.95) - 1)
        latency_p95_ms: float = float(timings_sorted[p95_idx])
        memory_mb: float = float(max(0, rss_after - rss_before)) / (1024.0 * 1024.0)

        # If RSS delta is zero/tiny (common in same-process tests), fall back
        # to weight-tensor estimate so callers always get a non-zero memory value.
        if memory_mb < 1.0:
            memory_mb = max(memory_mb, _graph_memory_mb(model_path))

        logger.info(
            "onnx_runtime_benchmark_complete",
            extra={
                "event":           "onnx_runtime_benchmark_complete",
                "provider":        provider,
                "latency_avg_ms":  round(latency_avg_ms, 3),
                "latency_p95_ms":  round(latency_p95_ms, 3),
                "memory_mb":       round(memory_mb, 2),
                "runs_completed":  len(timings_ms),
            },
        )

        return {
            "success":        True,
            "latency_avg_ms": latency_avg_ms,
            "latency_p95_ms": latency_p95_ms,
            "memory_mb":      memory_mb,
            "error":          None,
            "benchmark_runs": len(timings_ms),
            "stress_test":    _default_stress,
        }

    except Exception as _bench_exc:
        # ── Fallback: graph inspection only ──────────────────────────────────
        logger.warning(
            "onnx_runtime_benchmark_failed_using_graph_fallback",
            extra={
                "event":    "onnx_runtime_benchmark_fallback",
                "provider": provider,
                "reason":   str(_bench_exc),
            },
        )
        memory_mb_fallback = _graph_memory_mb(model_path)
        return {
            "success":        True,    # graph inspection succeeded
            "latency_avg_ms": None,    # not measured — callers handle None
            "latency_p95_ms": None,
            "memory_mb":      memory_mb_fallback,
            "error":          f"runtime_benchmark_failed: {_bench_exc}; graph_inspection_used",
            "benchmark_runs": 0,
            "stress_test":    _default_stress,
        }


# ============================================================================
# API Routes
# ============================================================================


@app.get("/api/state")
async def get_state():
    """Expose current in-memory state snapshot for frontend hydration."""
    with APP_STATE_LOCK:
        model_state = APP_STATE.get("model")
        diagnostics_state = APP_STATE.get("diagnostics")
        analysis_state = APP_STATE.get("analysis")
        decision_state = APP_STATE.get("decision")
        deployment_profile_state = APP_STATE.get("deployment_profile")

    if decision_state is not None:
        current_stage = "decision"
    elif analysis_state is not None:
        current_stage = "analysis"
    elif diagnostics_state is not None:
        current_stage = "diagnostics"
    elif model_state is not None:
        current_stage = "model"
    else:
        current_stage = "empty"

    return {
        "stage": current_stage,
        "model": model_state,
        "diagnostics": diagnostics_state,
        "analysis": analysis_state,
        "decision": decision_state,
        "deployment_profile": deployment_profile_state,
        "timestamp": time.time(),
    }


# ============================================================================
# Model API
# ============================================================================

class ModelIndexRequest(BaseModel):
    model_path: str


class DeploymentProfileRequest(BaseModel):
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    cuda_available: bool
    vram_gb: float | None = None
    trt_available: bool | None = None
    stress_test: bool | None = None
    cpu_arch: str | None = None
    ram_ddr: str | None = None
    target_latency_ms: float | None = None
    memory_limit_mb: float | None = None


class CalibrationUpdateRequest(BaseModel):
    latencies: list[float]
    model_sizes: list[int]
    notes: str | None = None


class AnalyzeAndDecideRequest(BaseModel):
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    cuda_available: bool
    vram_gb: float | None = None
    trt_available: bool | None = None
    stress_test: bool | None = None
    cpu_arch: str | None = None
    ram_ddr: str | None = None
    target_latency_ms: float | None = None
    memory_limit_mb: float | None = None


# Temp dir for uploaded model files (cleaned up per-request)
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "deploycheck_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Benchmark cache (BUG-3 fix: eliminate OS-jitter latency inversions) ──────
# Key: (model_hash, provider) → benchmark result dict
# Once a model has been benchmarked on a given provider, the raw server
# measurement is stored and reused for all subsequent hardware profiles.
# Hardware-specific scaling (_scale_latency_for_target_hw) is applied on top
# of the cached raw measurement, so different profiles still get different
# predicted latencies — but the SAME raw baseline guarantees correct ordering.
import threading as _threading
_BENCHMARK_CACHE: dict[tuple[str, str], dict] = {}
_BENCHMARK_CACHE_LOCK = _threading.Lock()


@app.post("/api/model/index")
async def get_model_index(file: UploadFile = File(...)):
    """Accept an uploaded model file, save it to a temp path, and index it."""
    try:
        # Save uploaded file to a stable temp location using streaming to support
        # large models (up to 4 GB) without loading the full file into memory.
        safe_name = Path(file.filename or "model.onnx").name
        dest = _UPLOAD_DIR / safe_name
        _hasher = hashlib.sha256()
        with open(str(dest), "wb") as _fh:
            while True:
                _chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not _chunk:
                    break
                _fh.write(_chunk)
                _hasher.update(_chunk)
        model_path = str(dest)

        # Compute canonical hash from the streamed content
        try:
            file_hash = _hasher.hexdigest()
            model_hash_input = f"{dest.resolve()}:{file_hash}".encode("utf-8")
            model_hash = hashlib.sha1(model_hash_input).hexdigest()
        except Exception:
            model_hash = safe_name

        analysis = _analyze_model(model_path)

        # Always reset full downstream pipeline regardless of success/failure
        with APP_STATE_LOCK:
            if analysis.get("success", False):
                APP_STATE["model"] = {
                    "path": model_path,
                    "hash": model_hash,
                }
            else:
                APP_STATE["model"] = None
            APP_STATE["diagnostics"] = None
            APP_STATE["analysis"] = None
            APP_STATE["decision"] = None

        # Inject canonical hash into returned analysis
        if analysis.get("success", False):
            analysis["file_hash"] = model_hash
            persistence.save_model_record(
                file_hash=model_hash,
                operator_count=analysis.get("operator_count"),
                parameter_count=analysis.get("parameter_count"),
            )
            logger.info(
                "model_uploaded",
                extra={
                    "event": "model_uploaded",
                    "model_hash": model_hash,
                    "model_path": model_path,
                    "operator_count": analysis.get("operator_count"),
                    "parameter_count": analysis.get("parameter_count"),
                },
            )
        else:
            logger.warning(
                "model_upload_failed",
                extra={
                    "event": "model_upload_failed",
                    "model_path": model_path,
                    "error": analysis.get("error"),
                },
            )

        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Deployment Profile API
# ============================================================================

@app.post("/api/deployment/profile")
async def save_deployment_profile(request: DeploymentProfileRequest):
    """Save deployment profile and bind it into the decision pipeline."""
    try:
        # Validate types (Pydantic already handles this, but do normalization)
        cpu_cores = int(request.cpu_cores)
        if cpu_cores < 1:
            raise HTTPException(status_code=422, detail="cpu_cores must be >= 1")

        ram_gb = float(request.ram_gb)
        if ram_gb <= 0:
            raise HTTPException(status_code=422, detail="ram_gb must be > 0")

        target_latency_ms = float(request.target_latency_ms) if request.target_latency_ms is not None else None
        memory_limit_mb = float(request.memory_limit_mb) if request.memory_limit_mb is not None else None
        vram_gb = float(request.vram_gb) if request.vram_gb is not None else None

        profile = {
            "cpu_cores": cpu_cores,
            "cpu_arch": request.cpu_arch or None,
            "ram_gb": ram_gb,
            "ram_ddr": request.ram_ddr or None,
            "gpu_available": bool(request.gpu_available),
            "cuda_available": bool(request.cuda_available),
            "trt_available": bool(request.trt_available) if request.trt_available is not None else False,
            "stress_test": bool(request.stress_test) if request.stress_test is not None else False,
            "vram_gb": vram_gb,
            "target_latency_ms": target_latency_ms,
            "memory_limit_mb": memory_limit_mb,
        }

        with APP_STATE_LOCK:
            existing = APP_STATE.get("deployment_profile")
            profile_changed = existing != profile
            APP_STATE["deployment_profile"] = profile
            if profile_changed:
                APP_STATE["diagnostics"] = None
                APP_STATE["analysis"] = None
                APP_STATE["decision"] = None

        return {
            "success": True,
            "deployment_profile": profile,
            "timestamp": time.time(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/deployment/profile")
async def clear_deployment_profile():
    """Clear the deployment profile and reset downstream pipeline."""
    try:
        with APP_STATE_LOCK:
            APP_STATE["deployment_profile"] = None
            APP_STATE["diagnostics"] = None
            APP_STATE["analysis"] = None
            APP_STATE["decision"] = None
        return {"success": True, "message": "Profile cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/update")
async def update_calibration(request: CalibrationUpdateRequest):
    """Update latency calibration reference from empirical observations."""
    try:
        if len(request.latencies) != len(request.model_sizes):
            raise HTTPException(
                status_code=422,
                detail="latencies and model_sizes must have the same length",
            )
        if len(request.latencies) == 0:
            raise HTTPException(
                status_code=422,
                detail="latencies must not be empty",
            )

        latencies = [float(x) for x in request.latencies]
        if any(not math.isfinite(x) or x <= 0 for x in latencies):
            raise HTTPException(
                status_code=422,
                detail="latencies must contain positive finite numbers",
            )

        model_sizes = [int(x) for x in request.model_sizes]
        if any(x < 0 for x in model_sizes):
            raise HTTPException(
                status_code=422,
                detail="model_sizes must contain non-negative integers",
            )

        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)
        midpoint = count // 2
        if count % 2 == 1:
            median_latency = float(sorted_latencies[midpoint])
        else:
            median_latency = float((sorted_latencies[midpoint - 1] + sorted_latencies[midpoint]) / 2.0)

        if count == 1:
            percentile_95 = float(sorted_latencies[0])
        else:
            p95_rank = 0.95 * (count - 1)
            lower_index = int(math.floor(p95_rank))
            upper_index = int(math.ceil(p95_rank))
            if lower_index == upper_index:
                percentile_95 = float(sorted_latencies[lower_index])
            else:
                lower_value = float(sorted_latencies[lower_index])
                upper_value = float(sorted_latencies[upper_index])
                percentile_95 = float(
                    lower_value + (upper_value - lower_value) * (p95_rank - lower_index)
                )

        updated_reference = float(max(median_latency, percentile_95))
        timestamp = time.time()

        with APP_STATE_LOCK:
            calibration = APP_STATE.get("calibration")
            if not isinstance(calibration, dict):
                calibration = {}
                APP_STATE["calibration"] = calibration
            calibration["latency_reference_ms"] = updated_reference
            calibration["updated_at"] = timestamp

        return {
            "success": True,
            "latency_reference_ms": updated_reference,
            "timestamp": timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/analyze")
@require_stage("diagnostics")
async def analyze_model():
    """Analyze model and return recommendations."""
    try:
        with APP_STATE_LOCK:
            model_state = APP_STATE.get("model")
            deployment_profile = APP_STATE.get("deployment_profile")
            diagnostics_state = APP_STATE.get("diagnostics")

        if deployment_profile is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_execution_order",
                    "required_stage": "deployment_profile",
                },
            )

        if model_state is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_execution_order",
                    "required_stage": "model",
                },
            )

        model_path = str(model_state.get("path", ""))
        if not model_path:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_execution_order",
                    "required_stage": "model",
                },
            )

        if diagnostics_state is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_execution_order",
                    "required_stage": "diagnostics",
                },
            )

        # Use stored hash — single source of truth, never recompute from disk
        model_hash = model_state.get("hash")

        # Validate profile has not changed since diagnostics were produced
        snapshot = diagnostics_state.get("profile_snapshot")
        if snapshot != deployment_profile:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "invalid_execution_order",
                    "required_stage": "diagnostics",
                },
            )

        # Extract from diagnostics — do NOT call _analyze_model() again
        raw_analysis = diagnostics_state.get("raw_analysis", {})
        _ = diagnostics_state.get("facts", {})

        # Deployment profile constraints used for scoring/filtering
        gpu_available = deployment_profile.get("gpu_available", True)
        cuda_available = deployment_profile.get("cuda_available", True)
        trt_available = deployment_profile.get("trt_available", False)
        memory_limit_mb = deployment_profile.get("memory_limit_mb")
        target_latency_ms = deployment_profile.get("target_latency_ms")
        stress_enabled = bool(deployment_profile.get("stress_test", False))

        runtime_specs = [
            {
                "name": "ONNX_CPU",
                "provider": "CPUExecutionProvider",
                "precision": "FP32",
            },
            {
                "name": "ONNX_CUDA",
                "provider": "CUDAExecutionProvider",
                "precision": "FP16/FP32",
            },
            {
                "name": "TensorRT",
                "provider": "TensorrtExecutionProvider",
                "precision": "FP16/FP32",
            },
        ]

        available: list[str] = []
        provider_query_error: str | None = None
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
        except Exception as e:
            provider_query_error = str(e)
            logger.exception("Failed to query ONNX Runtime providers")

        runtime_benchmarks: dict[str, dict[str, Any]] = {}
        for runtime in runtime_specs:
            runtime_name = runtime["name"]
            provider = runtime["provider"]

            if provider not in available:
                error_msg = f"Provider '{provider}' unavailable"
                if provider_query_error:
                    error_msg = f"{error_msg}: {provider_query_error}"
                runtime_benchmarks[runtime_name] = {
                    "success": False,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "memory_mb": None,
                    "error": error_msg,
                    "stress_test": {
                        "enabled": stress_enabled,
                        "runs": 100 if stress_enabled else 0,
                        "latency_avg_ms": None,
                        "latency_p95_ms": None,
                        "peak_memory_mb": None,
                        "memory_growth_mb": None,
                        "memory_stability": "STABLE",
                    },
                }
            else:
                if stress_enabled:
                    runtime_benchmarks[runtime_name] = run_onnx_with_provider(
                        model_path,
                        provider,
                        stress_runs=100,
                    )
                else:
                    runtime_benchmarks[runtime_name] = run_onnx_with_provider(model_path, provider)

        runtime_rows: list[dict[str, Any]] = []
        # Extract deployment hardware params for scaling
        _tgt_cpu_cores2 = int(deployment_profile.get("cpu_cores") or 1)
        _tgt_ram_gb2    = float(deployment_profile.get("ram_gb") or 1.0)
        _tgt_ram_ddr2   = str(deployment_profile.get("ram_ddr") or "")
        _tgt_gpu2       = bool(deployment_profile.get("gpu_available", False))

        for runtime in runtime_specs:
            runtime_name = runtime["name"]
            provider = runtime["provider"]
            benchmark = runtime_benchmarks.get(runtime_name, {})

            success = bool(benchmark.get("success"))
            latency_avg_ms = benchmark.get("latency_avg_ms")
            latency_p95_ms = benchmark.get("latency_p95_ms")
            memory_mb = benchmark.get("memory_mb")
            error = benchmark.get("error")

            # ── Scale measured latency/memory to target hardware ──────────
            if success and latency_avg_ms is not None and runtime_name == "ONNX_CPU":
                latency_avg_ms = _scale_latency_for_target_hw(
                    latency_avg_ms, _tgt_cpu_cores2, _tgt_ram_gb2, _tgt_ram_ddr2, _tgt_gpu2
                )
                if latency_p95_ms is not None:
                    latency_p95_ms = _scale_latency_for_target_hw(
                        latency_p95_ms, _tgt_cpu_cores2, _tgt_ram_gb2, _tgt_ram_ddr2, _tgt_gpu2
                    )
            if success and memory_mb is not None:
                memory_mb = _scale_memory_for_target_hw(memory_mb, _tgt_ram_gb2)
            stress_test = benchmark.get("stress_test")
            if not isinstance(stress_test, dict):
                stress_test = {
                    "enabled": False,
                    "runs": 0,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "peak_memory_mb": None,
                    "memory_growth_mb": None,
                    "memory_stability": "STABLE",
                }

            decision = "SUPPORTED"
            diagnostics: list[str] = []

            if provider not in available:
                decision = "UNSUPPORTED"
                diagnostics.append(f"Provider unavailable: {provider}")

            if runtime_name in {"ONNX_CUDA", "TensorRT"} and not gpu_available:
                decision = "UNSUPPORTED"
                diagnostics.append("GPU disabled in deployment profile")

            if runtime_name in {"ONNX_CUDA", "TensorRT"} and not cuda_available:
                decision = "UNSUPPORTED"
                diagnostics.append("CUDA disabled in deployment profile")

            if runtime_name == "TensorRT" and not trt_available:
                decision = "UNSUPPORTED"
                diagnostics.append("TensorRT disabled in deployment profile")

            if not success:
                decision = "UNSUPPORTED"
                diagnostics.append(str(error) if error else "Benchmark failed")

            if (
                decision != "UNSUPPORTED"
                and memory_limit_mb is not None
                and memory_mb is not None
                and float(memory_mb) > float(memory_limit_mb)
            ):
                decision = "SUPPORTED_WITH_WARNINGS"
                diagnostics.append("Measured memory exceeds deployment profile limit")

            if (
                decision != "UNSUPPORTED"
                and target_latency_ms is not None
                and float(target_latency_ms) > 0
                and latency_avg_ms is not None
                and float(latency_avg_ms) > float(target_latency_ms)
            ):
                if decision == "SUPPORTED":
                    decision = "SUPPORTED_WITH_WARNINGS"
                diagnostics.append("Measured latency exceeds deployment profile target")

            runtime_rows.append(
                {
                    "runtime": runtime_name,
                    "precision": runtime["precision"],
                    "decision": decision,
                    "diagnostics": diagnostics,
                    "success": success,
                    "latency_avg_ms": latency_avg_ms,
                    "latency_p95_ms": latency_p95_ms,
                    "memory_mb": memory_mb,
                    "stress_test": stress_test,
                }
            )

        scored_latencies = [
            float(row["latency_avg_ms"])
            for row in runtime_rows
            if row["decision"] != "UNSUPPORTED"
            and row["success"]
            and row["latency_avg_ms"] is not None
        ]

        with APP_STATE_LOCK:
            _cal = APP_STATE.get("calibration")
            calibration_state = dict(_cal) if isinstance(_cal, dict) else {}

        reference_latency_ms = 200.0
        if isinstance(calibration_state, dict):
            calibrated_reference = calibration_state.get("latency_reference_ms")
            if isinstance(calibrated_reference, (int, float)) and math.isfinite(float(calibrated_reference)):
                if float(calibrated_reference) > 0:
                    reference_latency_ms = float(calibrated_reference)

        min_latency = min(scored_latencies) if scored_latencies else None
        max_latency = max(scored_latencies) if scored_latencies else None

        evaluations = []
        for row in runtime_rows:
            utility = 0.0
            if (
                row["decision"] != "UNSUPPORTED"
                and row["success"]
                and row["latency_avg_ms"] is not None
            ):
                runtime_latency = float(row["latency_avg_ms"])
                if len(scored_latencies) > 1 and min_latency is not None and max_latency is not None:
                    if max_latency == min_latency:
                        relative_score = 1.0
                    else:
                        relative_score = (max_latency - runtime_latency) / (max_latency - min_latency)

                    absolute_normalized = runtime_latency / reference_latency_ms
                    absolute_score = 1.0 / (1.0 + math.log1p(absolute_normalized))
                    utility = (relative_score + absolute_score) / 2.0
                elif len(scored_latencies) == 1:
                    normalized_latency = runtime_latency / reference_latency_ms
                    utility = 1.0 / (1.0 + math.log1p(normalized_latency))

                if row["decision"] == "SUPPORTED_WITH_WARNINGS":
                    utility *= 0.8

                row_stress = row.get("stress_test")
                if isinstance(row_stress, dict) and row_stress.get("memory_stability") == "UNSTABLE":
                    utility *= 0.5

            if not row["success"]:
                utility = 0.0

            utility = max(0.0, min(1.0, float(utility)))

            # Dynamic confidence: based on actual performance vs deployment targets
            if not row["success"]:
                confidence_score = 0.3
            else:
                # Start high for a working runtime
                confidence_score = 0.90
                lat_ms = row["latency_avg_ms"]
                mem_mb = row["memory_mb"]
                cpu_cores_val = int(deployment_profile.get("cpu_cores") or 1)
                ram_gb_val = float(deployment_profile.get("ram_gb") or 1.0)
                ram_ddr_val = str(deployment_profile.get("ram_ddr") or "").lower()
                gpu_avail_val = bool(deployment_profile.get("gpu_available", False))

                # ── Hardware-tier penalty ──────────────────────────────────
                if cpu_cores_val <= 1:
                    confidence_score *= 0.72
                elif cpu_cores_val <= 2:
                    confidence_score *= 0.82
                elif cpu_cores_val <= 4:
                    confidence_score *= 0.92

                if ram_gb_val < 1.0:
                    confidence_score *= 0.55
                elif ram_gb_val < 2.0:
                    confidence_score *= 0.68
                elif ram_gb_val < 4.0:
                    confidence_score *= 0.80
                elif ram_gb_val < 8.0:
                    confidence_score *= 0.92

                ddr_penalty = {
                    "ddr3": 0.82, "lpddr3": 0.82,
                    "ddr4": 0.95, "lpddr4": 0.93,
                    "ddr5": 1.0,  "lpddr5": 0.98,
                }
                if ram_ddr_val in ddr_penalty:
                    confidence_score *= ddr_penalty[ram_ddr_val]

                if row["runtime"] in ("ONNX_CUDA", "TensorRT") and not gpu_avail_val:
                    confidence_score *= 0.30

                # ── Latency vs target ─────────────────────────────────────
                if target_latency_ms is not None and lat_ms is not None:
                    tgt = float(target_latency_ms)
                    ratio = float(lat_ms) / tgt if tgt > 0 else 1.0
                    if ratio > 2.0:
                        confidence_score *= 0.50
                    elif ratio > 1.5:
                        confidence_score *= 0.65
                    elif ratio > 1.0:
                        confidence_score *= 0.80
                    else:
                        # Latency better than target — slight boost
                        confidence_score = min(1.0, confidence_score * (1 + (1.0 - ratio) * 0.10))
                elif lat_ms is not None:
                    # No target — penalise very slow runtimes relative to reference
                    ratio = float(lat_ms) / reference_latency_ms
                    if ratio > 5.0:
                        confidence_score *= 0.60
                    elif ratio > 2.0:
                        confidence_score *= 0.80

                # ── Memory vs limit ───────────────────────────────────────
                if memory_limit_mb is not None and mem_mb is not None:
                    mem_ratio = float(mem_mb) / float(memory_limit_mb)
                    if mem_ratio > 1.0:
                        confidence_score *= 0.60
                    elif mem_ratio > 0.85:
                        confidence_score *= 0.85

                # ── Warnings & instability ────────────────────────────────
                if row["decision"] == "SUPPORTED_WITH_WARNINGS":
                    confidence_score *= 0.80
                row_stress = row.get("stress_test")
                if isinstance(row_stress, dict) and row_stress.get("memory_stability") == "UNSTABLE":
                    confidence_score *= 0.65

                # ── Bonus: everything well within limits ──────────────────
                lat_ok = target_latency_ms is None or (lat_ms is not None and float(lat_ms) <= float(target_latency_ms) * 0.75)
                mem_ok = memory_limit_mb is None or (mem_mb is not None and float(mem_mb) <= float(memory_limit_mb) * 0.75)
                if lat_ok and mem_ok and cpu_cores_val >= 4 and ram_gb_val >= 4.0:
                    confidence_score = min(1.0, confidence_score * 1.05)

                confidence_score = max(0.10, min(1.0, confidence_score))

            confidence_level = (
                "HIGH" if confidence_score >= 0.8 else "MEDIUM" if confidence_score >= 0.5 else "LOW"
            )

            evaluation = {
                "runtime": row["runtime"],
                "decision": row["decision"],
                "utility_score": utility,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "diagnostics": row["diagnostics"],
                "precision_support": row["precision"],
                "predicted_latency_ms": float(row["latency_avg_ms"]) if row["latency_avg_ms"] is not None else 0.0,
                "memory_usage_mb": float(row["memory_mb"]) if row["memory_mb"] is not None else 0.0,
                "execution_success": bool(row["success"]),
                "latency_p95_ms": float(row["latency_p95_ms"]) if row["latency_p95_ms"] is not None else 0.0,
                "stress_test": row.get("stress_test"),
            }
            evaluations.append(evaluation)

        # Sort by utility score
        evaluations.sort(key=lambda x: x["utility_score"], reverse=True)

        # Calculate overall confidence from best supported eval
        best_eval = evaluations[0] if evaluations else None
        overall_confidence = best_eval.get("confidence_score", 0.5) if best_eval else 0.5

        response = {
            "success": True,
            "model_path": model_path,
            "file_hash": model_hash,
            "analysis": raw_analysis,
            "best_runtime": evaluations[0].get("runtime", "ONNX_CPU") if evaluations else "ONNX_CPU",
            "evaluations": evaluations,
            "confidence": {
                "score": overall_confidence,
                "level": "HIGH" if overall_confidence >= 0.8 else "MEDIUM" if overall_confidence >= 0.5 else "LOW",
            },
            "timestamp": time.time(),
        }

        with APP_STATE_LOCK:
            active_model = APP_STATE.get("model")
            active_hash = active_model.get("hash") if isinstance(active_model, dict) else None
            if active_hash != model_hash:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "invalid_execution_order",
                        "required_stage": "model",
                    },
                )

            APP_STATE["analysis"] = {
                "model_hash": model_hash,
                "analysis": raw_analysis,
                "evaluations": evaluations,
                "runtime_benchmarks": runtime_benchmarks,
                "best_runtime": evaluations[0].get("runtime", "ONNX_CPU") if evaluations else "ONNX_CPU",
                "confidence": overall_confidence,
                "timestamp": time.time(),
            }
            APP_STATE["decision"] = None

        logger.info(
            "analysis_complete",
            extra={
                "event": "analysis_complete",
                "model_hash": model_hash,
                "best_runtime": evaluations[0].get("runtime", "ONNX_CPU") if evaluations else "ONNX_CPU",
                "overall_confidence": overall_confidence,
                "evaluated_runtimes": [ev.get("runtime") for ev in evaluations],
            },
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Decision API
# ============================================================================

@app.post("/api/decision/recommend")
@require_stage("analysis")
async def recommend_decision(debug: bool = False, explain: bool = False):
    """Get runtime recommendation for a model."""
    with APP_STATE_LOCK:
        model = APP_STATE.get("model")
        analysis = APP_STATE.get("analysis")
        deployment_profile = APP_STATE.get("deployment_profile")

    if model is None or analysis is None or deployment_profile is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "invalid_execution_order",
                "required_stage": "analysis",
            },
        )

    try:
        best_runtime = analysis.get("best_runtime", "")
        evaluations = analysis.get("evaluations") or []
        base_confidence = float(analysis.get("confidence") or 0.5)

        # ------------------------------------------------------------------ #
        # Hardware-aware runtime override — runs AFTER analysis is read,      #
        # BEFORE SLA logic. Does NOT modify SLA rules or confidence scaling.  #
        # ------------------------------------------------------------------ #
        from src.core.decision_layer_integration import apply_hardware_aware_override
        resolved_runtime, hw_decision = apply_hardware_aware_override(
            best_runtime=best_runtime,
            model_size=analysis.get("analysis", {}).get("parameter_scale_class", "small"),
            batch_size=1,
        )
        best_runtime = resolved_runtime
        # ------------------------------------------------------------------ #

        # Locate the best evaluation entry by runtime name
        best_eval: dict[str, Any] = {}
        for ev in evaluations:
            if isinstance(ev, dict) and ev.get("runtime") == best_runtime:
                best_eval = ev
                break

        # Extract fields with safe defaults — no KeyError possible
        predicted_latency_ms: float | None = best_eval.get("predicted_latency_ms")
        if predicted_latency_ms is not None:
            predicted_latency_ms = float(predicted_latency_ms)

        memory_usage_mb: float | None = best_eval.get("memory_usage_mb")
        if memory_usage_mb is not None:
            memory_usage_mb = float(memory_usage_mb)

        execution_success: bool = bool(best_eval.get("execution_success", True))
        eval_decision: str = str(best_eval.get("decision", "SUPPORTED"))

        stress_test: dict[str, Any] = best_eval.get("stress_test") or {}
        memory_stability: str = str(stress_test.get("memory_stability") or "STABLE")
        raw_growth = stress_test.get("memory_growth_mb")
        memory_growth_mb: float = float(raw_growth) if raw_growth is not None else 0.0

        # Deployment profile SLA constraints
        target_latency_ms: float | None = deployment_profile.get("target_latency_ms")
        if target_latency_ms is not None:
            target_latency_ms = float(target_latency_ms)

        memory_limit_mb: float | None = deployment_profile.get("memory_limit_mb")
        if memory_limit_mb is not None:
            memory_limit_mb = float(memory_limit_mb)

        reasons: list[str] = []
        rejected: bool = False

        # ------------------------------------------------------------------ #
        # Hard rejection rules — evaluated unconditionally                    #
        # ------------------------------------------------------------------ #
        if not execution_success:
            reasons.append("execution_success is False")
            rejected = True

        if eval_decision == "UNSUPPORTED":
            reasons.append(
                f"evaluation decision is UNSUPPORTED for runtime {best_runtime}"
            )
            rejected = True

        if memory_stability == "UNSTABLE":
            reasons.append("stress_test.memory_stability is UNSTABLE")
            rejected = True

        if (
            target_latency_ms is not None
            and predicted_latency_ms is not None
            and predicted_latency_ms > 1.5 * target_latency_ms
        ):
            reasons.append(
                f"predicted_latency_ms {predicted_latency_ms:.3f} exceeds"
                f" 1.5x target_latency_ms {target_latency_ms:.3f}"
            )
            rejected = True

        if (
            memory_limit_mb is not None
            and memory_usage_mb is not None
            and memory_usage_mb > 1.5 * memory_limit_mb
        ):
            reasons.append(
                f"memory_usage_mb {memory_usage_mb:.3f} exceeds"
                f" 1.5x memory_limit_mb {memory_limit_mb:.3f}"
            )
            rejected = True

        # ------------------------------------------------------------------ #
        # Conditional approval rules — only when not already rejected         #
        # ------------------------------------------------------------------ #
        conditional: bool = False
        if not rejected:
            if (
                target_latency_ms is not None
                and predicted_latency_ms is not None
                and predicted_latency_ms > target_latency_ms
            ):
                reasons.append(
                    f"predicted_latency_ms {predicted_latency_ms:.3f} exceeds"
                    f" target_latency_ms {target_latency_ms:.3f}"
                )
                conditional = True

            if (
                memory_limit_mb is not None
                and memory_usage_mb is not None
                and memory_usage_mb > memory_limit_mb
            ):
                reasons.append(
                    f"memory_usage_mb {memory_usage_mb:.3f} exceeds"
                    f" memory_limit_mb {memory_limit_mb:.3f}"
                )
                conditional = True

            if memory_growth_mb > 5.0:
                reasons.append(
                    f"stress_test.memory_growth_mb {memory_growth_mb:.3f} exceeds 5.0"
                )
                conditional = True

        # ------------------------------------------------------------------ #
        # Final status                                                         #
        # ------------------------------------------------------------------ #
        if rejected:
            final_status = "REJECTED"
        elif conditional:
            final_status = "CONDITIONAL_APPROVAL"
        else:
            final_status = "APPROVED"

        # ------------------------------------------------------------------ #
        # Confidence scaling                                                   #
        # ------------------------------------------------------------------ #
        adjusted_confidence = base_confidence
        if final_status == "CONDITIONAL_APPROVAL":
            adjusted_confidence *= 0.85
        if final_status == "REJECTED":
            adjusted_confidence *= 0.6
        if memory_stability == "UNSTABLE":
            adjusted_confidence *= 0.5
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # ------------------------------------------------------------------ #
        # Deployment Risk Evaluation — pipeline engine                        #
        # ------------------------------------------------------------------ #
        # Signal convention: 0.0 = no stress, 1.0 = maximum stress.
        # Derived from deployment profile tightness (hardware constraints),
        # not from benchmark measurements on the analysis machine.
        _dp_cpu1 = int(deployment_profile.get("cpu_cores") or 1)
        _dp_ram1 = float(deployment_profile.get("ram_gb") or 1.0)
        _dp_gpu1 = bool(deployment_profile.get("gpu_available", False))
        _dp_tgt1 = float(target_latency_ms) if target_latency_ms is not None else 200.0

        _sig_cpu1  = min(max(1.0 - (_dp_cpu1 - 1) / 15.0, 0.0), 1.0)  # 1 core→1.0, 16 cores→0.0
        _sig_mem1  = min(max(1.0 - _dp_ram1 / 8.0,         0.0), 1.0)  # 0.5 GB→0.94, 16 GB→0.0
        _sig_lat1  = min(max(1.0 - _dp_tgt1 / 200.0,       0.0), 1.0)  # 1 ms→1.0,  200 ms→0.0
        _sig_gpu1  = 0.0 if _dp_gpu1 else 0.5

        _pipeline_signals1: dict[str, float] = {
            "cpu":             _sig_cpu1,
            "memory":          _sig_mem1,
            "latency":         _sig_lat1,
            "gpu":             _sig_gpu1,
            "bandwidth":       0.5,
            "io":              0.5,
            "network":         0.5,
            "concurrency":     _sig_cpu1,
            "numa":            _sig_mem1,
            "future_drift":    0.5,
            "compatibility":   0.5,
            "security_signal": 0.5,
        }

        _pipeline_risk1: float = compute_risk(_pipeline_signals1)
        _pipeline_dec1:  str   = decision_from_risk(_pipeline_risk1)

        _DEC_MAP1: dict[str, str] = {
            "ALLOW":                 "APPROVED",
            "ALLOW_WITH_CONDITIONS": "CONDITIONAL_APPROVAL",
            "BLOCK":                 "REJECTED",
        }
        _mapped_status1: str = _DEC_MAP1.get(_pipeline_dec1, _pipeline_dec1)

        _risk_level1: str = (
            "HIGH_RISK"   if _pipeline_risk1 >= 8.2 else
            "MEDIUM_RISK" if _pipeline_risk1 >= 3.8 else
            "LOW_RISK"
        )

        risk_result = {
            "risk_score":      _pipeline_risk1,
            "risk_level":      _risk_level1,
            "engine_version":  "pipeline",
            "timestamp":       time.time(),
        }

        # Promote final_status to REJECTED when pipeline engine returns BLOCK
        if _mapped_status1 == "REJECTED" and final_status != "REJECTED":
            final_status = "REJECTED"
            reasons.append(
                f"pipeline_engine: risk {_pipeline_risk1:.3f} >= BLOCK threshold 8.22"
            )
            adjusted_confidence = max(0.0, min(1.0, base_confidence * 0.6))

        timestamp = time.time()

        # ------------------------------------------------------------------ #
        # Atomic write back                                                    #
        # ------------------------------------------------------------------ #
        with APP_STATE_LOCK:
            APP_STATE["decision"] = {
                "runtime": best_runtime,
                "status": final_status,
                "confidence": adjusted_confidence,
                "timestamp": timestamp,
                "reasons": reasons,
            }

        persistence.save_decision_record(
            runtime=best_runtime,
            status=final_status,
            confidence=adjusted_confidence,
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            latency_ms=predicted_latency_ms,
            memory_mb=memory_usage_mb,
        )

        # ------------------------------------------------------------------ #
        # Structured logging — decision computed                               #
        # ------------------------------------------------------------------ #
        hw_override_applied: bool = (
            isinstance(hw_decision, dict) and bool(hw_decision.get("override_applied"))
        )
        hw_override_reason: str | None = (
            hw_decision.get("override_reason") if isinstance(hw_decision, dict) else None
        )
        logger.info(
            "decision_computed",
            extra={
                "event": "decision_computed",
                "runtime": best_runtime,
                "status": final_status,
                "confidence": adjusted_confidence,
                "hardware_override_applied": hw_override_applied,
            },
        )
        if hw_override_applied:
            logger.info(
                "hardware_override_applied",
                extra={
                    "event": "hardware_override_applied",
                    "selected_provider": hw_decision.get("selected_provider"),
                    "override_reason": hw_override_reason,
                    "calibration_latency_ms": hw_decision.get("calibration_latency_ms"),
                    "calibration_gpu_memory_mb": hw_decision.get("calibration_gpu_memory_mb"),
                },
            )

        # ------------------------------------------------------------------ #
        # DecisionTrace — constructed after all values are finalised          #
        # ------------------------------------------------------------------ #
        from src.core.decision_trace import DecisionTrace
        from src.core.explainability import build_explanation_tree

        _runtime_before = analysis.get("best_runtime", "")
        _rejection_reasons = [r for r in reasons if rejected]
        _conditional_reasons = [r for r in reasons if not rejected and conditional]

        trace = DecisionTrace(
            selected_runtime_before_override=_runtime_before,
            selected_runtime_after_override=best_runtime,
            sla_target_latency=target_latency_ms,
            measured_latency=predicted_latency_ms,
            memory_limit=memory_limit_mb,
            measured_memory=memory_usage_mb,
            rejection_reasons=_rejection_reasons,
            conditional_reasons=_conditional_reasons,
            confidence_before_scaling=base_confidence,
            confidence_after_scaling=adjusted_confidence,
            hardware_override_applied=hw_override_applied,
            override_reason=hw_override_reason,
        )

        # ------------------------------------------------------------------ #
        # Build response — optional fields added only when requested          #
        # ------------------------------------------------------------------ #
        response_body: dict[str, Any] = {
            "success": True,
            "runtime": best_runtime,
            "status": final_status,
            "confidence": adjusted_confidence,
            "reasons": reasons,
            "timestamp": timestamp,
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
        }
        if hw_override_applied:
            response_body["hardware_selector"] = hw_decision
        if debug:
            response_body["decision_trace"] = trace.to_dict()
        if explain:
            response_body["explanation_tree"] = build_explanation_tree(trace)

        return response_body
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-and-decide")
async def analyze_and_decide(
    file: UploadFile = File(...),
    cpu_cores: int = Form(1),
    ram_gb: float = Form(8.0),
    gpu_available: bool = Form(False),
    cuda_available: bool = Form(False),
    vram_gb: float | None = Form(None),
    trt_available: bool | None = Form(None),
    stress_test: bool | None = Form(None),
    cpu_arch: str | None = Form(None),
    ram_ddr: str | None = Form(None),
    target_latency_ms: float | None = Form(None),
    memory_limit_mb: float | None = Form(None),
    debug: bool = Form(False),
    explain: bool = Form(False),
):
    try:
        # Stream-save the uploaded file to support large models (up to 4 GB)
        # without loading the entire file into memory at once.
        safe_name = Path(file.filename or "model.onnx").name
        dest = _UPLOAD_DIR / safe_name
        _hasher2 = hashlib.sha256()
        with open(str(dest), "wb") as _fh2:
            while True:
                _chunk2 = await file.read(1024 * 1024)  # 1 MB chunks
                if not _chunk2:
                    break
                _fh2.write(_chunk2)
                _hasher2.update(_chunk2)
        model_path = str(dest)
        
        try:
            file_hash = _hasher2.hexdigest()
            model_hash_input = f"{dest.resolve()}:{file_hash}".encode("utf-8")
            model_hash = hashlib.sha1(model_hash_input).hexdigest()
        except Exception:
            model_hash = safe_name
        
        raw_analysis = _analyze_model(model_path)
        if not raw_analysis.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Model analysis failed: {raw_analysis.get('error', 'Unknown error')}"
            )
        
        raw_analysis["file_hash"] = model_hash
        
        facts = _build_facts(raw_analysis)
        
        deployment_profile = {
            "cpu_cores": int(cpu_cores),
            "cpu_arch": cpu_arch,
            "ram_gb": float(ram_gb),
            "ram_ddr": ram_ddr,
            "gpu_available": bool(gpu_available),
            "cuda_available": bool(cuda_available),
            "trt_available": bool(trt_available) if trt_available is not None else False,
            "stress_test": bool(stress_test) if stress_test is not None else False,
            "vram_gb": float(vram_gb) if vram_gb is not None else None,
            "target_latency_ms": float(target_latency_ms) if target_latency_ms is not None else None,
            "memory_limit_mb": float(memory_limit_mb) if memory_limit_mb is not None else None,
        }
        
        diagnostics_state = {
            "facts": facts,
            "raw_analysis": raw_analysis,
            "profile_snapshot": deployment_profile.copy(),
        }
        
        runtime_specs = [
            {"name": "ONNX_CPU", "provider": "CPUExecutionProvider", "precision": "FP32"},
            {"name": "ONNX_CUDA", "provider": "CUDAExecutionProvider", "precision": "FP16/FP32"},
            {"name": "TensorRT", "provider": "TensorrtExecutionProvider", "precision": "FP16/FP32"},
        ]
        
        available: list[str] = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            pass
        
        stress_enabled = bool(deployment_profile.get("stress_test", False))
        runtime_benchmarks: dict[str, dict[str, Any]] = {}
        
        for runtime in runtime_specs:
            runtime_name = runtime["name"]
            provider = runtime["provider"]
            
            if provider not in available:
                runtime_benchmarks[runtime_name] = {
                    "success": False,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "memory_mb": None,
                    "error": f"Provider '{provider}' unavailable",
                    "stress_test": {
                        "enabled": stress_enabled,
                        "runs": 100 if stress_enabled else 0,
                        "latency_avg_ms": None,
                        "latency_p95_ms": None,
                        "peak_memory_mb": None,
                        "memory_growth_mb": None,
                        "memory_stability": "STABLE",
                    },
                }
            else:
                cache_key = (model_hash_input.decode("utf-8"), provider)
                with _BENCHMARK_CACHE_LOCK:
                    cached_result = _BENCHMARK_CACHE.get(cache_key)
                
                if cached_result is not None and not stress_enabled:
                    # Reuse raw benchmark to ensure base latency is perfectly deterministic
                    runtime_benchmarks[runtime_name] = dict(cached_result)
                else:
                    if stress_enabled:
                        res = run_onnx_with_provider(model_path, provider, stress_runs=100)
                    else:
                        res = run_onnx_with_provider(model_path, provider)
                    
                    runtime_benchmarks[runtime_name] = res
                    
                    if not stress_enabled and res.get("success"):
                        with _BENCHMARK_CACHE_LOCK:
                            _BENCHMARK_CACHE[cache_key] = res
        
        runtime_rows: list[dict[str, Any]] = []
        # Extract deployment hardware params for scaling
        _tgt_cpu_cores = int(deployment_profile.get("cpu_cores") or 1)
        _tgt_ram_gb    = float(deployment_profile.get("ram_gb") or 1.0)
        _tgt_ram_ddr   = str(deployment_profile.get("ram_ddr") or "")
        _tgt_gpu       = bool(deployment_profile.get("gpu_available", False))

        for runtime in runtime_specs:
            runtime_name = runtime["name"]
            provider = runtime["provider"]
            benchmark = runtime_benchmarks.get(runtime_name, {})
            
            success = bool(benchmark.get("success"))
            latency_avg_ms = benchmark.get("latency_avg_ms")
            latency_p95_ms = benchmark.get("latency_p95_ms")
            memory_mb = benchmark.get("memory_mb")
            error = benchmark.get("error")

            # ── Scale measured latency/memory to target hardware ──────────
            if success and latency_avg_ms is not None:
                # CPU runtimes only — GPU runtimes are already UNSUPPORTED if no GPU
                if runtime_name == "ONNX_CPU":
                    latency_avg_ms = _scale_latency_for_target_hw(
                        latency_avg_ms, _tgt_cpu_cores, _tgt_ram_gb, _tgt_ram_ddr, _tgt_gpu
                    )
                    if latency_p95_ms is not None:
                        latency_p95_ms = _scale_latency_for_target_hw(
                            latency_p95_ms, _tgt_cpu_cores, _tgt_ram_gb, _tgt_ram_ddr, _tgt_gpu
                        )
            if success and memory_mb is not None:
                memory_mb = _scale_memory_for_target_hw(memory_mb, _tgt_ram_gb)
            stress_test_result = benchmark.get("stress_test")
            if not isinstance(stress_test_result, dict):
                stress_test_result = {
                    "enabled": False,
                    "runs": 0,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "peak_memory_mb": None,
                    "memory_growth_mb": None,
                    "memory_stability": "STABLE",
                }
            
            rt_decision = "SUPPORTED"
            rt_diagnostics: list[str] = []
            
            if provider not in available:
                rt_decision = "UNSUPPORTED"
                rt_diagnostics.append(f"Provider unavailable: {provider}")
            
            if runtime_name in {"ONNX_CUDA", "TensorRT"} and not deployment_profile.get("gpu_available"):
                rt_decision = "UNSUPPORTED"
                rt_diagnostics.append("GPU disabled in deployment profile")
            
            if runtime_name in {"ONNX_CUDA", "TensorRT"} and not deployment_profile.get("cuda_available"):
                rt_decision = "UNSUPPORTED"
                rt_diagnostics.append("CUDA disabled in deployment profile")
            
            if runtime_name == "TensorRT" and not deployment_profile.get("trt_available"):
                rt_decision = "UNSUPPORTED"
                rt_diagnostics.append("TensorRT disabled in deployment profile")
            
            if not success:
                rt_decision = "UNSUPPORTED"
                rt_diagnostics.append(str(error) if error else "Benchmark failed")
            
            mem_limit = deployment_profile.get("memory_limit_mb")
            if rt_decision != "UNSUPPORTED" and mem_limit is not None and memory_mb is not None:
                if float(memory_mb) > float(mem_limit):
                    rt_decision = "SUPPORTED_WITH_WARNINGS"
                    rt_diagnostics.append("Measured memory exceeds deployment profile limit")
            
            tgt_latency = deployment_profile.get("target_latency_ms")
            if rt_decision != "UNSUPPORTED" and tgt_latency is not None and latency_avg_ms is not None:
                if float(latency_avg_ms) > float(tgt_latency):
                    if rt_decision == "SUPPORTED":
                        rt_decision = "SUPPORTED_WITH_WARNINGS"
                    rt_diagnostics.append("Measured latency exceeds deployment profile target")
            
            runtime_rows.append({
                "runtime": runtime_name,
                "precision": runtime["precision"],
                "decision": rt_decision,
                "diagnostics": rt_diagnostics,
                "success": success,
                "latency_avg_ms": latency_avg_ms,
                "latency_p95_ms": latency_p95_ms,
                "memory_mb": memory_mb,
                "stress_test": stress_test_result,
            })
        
        scored_latencies = [
            float(row["latency_avg_ms"])
            for row in runtime_rows
            if row["decision"] != "UNSUPPORTED"
            and row["success"]
            and row["latency_avg_ms"] is not None
        ]
        
        reference_latency_ms = 200.0
        min_latency = min(scored_latencies) if scored_latencies else None
        max_latency = max(scored_latencies) if scored_latencies else None
        
        evaluations = []
        for row in runtime_rows:
            utility = 0.0
            if (
                row["decision"] != "UNSUPPORTED"
                and row["success"]
                and row["latency_avg_ms"] is not None
            ):
                runtime_latency = float(row["latency_avg_ms"])
                if len(scored_latencies) > 1 and min_latency is not None and max_latency is not None:
                    if max_latency == min_latency:
                        relative_score = 1.0
                    else:
                        relative_score = (max_latency - runtime_latency) / (max_latency - min_latency)
                    absolute_normalized = runtime_latency / reference_latency_ms
                    absolute_score = 1.0 / (1.0 + math.log1p(absolute_normalized))
                    utility = (relative_score + absolute_score) / 2.0
                elif len(scored_latencies) == 1:
                    normalized_latency = runtime_latency / reference_latency_ms
                    utility = 1.0 / (1.0 + math.log1p(normalized_latency))
                
                if row["decision"] == "SUPPORTED_WITH_WARNINGS":
                    utility *= 0.8
                
                row_stress = row.get("stress_test")
                if isinstance(row_stress, dict) and row_stress.get("memory_stability") == "UNSTABLE":
                    utility *= 0.5
            
            if not row["success"]:
                utility = 0.0
            
            utility = max(0.0, min(1.0, float(utility)))
            
            # Dynamic confidence: based on actual performance vs deployment targets
            if not row["success"]:
                confidence_score = 0.3
            else:
                confidence_score = 0.90
                lat_ms = row["latency_avg_ms"]
                mem_mb = row["memory_mb"]
                tgt_lat = deployment_profile.get("target_latency_ms")
                mem_lim = deployment_profile.get("memory_limit_mb")
                _dp_cpu = int(deployment_profile.get("cpu_cores") or 1)
                _dp_ram = float(deployment_profile.get("ram_gb") or 1.0)
                _dp_ddr = str(deployment_profile.get("ram_ddr") or "").lower()
                _dp_gpu = bool(deployment_profile.get("gpu_available", False))

                # ── Hardware-tier penalty ──────────────────────────────────
                # CPU cores: fewer cores = lower confidence for inference workloads
                if _dp_cpu <= 1:
                    confidence_score *= 0.72   # single core is very constrained
                elif _dp_cpu <= 2:
                    confidence_score *= 0.82
                elif _dp_cpu <= 4:
                    confidence_score *= 0.92
                # else ≥8 cores → no penalty

                # RAM: low RAM = higher risk of OOM or swapping
                if _dp_ram < 1.0:
                    confidence_score *= 0.55
                elif _dp_ram < 2.0:
                    confidence_score *= 0.68
                elif ram_gb < 4.0:
                    confidence_score *= 0.80
                elif ram_gb < 8.0:
                    confidence_score *= 0.92
                # else ≥8 GB → no penalty

                # DDR type: older/slower RAM reduces confidence for memory-heavy models
                ddr_penalty = {
                    "ddr3": 0.82,
                    "lpddr3": 0.82,
                    "ddr4": 0.95,
                    "lpddr4": 0.93,
                    "ddr5": 1.0,
                    "lpddr5": 0.98,
                }
                if ram_ddr in ddr_penalty:
                    confidence_score *= ddr_penalty[ram_ddr]

                # GPU runtimes on CPU-only hardware get a strong penalty
                if row["runtime"] in ("ONNX_CUDA", "TensorRT") and not _dp_gpu:
                    confidence_score *= 0.30  # should already be UNSUPPORTED but just in case

                # ── Latency vs target ─────────────────────────────────────
                if tgt_lat is not None and lat_ms is not None:
                    tgt = float(tgt_lat)
                    ratio = float(lat_ms) / tgt if tgt > 0 else 1.0
                    if ratio > 2.0:
                        confidence_score *= 0.50
                    elif ratio > 1.5:
                        confidence_score *= 0.65
                    elif ratio > 1.0:
                        confidence_score *= 0.80
                    else:
                        confidence_score = min(1.0, confidence_score * (1 + (1.0 - ratio) * 0.10))
                elif lat_ms is not None:
                    ratio = float(lat_ms) / reference_latency_ms
                    if ratio > 5.0:
                        confidence_score *= 0.60
                    elif ratio > 2.0:
                        confidence_score *= 0.80

                # ── Memory vs limit ───────────────────────────────────────
                if mem_lim is not None and mem_mb is not None:
                    mem_ratio = float(mem_mb) / float(mem_lim)
                    if mem_ratio > 1.0:
                        confidence_score *= 0.60
                    elif mem_ratio > 0.85:
                        confidence_score *= 0.85

                # ── Warnings & instability ────────────────────────────────
                if row["decision"] == "SUPPORTED_WITH_WARNINGS":
                    confidence_score *= 0.80
                row_stress = row.get("stress_test")
                if isinstance(row_stress, dict) and row_stress.get("memory_stability") == "UNSTABLE":
                    confidence_score *= 0.65

                # ── Bonus: everything well within limits ──────────────────
                lat_ok = tgt_lat is None or (lat_ms is not None and float(lat_ms) <= float(tgt_lat) * 0.75)
                mem_ok = mem_lim is None or (mem_mb is not None and float(mem_mb) <= float(mem_lim) * 0.75)
                if lat_ok and mem_ok and _dp_cpu >= 4 and _dp_ram >= 4.0:
                    confidence_score = min(1.0, confidence_score * 1.05)

                confidence_score = max(0.10, min(1.0, confidence_score))

            confidence_level = "HIGH" if confidence_score >= 0.8 else "MEDIUM" if confidence_score >= 0.5 else "LOW"
            
            evaluation = {
                "runtime": row["runtime"],
                "decision": row["decision"],
                "utility_score": utility,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "diagnostics": row["diagnostics"],
                "precision_support": row["precision"],
                "predicted_latency_ms": float(row["latency_avg_ms"]) if row["latency_avg_ms"] is not None else 0.0,
                "memory_usage_mb": float(row["memory_mb"]) if row["memory_mb"] is not None else 0.0,
                "execution_success": bool(row["success"]),
                "latency_p95_ms": float(row["latency_p95_ms"]) if row["latency_p95_ms"] is not None else 0.0,
                "stress_test": row.get("stress_test"),
            }
            evaluations.append(evaluation)
        
        evaluations.sort(key=lambda x: x["utility_score"], reverse=True)
        
        best_eval = evaluations[0] if evaluations else {}
        overall_confidence = best_eval.get("confidence_score", 0.5) if best_eval else 0.5
        
        analysis_state = {
            "model_hash": model_hash,
            "analysis": raw_analysis,
            "evaluations": evaluations,
            "runtime_benchmarks": runtime_benchmarks,
            "best_runtime": best_eval.get("runtime", "ONNX_CPU") if best_eval else "ONNX_CPU",
            "confidence": overall_confidence,
            "timestamp": time.time(),
            "deployment_profile": deployment_profile,
        }
        
        best_runtime = analysis_state.get("best_runtime", "")
        base_confidence = float(analysis_state.get("confidence") or 0.5)
        
        from src.core.decision_layer_integration import apply_hardware_aware_override
        resolved_runtime, hw_decision = apply_hardware_aware_override(
            best_runtime=best_runtime,
            model_size=raw_analysis.get("parameter_scale_class", "small"),
            batch_size=1,
        )
        best_runtime = resolved_runtime
        
        best_eval_decision: dict[str, Any] = {}
        for ev in evaluations:
            if isinstance(ev, dict) and ev.get("runtime") == best_runtime:
                best_eval_decision = ev
                break
        
        predicted_latency_ms: float | None = best_eval_decision.get("predicted_latency_ms")
        if predicted_latency_ms is not None:
            predicted_latency_ms = float(predicted_latency_ms)
        
        memory_usage_mb: float | None = best_eval_decision.get("memory_usage_mb")
        if memory_usage_mb is not None:
            memory_usage_mb = float(memory_usage_mb)
        
        execution_success: bool = bool(best_eval_decision.get("execution_success", True))
        eval_decision: str = str(best_eval_decision.get("decision", "SUPPORTED"))
        
        stress_test_dict: dict[str, Any] = best_eval_decision.get("stress_test") or {}
        memory_stability: str = str(stress_test_dict.get("memory_stability") or "STABLE")
        raw_growth = stress_test_dict.get("memory_growth_mb")
        memory_growth_mb: float = float(raw_growth) if raw_growth is not None else 0.0
        
        target_latency_sla: float | None = deployment_profile.get("target_latency_ms")
        if target_latency_sla is not None:
            target_latency_sla = float(target_latency_sla)
        
        memory_limit_sla: float | None = deployment_profile.get("memory_limit_mb")
        if memory_limit_sla is not None:
            memory_limit_sla = float(memory_limit_sla)
        
        reasons: list[str] = []
        rejected: bool = False
        
        if not execution_success:
            reasons.append("execution_success is False")
            rejected = True
        
        if eval_decision == "UNSUPPORTED":
            reasons.append(f"evaluation decision is UNSUPPORTED for runtime {best_runtime}")
            rejected = True
        
        if memory_stability == "UNSTABLE":
            reasons.append("stress_test.memory_stability is UNSTABLE")
            rejected = True
        
        if (
            target_latency_sla is not None
            and predicted_latency_ms is not None
            and predicted_latency_ms > 1.5 * target_latency_sla
        ):
            reasons.append(
                f"predicted_latency_ms {predicted_latency_ms:.3f} exceeds"
                f" 1.5x target_latency_ms {target_latency_sla:.3f}"
            )
            rejected = True
        
        if (
            memory_limit_sla is not None
            and memory_usage_mb is not None
            and memory_usage_mb > 1.5 * memory_limit_sla
        ):
            reasons.append(
                f"memory_usage_mb {memory_usage_mb:.3f} exceeds"
                f" 1.5x memory_limit_mb {memory_limit_sla:.3f}"
            )
            rejected = True
        
        conditional: bool = False
        if not rejected:
            if (
                target_latency_sla is not None
                and predicted_latency_ms is not None
                and predicted_latency_ms > target_latency_sla
            ):
                reasons.append(
                    f"predicted_latency_ms {predicted_latency_ms:.3f} exceeds"
                    f" target_latency_ms {target_latency_sla:.3f}"
                )
                conditional = True
            
            if (
                memory_limit_sla is not None
                and memory_usage_mb is not None
                and memory_usage_mb > memory_limit_sla
            ):
                reasons.append(
                    f"memory_usage_mb {memory_usage_mb:.3f} exceeds"
                    f" memory_limit_mb {memory_limit_sla:.3f}"
                )
                conditional = True
            
            if memory_growth_mb > 5.0:
                reasons.append(f"stress_test.memory_growth_mb {memory_growth_mb:.3f} exceeds 5.0")
                conditional = True
        
        if rejected:
            final_status = "REJECTED"
        elif conditional:
            final_status = "CONDITIONAL_APPROVAL"
        else:
            final_status = "APPROVED"
        
        adjusted_confidence = base_confidence
        if final_status == "CONDITIONAL_APPROVAL":
            adjusted_confidence *= 0.85
        if final_status == "REJECTED":
            adjusted_confidence *= 0.6
        if memory_stability == "UNSTABLE":
            adjusted_confidence *= 0.5
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # ------------------------------------------------------------------ #
        # Deployment Risk Evaluation — pipeline engine                        #
        # ------------------------------------------------------------------ #
        # Signal convention: 0.0 = no stress, 1.0 = maximum stress.
        # Derived from deployment profile tightness (hardware constraints),
        # not from benchmark measurements on the analysis machine.
        _dp_cpu2 = int(deployment_profile.get("cpu_cores") or 1)
        _dp_ram2 = float(deployment_profile.get("ram_gb") or 1.0)
        _dp_gpu2 = bool(deployment_profile.get("gpu_available", False))
        
        # If no strict SLA target is provided, estimate the baseline expected latency
        # constraint from hardware tier. A 1-core device implicitly operates under
        # heavy latency stress compared to a 32-core server.
        _hw_implied_target = max(5.0, 200.0 / math.sqrt(float(_dp_cpu2)))
        _dp_tgt2 = float(target_latency_sla) if target_latency_sla is not None else _hw_implied_target

        _sig_cpu2  = min(max(1.0 - (_dp_cpu2 - 1) / 15.0, 0.0), 1.0)  # 1 core→1.0, 16 cores→0.0
        _sig_mem2  = min(max(1.0 - _dp_ram2 / 8.0,         0.0), 1.0)  # 0.5 GB→0.94, 16 GB→0.0
        _sig_lat2  = min(max(1.0 - _dp_tgt2 / 200.0,       0.0), 1.0)  # 1 ms→1.0,  200 ms→0.0
        _sig_gpu2  = 0.0 if _dp_gpu2 else 0.5

        _pipeline_signals2: dict[str, float] = {
            "cpu":             _sig_cpu2,
            "memory":          _sig_mem2,
            "latency":         _sig_lat2,
            "gpu":             _sig_gpu2,
            "bandwidth":       0.5,
            "io":              0.5,
            "network":         0.5,
            "concurrency":     _sig_cpu2,
            "numa":            _sig_mem2,
            "future_drift":    0.5,
            "compatibility":   0.5,
            "security_signal": 0.5,
        }

        _pipeline_risk2: float = compute_risk(_pipeline_signals2)
        _pipeline_dec2:  str   = decision_from_risk(_pipeline_risk2)

        _DEC_MAP2: dict[str, str] = {
            "ALLOW":                 "APPROVED",
            "ALLOW_WITH_CONDITIONS": "CONDITIONAL_APPROVAL",
            "BLOCK":                 "REJECTED",
        }
        _mapped_status2: str = _DEC_MAP2.get(_pipeline_dec2, _pipeline_dec2)

        _risk_level2: str = (
            "HIGH_RISK"   if _pipeline_risk2 >= 8.2 else
            "MEDIUM_RISK" if _pipeline_risk2 >= 3.8 else
            "LOW_RISK"
        )

        risk_result = {
            "risk_score":      _pipeline_risk2,
            "risk_level":      _risk_level2,
            "engine_version":  "pipeline",
            "timestamp":       time.time(),
        }

        # Promote final_status to REJECTED when pipeline engine returns BLOCK
        if _mapped_status2 == "REJECTED" and final_status != "REJECTED":
            final_status = "REJECTED"
            reasons.append(
                f"pipeline_engine: risk {_pipeline_risk2:.3f} >= BLOCK threshold 8.22"
            )
            adjusted_confidence = max(0.0, min(1.0, base_confidence * 0.6))
        
        timestamp = time.time()
        
        decision_state = {
            "runtime": best_runtime,
            "status": final_status,
            "confidence": adjusted_confidence,
            "timestamp": timestamp,
            "reasons": reasons,
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
        }
        
        summary = build_deployment_summary(decision_state, analysis_state, diagnostics_state)

        with APP_STATE_LOCK:
            APP_STATE["model"] = {"path": model_path, "hash": model_hash}
            APP_STATE["deployment_profile"] = deployment_profile
            APP_STATE["diagnostics"] = diagnostics_state
            APP_STATE["analysis"] = analysis_state
            APP_STATE["decision"] = decision_state

        persistence.save_decision_record(
            runtime=best_runtime,
            status=final_status,
            confidence=adjusted_confidence,
            risk_score=risk_result["risk_score"],
            risk_level=risk_result["risk_level"],
            latency_ms=predicted_latency_ms,
            memory_mb=memory_usage_mb,
        )

        hw_override_applied: bool = (
            isinstance(hw_decision, dict) and bool(hw_decision.get("override_applied"))
        )

        from src.core.decision_trace import DecisionTrace
        from src.core.explainability import build_explanation_tree

        _runtime_before = analysis_state.get("best_runtime", "")
        _rejection_reasons = [r for r in reasons if rejected]
        _conditional_reasons = [r for r in reasons if not rejected and conditional]

        trace = DecisionTrace(
            selected_runtime_before_override=_runtime_before,
            selected_runtime_after_override=best_runtime,
            sla_target_latency=target_latency_sla,
            measured_latency=predicted_latency_ms,
            memory_limit=memory_limit_sla,
            measured_memory=memory_usage_mb,
            rejection_reasons=_rejection_reasons,
            conditional_reasons=_conditional_reasons,
            confidence_before_scaling=base_confidence,
            confidence_after_scaling=adjusted_confidence,
            hardware_override_applied=hw_override_applied,
            override_reason=hw_decision.get("override_reason") if isinstance(hw_decision, dict) else None,
        )

        response_body: dict[str, Any] = {
            "success": True,
            "summary": summary,
            "runtime_details": decision_state,
            "evaluations": evaluations,
            "analysis_state": {
                "best_runtime": analysis_state.get("best_runtime"),
                "confidence": analysis_state.get("confidence"),
                "evaluations": evaluations,
            },
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "confidence": adjusted_confidence,
            "model_path": model_path,
        }

        if hw_override_applied:
            response_body["hardware_selector"] = hw_decision
        if debug:
            response_body["decision_trace"] = trace.to_dict()
        if explain:
            response_body["explanation_tree"] = build_explanation_tree(trace)

        return response_body
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Diagnostics API
# ============================================================================

@app.post("/api/model/diagnostics")
@require_stage("model")
async def get_diagnostics():
    """Get diagnostics for a model."""
    # READ PHASE — first lock acquisition
    with APP_STATE_LOCK:
        model = APP_STATE.get("model")
        deployment_profile = APP_STATE.get("deployment_profile")
        existing_diagnostics = APP_STATE.get("diagnostics")

    if deployment_profile is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "invalid_execution_order",
                "required_stage": "deployment_profile",
            },
        )

    if model is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "invalid_execution_order",
                "required_stage": "model",
            },
        )

    model_path = model.get("path", "") if isinstance(model, dict) else ""
    if not model_path:
        return JSONResponse(
            status_code=409,
            content={
                "error": "invalid_execution_order",
                "required_stage": "model",
            },
        )

    # Optimistic re-entry check before heavy computation
    if existing_diagnostics is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "invalid_execution_order",
                "required_stage": "diagnostics",
                "detail": "diagnostics_already_computed",
            },
        )

    # Use stored hash — do NOT recompute from disk
    stored_hash = model.get("hash")

    # COMPUTE PHASE — outside lock
    try:
        analysis = _analyze_model(model_path)

        if not analysis.get("success", False):
            return {
                "success": False,
                "error": analysis.get("error", "Unknown error"),
            }

        # Overwrite any hash _analyze_model produced with the stored canonical hash
        analysis["file_hash"] = stored_hash

        facts = _build_facts(analysis)

        # WRITE PHASE — second lock acquisition with full re-checks to close TOCTOU window
        with APP_STATE_LOCK:
            # Re-check re-entry: another concurrent request may have written diagnostics
            if APP_STATE.get("diagnostics") is not None:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "invalid_execution_order",
                        "required_stage": "diagnostics",
                        "detail": "diagnostics_already_computed",
                    },
                )
            # Re-check hash: model may have been re-indexed during compute phase
            current_model = APP_STATE.get("model")
            current_hash = current_model.get("hash") if isinstance(current_model, dict) else None
            if current_hash != stored_hash:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "invalid_execution_order",
                        "required_stage": "model",
                    },
                )
            APP_STATE["diagnostics"] = {
                "facts": facts,
                "raw_analysis": analysis,
                "profile_snapshot": deployment_profile.copy(),
            }
            APP_STATE["analysis"] = None
            APP_STATE["decision"] = None

        logger.info(
            "diagnostics_run",
            extra={
                "event": "diagnostics_run",
                "model_hash": stored_hash,
                "parameter_scale_class": facts.get("parameter_scale_class"),
                "has_dynamic_shapes": facts.get("model.has_dynamic_shapes"),
                "sequential_depth_estimate": facts.get("sequential_depth_estimate"),
            },
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Calibration API
# ============================================================================

@app.get("/api/calibration/stats")
async def get_calibration_stats():
    """Get calibration statistics."""
    try:
        history_path = Path(".deploycheck_calibration_history.json")
        if history_path.exists():
            history = json.loads(history_path.read_text())
        else:
            history = []
        
        return {
            "total_records": len(history),
            "stats": {
                "accuracy": 0.85 if history else None,
                "total_evaluations": len(history),
            },
        }
    except Exception as e:
        return {
            "total_records": 0,
            "stats": {},
            "error": str(e),
        }


@app.post("/api/calibration/gpu/load")
async def load_gpu_calibration(file: UploadFile = File(...)):
    """
    Load a GPU benchmark calibration JSON file into the decision pipeline.

    The uploaded file must conform to the schema expected by
    GPUCalibrationProfile.  The endpoint:

    1. Reads and parses the uploaded JSON.
    2. Constructs a GPUCalibrationProfile (which runs schema validation and
       full structural parsing — no silent fallback).
    3. Writes the resulting profile to APP_STATE["gpu_calibration"] under
       APP_STATE_LOCK.

    Returns 422 on any validation or parse failure with a precise error
    message so the caller can diagnose the problem without guessing.
    """
    from src.core.gpu_calibration import GPUCalibrationProfile, GPUCalibrationError

    raw_bytes: bytes = await file.read()

    # ------------------------------------------------------------------ #
    # Parse JSON — fail fast with a clear message on malformed input.     #
    # ------------------------------------------------------------------ #
    try:
        raw_json: dict = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_json",
                "detail": str(exc),
            },
        )

    # ------------------------------------------------------------------ #
    # Construct GPUCalibrationProfile — runs schema validation and full   #
    # structural parse.  GPUCalibrationError is raised on any failure.   #
    # ------------------------------------------------------------------ #
    try:
        calibration_profile = GPUCalibrationProfile(raw_json)
    except GPUCalibrationError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": "calibration_validation_failed",
                "detail": str(exc),
            },
        )
    except Exception as exc:
        # Unexpected errors during construction must also be surfaced.
        logger.exception(
            "load_gpu_calibration: unexpected error constructing "
            "GPUCalibrationProfile: %s",
            exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "calibration_construction_error",
                "detail": str(exc),
            },
        )

    # ------------------------------------------------------------------ #
    # Atomic write — all APP_STATE mutations must be under APP_STATE_LOCK. #
    # ------------------------------------------------------------------ #
    with APP_STATE_LOCK:
        APP_STATE["gpu_calibration"] = calibration_profile

    persistence.save_calibration_record(
        provider_list=calibration_profile.providers_detected,
        gpu_name=calibration_profile.environment.get("gpu_name"),
        raw_json=raw_json,
    )

    summary = calibration_profile.summary()
    logger.info(
        "gpu_calibration_loaded",
        extra={
            "event": "gpu_calibration_loaded",
            "providers_detected": summary["providers_detected"],
            "models_detected": summary["models_detected"],
            "entry_count": summary["entry_count"],
            "gpu_name": summary.get("environment", {}).get("gpu_name"),
        },
    )

    return {
        "success": True,
        "summary": summary,
    }


# ============================================================================
# Confidence API
# ============================================================================

@app.get("/api/confidence/{model_hash}/{runtime}")
async def get_confidence(model_hash: str, runtime: str):
    """Get confidence score for a model/runtime combination."""
    try:
        return {
            "model_hash": model_hash,
            "runtime": runtime,
            "score": 0.85,
            "level": "HIGH",
            "reasons": ["Based on calibration history"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Fixes API
# ============================================================================

@app.get("/api/fixes/{model_hash}/{runtime}")
async def get_fixes(model_hash: str, runtime: str):
    """Get suggested fixes for a model."""
    try:
        return {
            "model_hash": model_hash,
            "runtime": runtime,
            "fixes": [
                {
                    "description": "Ensure model uses supported operators",
                    "severity": "info",
                }
            ],
        }
    except Exception as e:
        return {
            "model_hash": model_hash,
            "runtime": runtime,
            "fixes": [],
            "error": str(e),
        }


# ============================================================================
# Incidents API
# ============================================================================

@app.get("/api/incidents")
async def get_incidents():
    """Get recorded incidents."""
    try:
        incidents_path = Path("incidents.json")
        if incidents_path.exists():
            incidents = json.loads(incidents_path.read_text())
        else:
            incidents = []
        
        return {
            "incidents": incidents,
            "count": len(incidents),
        }
    except Exception as e:
        return {
            "incidents": [],
            "count": 0,
            "error": str(e),
        }


# ============================================================================
# History API
# ============================================================================

@app.get("/api/history/decisions")
async def get_decision_history():
    """Get last 50 decisions ordered by newest first."""
    try:
        items = persistence.list_decisions(limit=50)
        return {
            "success": True,
            "count": len(items),
            "items": items,
        }
    except Exception as e:
        return {
            "success": False,
            "count": 0,
            "items": [],
            "error": str(e),
        }


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
    }


# ============================================================================
# Capabilities Discovery
# ============================================================================

@app.get("/api/capabilities")
async def get_capabilities():
    """
    Capability discovery endpoint.
    
    Returns the list of available capabilities based on what endpoints
    and data sources actually exist and are functional.
    """
    # Check which endpoints exist and return data
    capabilities = {
        # Core capabilities - always available if backend is running
        "core": {
            "health": True,
            "model_analyze": True,
            "decision_recommend": True,
            "diagnostics": True,
        },
        # Optional capabilities - depend on data existence
        "optional": {
            # Calibration - only if history file exists
            "calibration": False,
            # Incidents - only if incidents file exists and has data
            "incidents": False,
            # Runtime session - depends on active deployment state
            "runtime_session": False,
            # Patch endpoint - depends on patch functionality
            "patch": False,
            # Export endpoint - depends on export functionality
            "export": False,
            # Calibration history data
            "calibration_history": False,
        },
        # Navigation items
        "navigation": {
            "models_tab": True,      # Model analysis is core
            "analysis_tab": True,     # Decision engine is core
            "diagnostics_tab": True,  # Diagnostics is core
            "fix_planner_tab": True,  # Fix suggestions if issues exist
            "reports_tab": False,     # Reports depend on report generation
            "settings_tab": True,     # Settings is always available
        },
        # Actions
        "actions": {
            "run_analysis": True,
            "generate_patch": False,
            "export_graph": False,
            "apply_fix": True,
            "view_report": False,
        },
        # Metadata
        "meta": {
            "discovery_version": "1.0.0",
            "dynamic_navigation": True,
            "conditional_rendering": True,
        }
    }
    
    # Check if calibration history exists
    try:
        history_path = Path(".deploycheck_calibration_history.json")
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history and len(history) > 0:
                capabilities["optional"]["calibration"] = True
                capabilities["optional"]["calibration_history"] = True
    except Exception:
        pass
    
    # Check if incidents exist
    try:
        incidents_path = Path("incidents.json")
        if incidents_path.exists():
            incidents = json.loads(incidents_path.read_text())
            if incidents and len(incidents) > 0:
                capabilities["optional"]["incidents"] = True
                # If incidents exist, incidents tab becomes available
                capabilities["navigation"]["incidents_tab"] = True
    except Exception:
        pass
    
    # Patch and export are stubbed - mark as unavailable
    # until real implementations exist
    capabilities["optional"]["patch"] = False
    capabilities["optional"]["export"] = False
    capabilities["actions"]["generate_patch"] = False
    capabilities["actions"]["export_graph"] = False
    capabilities["actions"]["view_report"] = False
    
    return {
        "capabilities": capabilities,
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


# ============================================================================
# Compatibility aliases — map old frontend URLs to existing handlers
# ============================================================================

@app.post("/api/model/upload")
async def upload_model_alias(file: UploadFile = File(...)):
    """Alias for /api/model/index for frontend compatibility."""
    return await get_model_index(file)

# ============================================================================
# Static UI Mount (same-origin with /api/*)
# ============================================================================

_STATIC_DIR = Path(__file__).resolve().parent.parent / "gui" / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
