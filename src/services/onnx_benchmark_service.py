"""
services/onnx_benchmark_service.py

Orchestrates runtime evaluation for all supported model formats.

Dispatch table maps file extension → list of runtime specs.  Each spec
describes one benchmark run: which evaluator function to call and what
arguments to pass.  The ONNX / PyTorch paths preserve the existing
PyTorch → ONNX conversion behaviour (handled by evaluate_onnx_runtime).

ZERO business logic — all computation delegated to core modules.

CONTRACT: Returns list[RuntimeEvaluation] — typed, frozen, no raw dicts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.runtime.evaluator import evaluate_onnx_runtime
from src.core.runtime.tf_evaluator import evaluate_tf_runtime
from src.core.runtime.tflite_evaluator import evaluate_tflite_runtime
from src.core.runtime.tensorrt_native_evaluator import evaluate_tensorrt_native_runtime
from src.core.runtime.openvino_evaluator import evaluate_openvino_runtime
from src.core.runtime.execution_loop import execute_model_runtimes, to_evaluator_dict
from src.core.runtime.registry import get_runtimes_for_model
from src.core.runtime.postprocess import postprocess_runtime_result, query_available_providers
from src.core.contracts.runtime_evaluation import RuntimeEvaluation
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Runtime spec dispatch table
# ---------------------------------------------------------------------------
# Each entry in the lists below is a dict that fully describes one benchmark:
#   name             – RuntimeEvaluation.runtime string (matches RuntimeName enum)
#   precision        – numeric precision label surfaced to callers
#   kind             – routing key: "onnx"|"tflite"|"tf"|"trt_native"|"openvino"
#   provider         – (onnx only) onnxruntime ExecutionProvider string
#   use_gpu_delegate – (tflite only) whether to load the GPU delegate
#   use_gpu          – (tf only) whether to allow GPU execution
#   device_name      – (openvino only) "CPU" or "GPU"
#   is_cpu           – True iff this runtime runs on the host CPU (drives hw scaling)
#   gpu_flags        – deployment-profile bool keys that must ALL be True for this
#                      runtime to be attempted (empty = no GPU requirement)

_FORMAT_RUNTIME_SPECS: dict[str, list[dict[str, Any]]] = {
    # ── ONNX: evaluated via onnxruntime execution providers ──────────────────
    "onnx": [
        {
            "name": "ONNX_CPU", "precision": "FP32",
            "kind": "onnx", "provider": "CPUExecutionProvider",
            "is_cpu": True, "gpu_flags": [],
        },
        {
            "name": "ONNX_CUDA", "precision": "FP16/FP32",
            "kind": "onnx", "provider": "CUDAExecutionProvider",
            "is_cpu": False, "gpu_flags": ["gpu_available", "cuda_available"],
        },
        {
            "name": "TensorRT", "precision": "FP16/FP32",
            "kind": "onnx", "provider": "TensorrtExecutionProvider",
            "is_cpu": False,
            "gpu_flags": ["gpu_available", "cuda_available", "trt_available"],
        },
    ],

    # ── PyTorch: convert to ONNX first, then evaluate via onnxruntime ────────
    "pytorch": [
        {
            "name": "ONNX_CPU", "precision": "FP32",
            "kind": "onnx", "provider": "CPUExecutionProvider",
            "is_cpu": True, "gpu_flags": [],
        },
        {
            "name": "ONNX_CUDA", "precision": "FP16/FP32",
            "kind": "onnx", "provider": "CUDAExecutionProvider",
            "is_cpu": False, "gpu_flags": ["gpu_available", "cuda_available"],
        },
        {
            "name": "TensorRT", "precision": "FP16/FP32",
            "kind": "onnx", "provider": "TensorrtExecutionProvider",
            "is_cpu": False,
            "gpu_flags": ["gpu_available", "cuda_available", "trt_available"],
        },
    ],

    # ── TFLite: native .tflite interpreter (CPU + optional GPU delegate) ─────
    "tflite": [
        {
            "name": "TFLite_CPU", "precision": "FP32",
            "kind": "tflite", "use_gpu_delegate": False,
            "is_cpu": True, "gpu_flags": [],
        },
        {
            "name": "TFLite_GPU", "precision": "FP16/FP32",
            "kind": "tflite", "use_gpu_delegate": True,
            "is_cpu": False, "gpu_flags": ["gpu_available"],
        },
    ],

    # ── TensorFlow: SavedModel / frozen-graph benchmark ───────────────────────
    "tensorflow": [
        {
            "name": "TF_CPU", "precision": "FP32",
            "kind": "tf", "use_gpu": False,
            "is_cpu": True, "gpu_flags": [],
        },
        {
            "name": "TF_GPU", "precision": "FP16/FP32",
            "kind": "tf", "use_gpu": True,
            "is_cpu": False, "gpu_flags": ["gpu_available"],
        },
    ],

    # ── TensorRT: native serialised engine ───────────────────────────────────
    "tensorrt": [
        {
            "name": "TensorRT_Native", "precision": "FP16/FP32",
            "kind": "trt_native",
            "is_cpu": False,
            "gpu_flags": ["gpu_available", "cuda_available"],
        },
    ],

    # ── OpenVINO IR: CPU and GPU devices ─────────────────────────────────────
    "openvino": [
        {
            "name": "OpenVINO_CPU", "precision": "FP32",
            "kind": "openvino", "device_name": "CPU",
            "is_cpu": True, "gpu_flags": [],
        },
        {
            "name": "OpenVINO_GPU", "precision": "FP16/FP32",
            "kind": "openvino", "device_name": "GPU",
            "is_cpu": False, "gpu_flags": ["gpu_available"],
        },
    ],
}

# Extension → framework key for _FORMAT_RUNTIME_SPECS lookup
_EXT_TO_FRAMEWORK: dict[str, str] = {
    ".onnx":   "onnx",
    ".pt":     "pytorch",
    ".pth":    "pytorch",
    ".tflite": "tflite",
    ".pb":     "tensorflow",
    ".engine": "tensorrt",
    ".trt":    "tensorrt",
    ".xml":    "openvino",
    # NOTE: ".bin" is intentionally excluded — the extension is ambiguous
    # (TF frozen graph weights, GGUF, and other formats also use .bin).
    # OpenVINO models should be submitted via their .xml descriptor file;
    # the evaluator will resolve the companion .bin automatically.
}


# ---------------------------------------------------------------------------
# Evaluator dispatch
# ---------------------------------------------------------------------------

def _call_evaluator(
    kind: str,
    model_path: str,
    spec: dict[str, Any],
    stress_runs: int,
    eval_map: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """
    Return a raw benchmark dict for one spec entry.

    Routing
    -------
    "onnx" / "tflite":
        Lookup in *eval_map* (built once by evaluate_all_runtimes from the
        single execute_model_runtimes() call).  Never calls execute_model_runtimes
        itself.  Registry key derived from spec["name"].upper() — no mapping dict.

    "tf" / "trt_native" / "openvino":
        Delegated to their dedicated evaluator functions (unchanged).

    Never raises.
    """
    if kind in ("onnx", "tflite"):
        # Phase 2: derive registry key from spec name — no hardcoded dict.
        # spec["name"] examples: "ONNX_CPU", "ONNX_CUDA", "TensorRT",
        #                        "TFLite_CPU", "TFLite_GPU"
        # registry keys:         "ONNX_CPU", "ONNX_CUDA", "TENSORRT", "TFLITE"
        registry_key = spec["name"].upper()
        if registry_key == "TENSORRT":
            registry_key = "TENSORRT"
        if registry_key.startswith("TFLITE"):
            registry_key = "TFLITE"

        # Phase 1: reuse pre-built eval_map — execution loop already ran once.
        match = (eval_map or {}).get(registry_key)
        if match is None:
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          f"No evaluation found for registry key '{registry_key}'",
                "benchmark_runs": 0,
                "stress_test": {
                    "enabled": False, "runs": 0,
                    "latency_avg_ms": None, "latency_p95_ms": None,
                    "peak_memory_mb": None, "memory_growth_mb": None,
                    "memory_stability": "STABLE",
                },
            }
        return to_evaluator_dict(match)

    # ── TensorFlow (SavedModel / frozen-graph) — unchanged ────────────────────
    if kind == "tf":
        return evaluate_tf_runtime(
            model_path,
            use_gpu=spec.get("use_gpu", False),
            stress_runs=stress_runs,
        )

    # ── Native TensorRT serialised engine — unchanged ─────────────────────────
    if kind == "trt_native":
        return evaluate_tensorrt_native_runtime(
            model_path,
            stress_runs=stress_runs,
        )

    # ── OpenVINO IR — unchanged ───────────────────────────────────────────────
    if kind == "openvino":
        return evaluate_openvino_runtime(
            model_path,
            device_name=spec.get("device_name", "CPU"),
            stress_runs=stress_runs,
        )

    # ── Unknown kind — safe failure payload ───────────────────────────────────
    return {
        "success":        False,
        "latency_avg_ms": None,
        "latency_p95_ms": None,
        "memory_mb":      None,
        "error":          f"unknown evaluator kind: {kind!r}",
        "benchmark_runs": 0,
        "stress_test": {
            "enabled": False, "runs": 0,
            "latency_avg_ms": None, "latency_p95_ms": None,
            "peak_memory_mb": None, "memory_growth_mb": None,
            "memory_stability": "STABLE",
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all_runtimes(
    model_path: str,
    deployment_profile: dict[str, Any],
    server_hw_profile: dict[str, Any],
) -> list[RuntimeEvaluation]:
    """
    Evaluate all runtimes applicable to *model_path*'s format.

    Orchestration only — no business logic:
        1. Resolve framework from file extension.
        2. Run execution loop once for registry-backed formats.
        3. Dispatch each spec to _call_evaluator.
        4. Delegate all gating, scaling, and classification to postprocess.

    Returns list[RuntimeEvaluation] — typed and frozen, one per runtime.
    Confidence and utility scores default to 0.0 — set by the scoring stage.
    """
    # ── Resolve framework from file extension ─────────────────────────────────
    suffix    = Path(model_path).suffix.lower()
    framework = _EXT_TO_FRAMEWORK.get(suffix)

    if framework is None or framework not in _FORMAT_RUNTIME_SPECS:
        logger.warning(
            "evaluate_all_runtimes_unknown_format",
            extra={
                "event":      "evaluate_all_runtimes_unknown_format",
                "model_path": model_path,
                "suffix":     suffix,
            },
        )
        return []

    specs          = _FORMAT_RUNTIME_SPECS[framework]
    stress_enabled = bool(deployment_profile.get("stress_test", False))

    # ── Run execution loop ONCE for registry-backed formats ───────────────────
    eval_map: dict[str, Any] = {}
    if framework in ("onnx", "pytorch", "tflite"):
        try:
            loop_result = execute_model_runtimes(model_path)
            eval_map = {e["runtime"]: e for e in loop_result["evaluations"]}
        except RuntimeError as exc:
            logger.error(
                "execute_model_runtimes_failed",
                extra={
                    "event":      "execute_model_runtimes_failed",
                    "model_path": model_path,
                    "reason":     str(exc),
                },
            )

    onnx_providers = query_available_providers()

    # ── Dispatch each spec ────────────────────────────────────────────────────
    results: list[RuntimeEvaluation] = []

    for spec in specs:
        kind  = spec["kind"]
        bench = _call_evaluator(kind, model_path, spec, 100 if stress_enabled else 0, eval_map)
        result = postprocess_runtime_result(
            spec, bench, deployment_profile, server_hw_profile, onnx_providers,
        )
        results.append(result)

    return results

