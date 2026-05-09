"""
core/runtime/__init__.py

Re-exports RuntimeName so all existing `from src.core.runtime import RuntimeName`
imports continue to work after runtime.py was superseded by this package.

CANONICAL_RUNTIMES is the authoritative ordered tuple for pipeline iteration.
It excludes the three legacy aliases (TENSORRT, ONNXRUNTIME, TFLITE) that exist
only so runtime_validator.py and rules/ can reference them as dict keys / family-
set members without iterating them as independent pipeline entries.  All pipeline
loops (recommend_runtime, run_analysis_stage, run_full_pipeline) MUST iterate
CANONICAL_RUNTIMES, never `for runtime in RuntimeName`.
"""
from enum import Enum


class RuntimeName(str, Enum):
    # ── Legacy / internal names (kept for backward compatibility) ──────────────
    # These exist ONLY as symbolic references in runtime_validator.py family sets
    # and rules/ operator tables.  They MUST NOT appear in CANONICAL_RUNTIMES and
    # MUST NOT be iterated by the pipeline (see BUG-03 fix notes).
    TENSORRT    = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    TFLITE      = "tflite"

    # ── ONNX execution providers ───────────────────────────────────────────────
    ONNX_CPU  = "ONNX_CPU"
    ONNX_CUDA = "ONNX_CUDA"
    TRT       = "TensorRT"          # ONNX-based TensorRT execution provider

    # ── TFLite runtimes ────────────────────────────────────────────────────────
    TFLITE_CPU = "TFLite_CPU"
    TFLITE_GPU = "TFLite_GPU"

    # ── TensorFlow (SavedModel / frozen-graph) runtimes ───────────────────────
    TF_CPU = "TF_CPU"
    TF_GPU = "TF_GPU"

    # ── Native TensorRT serialised-engine runtime ─────────────────────────────
    TRT_NATIVE = "TensorRT_Native"

    # ── OpenVINO IR runtimes ───────────────────────────────────────────────────
    OV_CPU = "OpenVINO_CPU"
    OV_GPU = "OpenVINO_GPU"


# ── Canonical iteration set ───────────────────────────────────────────────────
# Ordered tuple of the 10 real pipeline runtimes.  Use this in every `for`
# loop that feeds validate_runtime / recommend_runtime.  Never iterate
# RuntimeName directly — that includes the 3 legacy alias members.
CANONICAL_RUNTIMES: tuple["RuntimeName", ...] = (
    RuntimeName.ONNX_CPU,
    RuntimeName.ONNX_CUDA,
    RuntimeName.TRT,
    RuntimeName.TFLITE_CPU,
    RuntimeName.TFLITE_GPU,
    RuntimeName.TF_CPU,
    RuntimeName.TF_GPU,
    RuntimeName.TRT_NATIVE,
    RuntimeName.OV_CPU,
    RuntimeName.OV_GPU,
)
