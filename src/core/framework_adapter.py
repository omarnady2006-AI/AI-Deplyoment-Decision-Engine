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
framework_adapter.py — Multi-Framework Inference Adapter
=========================================================
Detects model format by file extension and dispatches to the correct
inference runtime.  All runners return the same two-tuple:

    (latency_ms: float, memory_mb: float)

where:
    latency_ms  — wall-clock time for a single inference call, in ms
    memory_mb   — RSS delta across the call, in MB (clamped to 0.0 minimum)

Framework libraries are imported lazily so that missing runtimes do not
prevent the engine from running models for the frameworks that *are*
installed.
"""


import time

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

def detect_framework(model_path: str) -> str:
    """Return the framework name for *model_path* based on its extension.

    Raises ``ValueError`` for unsupported formats.
    """
    path = model_path.lower()

    if path.endswith(".onnx"):
        return "onnx"

    if path.endswith(".pt") or path.endswith(".pth"):
        return "pytorch"

    if path.endswith(".tflite"):
        return "tflite"

    if path.endswith(".pb") or path.endswith(".savedmodel"):
        return "tensorflow"

    raise ValueError(f"Unsupported model format: {model_path}")


# ---------------------------------------------------------------------------
# Per-framework runners
# ---------------------------------------------------------------------------

def run_onnx(model_path: str) -> tuple[float, float]:
    """Run one ONNX inference and return (latency_ms, memory_mb)."""
    if ort is None:
        raise RuntimeError("ONNX Runtime not available")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inputs = session.get_inputs()
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inputs[0].shape]

    if np is not None:
        dummy = np.random.rand(*shape).astype(np.float32)
    else:
        import array as _array
        n = 1
        for d in shape:
            n *= d
        dummy_raw = _array.array('f', [0.0] * n)
        dummy = dummy_raw  # ort accepts array.array for float32 inputs

    if psutil is not None:
        proc = psutil.Process()
        start_mem = proc.memory_info().rss
    else:
        start_mem = 0

    start = time.time()
    session.run(None, {inputs[0].name: dummy})
    latency = (time.time() - start) * 1000.0

    if psutil is not None:
        end_mem = proc.memory_info().rss  # type: ignore[possibly-undefined]
        memory_mb = max(0.0, (end_mem - start_mem) / (1024.0 * 1024.0))
    else:
        memory_mb = 0.0

    return latency, memory_mb


def run_pytorch(model_path: str) -> tuple[float, float]:
    """Run one PyTorch inference and return (latency_ms, memory_mb)."""
    if torch is None:
        raise RuntimeError("PyTorch not available")

    model = torch.load(model_path, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    if psutil is not None:
        proc = psutil.Process()
        start_mem = proc.memory_info().rss
    else:
        start_mem = 0

    start = time.time()

    with torch.no_grad():
        model(dummy)

    latency = (time.time() - start) * 1000.0

    if psutil is not None:
        end_mem = proc.memory_info().rss  # type: ignore[possibly-undefined]
        memory_mb = max(0.0, (end_mem - start_mem) / (1024.0 * 1024.0))
    else:
        memory_mb = 0.0

    return latency, memory_mb


def run_tensorflow(model_path: str) -> tuple[float, float]:
    """Run one TensorFlow SavedModel inference and return (latency_ms, memory_mb)."""
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    model = tf.saved_model.load(model_path)

    dummy = tf.random.normal([1, 224, 224, 3])

    if psutil is not None:
        proc = psutil.Process()
        start_mem = proc.memory_info().rss
    else:
        start_mem = 0

    start = time.time()

    model(dummy)

    latency = (time.time() - start) * 1000.0

    if psutil is not None:
        end_mem = proc.memory_info().rss  # type: ignore[possibly-undefined]
        memory_mb = max(0.0, (end_mem - start_mem) / (1024.0 * 1024.0))
    else:
        memory_mb = 0.0

    return latency, memory_mb


def run_tflite(model_path: str) -> tuple[float, float]:
    """Run one TFLite inference and return (latency_ms, memory_mb)."""
    if tf is None:
        raise RuntimeError("TensorFlow (TFLite) not available")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    shape = input_details[0]["shape"]

    if np is not None:
        dummy = np.random.rand(*shape).astype(np.float32)
    else:
        import array as _array
        n = 1
        for d in shape:
            n *= d
        dummy = _array.array('f', [0.0] * n)

    interpreter.set_tensor(input_details[0]["index"], dummy)

    if psutil is not None:
        proc = psutil.Process()
        start_mem = proc.memory_info().rss
    else:
        start_mem = 0

    start = time.time()

    interpreter.invoke()

    latency = (time.time() - start) * 1000.0

    if psutil is not None:
        end_mem = proc.memory_info().rss  # type: ignore[possibly-undefined]
        memory_mb = max(0.0, (end_mem - start_mem) / (1024.0 * 1024.0))
    else:
        memory_mb = 0.0

    return latency, memory_mb


# ---------------------------------------------------------------------------
# Unified entrypoint
# ---------------------------------------------------------------------------

def run_inference(model_path: str) -> tuple[float, float]:
    """Detect the framework for *model_path* and run one inference pass.

    Returns ``(latency_ms, memory_mb)`` in the same format for all frameworks
    so callers need no framework-specific logic.

    Raises ``ValueError`` for unsupported formats.
    Propagates any runtime errors from the underlying framework.
    """
    framework = detect_framework(model_path)

    if framework == "onnx":
        return run_onnx(model_path)

    if framework == "pytorch":
        return run_pytorch(model_path)

    if framework == "tensorflow":
        return run_tensorflow(model_path)

    if framework == "tflite":
        return run_tflite(model_path)

    # detect_framework already raises for unknown formats; this is a safety net.
    raise ValueError(f"No runner registered for framework: {framework}")
