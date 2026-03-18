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
Optional Dependencies Module

Centralizes all optional dependency imports to avoid scattered try/except blocks.
Each dependency is checked once at import time and cached.

Usage:
    from src.core.optional_dependencies import TORCH
    if TORCH.available:
        tensor = TORCH.module.randn(3, 3)
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OptionalDependency:
    """Represents an optional dependency that may or may not be available."""
    name: str
    import_path: str
    available: bool
    module: Any
    error_message: Optional[str] = None
    
    def require(self) -> Any:
        """Return the module if available, otherwise raise ImportError."""
        if not self.available:
            raise ImportError(
                f"Optional dependency '{self.name}' is not available. "
                f"Install it with: pip install {self.name}"
            )
        return self.module


def _try_import(name: str, import_path: str) -> OptionalDependency:
    """Attempt to import an optional dependency."""
    try:
        module = __import__(import_path, fromlist=[''])
        return OptionalDependency(
            name=name,
            import_path=import_path,
            available=True,
            module=module,
        )
    except ImportError as e:
        return OptionalDependency(
            name=name,
            import_path=import_path,
            available=False,
            module=None,
            error_message=str(e),
        )


TORCH = _try_import("torch", "torch")
TENSORRT = _try_import("tensorrt", "tensorrt")
PYCUDA = _try_import("pycuda", "pycuda")
OPENVINO = _try_import("openvino", "openvino.runtime")
TFLITE = _try_import("tflite_runtime", "tflite_runtime.interpreter")
TENSORFLOW = _try_import("tensorflow", "tensorflow")
ONNXRUNTIME = _try_import("onnxruntime", "onnxruntime")
PYCUDA_DRIVER = _try_import("pycuda.driver", "pycuda.driver")
PYCUDA_AUTOINIT = _try_import("pycuda.autoinit", "pycuda.autoinit")
NUMPY = _try_import("numpy", "numpy")
ONNX = _try_import("onnx", "onnx")
PSUTIL = _try_import("psutil", "psutil")
SKLEARN = _try_import("sklearn", "sklearn")
SCIPY = _try_import("scipy", "scipy")
GPUTIL = _try_import("GPUtil", "GPUtil")
PYNVML = _try_import("pynvml", "pynvml")
PYROCM = _try_import("pyamdgpuinfo", "pyamdgpuinfo")


ALL_OPTIONAL = {
    "torch": TORCH,
    "tensorrt": TENSORRT,
    "pycuda": PYCUDA,
    "openvino": OPENVINO,
    "tflite_runtime": TFLITE,
    "tensorflow": TENSORFLOW,
    "onnxruntime": ONNXRUNTIME,
}


def get_available_backends() -> list[str]:
    """Return list of available inference backends."""
    backends = []
    if ONNXRUNTIME.available:
        backends.append("onnxruntime")
    if TENSORRT.available:
        backends.append("tensorrt")
    if OPENVINO.available:
        backends.append("openvino")
    if TFLITE.available:
        backends.append("tflite")
    return backends


def check_gpu_support() -> dict[str, bool]:
    """Check GPU-related dependency availability."""
    return {
        "cuda": TORCH.available and hasattr(TORCH.module, 'cuda') and TORCH.module.cuda.is_available(),
        "tensorrt": TENSORRT.available,
        "pycuda": PYCUDA.available,
    }
