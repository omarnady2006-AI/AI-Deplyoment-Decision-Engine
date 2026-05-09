from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from tflite_runtime import interpreter as _tflite_lib
    _Interpreter  = _tflite_lib.Interpreter
    _load_delegate = _tflite_lib.load_delegate
except ImportError:
    try:
        from tensorflow.lite.python import interpreter as _tflite_lib
        _Interpreter  = _tflite_lib.Interpreter
        _load_delegate = _tflite_lib.load_delegate
    except ImportError:
        _Interpreter  = None
        _load_delegate = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    trt  = None
    cuda = None

try:
    import psutil
except ImportError:
    psutil = None


# ── Constants ─────────────────────────────────────────────────────────────────

_WARMUP_RUNS    = 3
_BENCHMARK_RUNS = 10
_WALL_CAP_S     = 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rss() -> int:
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            pass
    return 0


def _resolve_shape(raw_shape: Any, ndim_fallback: int = 4) -> list[int]:
    if not raw_shape:
        return [1, 3, 224, 224]
    spatial = 224 if len(raw_shape) == 4 else 128
    out: list[int] = []
    for i, d in enumerate(raw_shape):
        if i == 0:
            out.append(1)
        elif isinstance(d, int) and d > 0:
            out.append(d)
        else:
            out.append(spatial)
    return out


def _ort_dtype(type_str: str):
    if np is None:
        return None
    _MAP = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)":  np.float64,
        "tensor(int64)":   np.int64,
        "tensor(int32)":   np.int32,
        "tensor(int8)":    np.int8,
        "tensor(uint8)":   np.uint8,
        "tensor(bool)":    np.bool_,
    }
    return _MAP.get(type_str, np.float32)


def _trt_volume(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= max(1, d)
    return n


# ── Base interface ────────────────────────────────────────────────────────────

class BaseRuntime(ABC):
    def __init__(self) -> None:
        self._model_path: str | None = None

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def run(self, input: Any) -> Any: ...

    @abstractmethod
    def benchmark(self, path: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def check_available(self) -> None:
        raise NotImplementedError


# ── ONNX_CPU ──────────────────────────────────────────────────────────────────

class ONNX_CPU(BaseRuntime):
    _PROVIDER = "CPUExecutionProvider"

    def check_available(self) -> None:
        import onnxruntime  # noqa: F401

    def load(self, path: str) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(path, sess_options=opts,
                                             providers=[self._PROVIDER])
        self._model_path = path

    def run(self, input: Any) -> Any:
        inp_name = self._session.get_inputs()[0].name
        return self._session.run(None, {inp_name: input})

    def _dummy_inputs(self) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        feed: dict[str, Any] = {}
        for inp in self._session.get_inputs():
            shape = _resolve_shape(getattr(inp, "shape", None) or [])
            dtype = _ort_dtype(getattr(inp, "type", "tensor(float)"))
            if dtype is None:
                feed[inp.name] = np.array([""] * int(np.prod(shape)), dtype=object).reshape(shape)
            elif np.issubdtype(dtype, np.floating):
                feed[inp.name] = rng.random(shape).astype(dtype)
            else:
                feed[inp.name] = np.zeros(shape, dtype=dtype)
        return feed

    def benchmark(self, path: str | None = None) -> dict[str, Any]:
        if path:
            self.load(path)
        if ort is None or np is None:
            return {"success": False, "error": "onnxruntime/numpy unavailable",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        if self._PROVIDER not in ort.get_available_providers():
            return {"success": False, "error": f"{self._PROVIDER} unavailable",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        try:
            dummy = self._dummy_inputs()
            for _ in range(_WARMUP_RUNS):
                self._session.run(None, dummy)
            rss_before = _rss()
            timings: list[float] = []
            wall = time.perf_counter()
            for _ in range(_BENCHMARK_RUNS):
                t0 = time.perf_counter()
                self._session.run(None, dummy)
                timings.append((time.perf_counter() - t0) * 1000.0)
                if (time.perf_counter() - wall) >= _WALL_CAP_S:
                    break
            memory_mb = float(max(0, _rss() - rss_before)) / (1024.0 * 1024.0)
            avg = float(sum(timings) / len(timings))
            p95 = float(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)])
            return {"success": True, "latency_avg_ms": avg, "latency_p95_ms": p95,
                    "memory_mb": memory_mb, "benchmark_runs": len(timings), "error": None}
        except Exception as exc:
            return {"success": False, "error": str(exc),
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}


# ── ONNX_CUDA ─────────────────────────────────────────────────────────────────

class ONNX_CUDA(ONNX_CPU):
    _PROVIDER = "CUDAExecutionProvider"

    def check_available(self) -> None:
        import onnxruntime  # noqa: F401

    def load(self, path: str) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        self._session = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._model_path = path


# ── TENSORRT ──────────────────────────────────────────────────────────────────

class TENSORRT(BaseRuntime):
    def check_available(self) -> None:
        import tensorrt  # noqa: F401

    def load(self, path: str) -> None:
        if trt is None or cuda is None:
            raise RuntimeError("tensorrt/pycuda not installed")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            engine_data = f.read()
        self._engine  = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()
        self._model_path = path

    def run(self, input: Any) -> Any:
        if trt is None or cuda is None:
            raise RuntimeError("tensorrt/pycuda not installed")
        n = self._engine.num_bindings
        host_bufs: list[Any] = []
        dev_bufs:  list[Any] = []
        bindings:  list[int] = []
        for i in range(n):
            shape = _resolve_shape(list(self._engine.get_binding_shape(i)))
            h = np.zeros(_trt_volume(shape), dtype=np.float32)
            if self._engine.binding_is_input(i) and input is not None:
                h = np.asarray(input, dtype=np.float32).ravel()
            host_bufs.append(h)
            d = cuda.mem_alloc(h.nbytes)
            dev_bufs.append(d)
            bindings.append(int(d))
        stream = cuda.Stream()
        for i in range(n):
            if self._engine.binding_is_input(i):
                cuda.memcpy_htod_async(dev_bufs[i], host_bufs[i], stream)
        self._context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        outputs = []
        for i in range(n):
            if not self._engine.binding_is_input(i):
                shape = _resolve_shape(list(self._engine.get_binding_shape(i)))
                out = np.empty(_trt_volume(shape), dtype=np.float32)
                cuda.memcpy_dtoh(out, dev_bufs[i])
                outputs.append(out)
        for d in dev_bufs:
            d.free()
        return outputs

    def benchmark(self, path: str | None = None) -> dict[str, Any]:
        if path:
            self.load(path)
        if trt is None or cuda is None or np is None:
            return {"success": False, "error": "tensorrt/pycuda/numpy unavailable",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        dev_bufs: list[Any] = []
        try:
            n = self._engine.num_bindings
            host_bufs: list[Any] = []
            bindings:  list[int] = []
            for i in range(n):
                shape = _resolve_shape(list(self._engine.get_binding_shape(i)))
                h = np.zeros(_trt_volume(shape), dtype=np.float32)
                host_bufs.append(h)
                d = cuda.mem_alloc(h.nbytes)
                dev_bufs.append(d)
                bindings.append(int(d))
            stream = cuda.Stream()

            def _run() -> None:
                for i in range(n):
                    if self._engine.binding_is_input(i):
                        cuda.memcpy_htod_async(dev_bufs[i], host_bufs[i], stream)
                self._context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()

            for _ in range(_WARMUP_RUNS):
                _run()
            rss_before = _rss()
            timings: list[float] = []
            wall = time.perf_counter()
            for _ in range(_BENCHMARK_RUNS):
                t0 = time.perf_counter()
                _run()
                timings.append((time.perf_counter() - t0) * 1000.0)
                if (time.perf_counter() - wall) >= _WALL_CAP_S:
                    break
            memory_mb = float(max(0, _rss() - rss_before)) / (1024.0 * 1024.0)
            avg = float(sum(timings) / len(timings))
            p95 = float(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)])
            return {"success": True, "latency_avg_ms": avg, "latency_p95_ms": p95,
                    "memory_mb": memory_mb, "benchmark_runs": len(timings), "error": None}
        except Exception as exc:
            return {"success": False, "error": str(exc),
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        finally:
            for d in dev_bufs:
                try:
                    d.free()
                except Exception:
                    pass


# ── TORCH ─────────────────────────────────────────────────────────────────────

class TORCH(BaseRuntime):
    def check_available(self) -> None:
        import torch  # noqa: F401

    def load(self, path: str) -> None:
        import os

        if torch is None:
            raise RuntimeError("torch not installed")

        # Phase 2 — security gate: weights_only=False allows arbitrary code
        # execution via pickle.  Caller must opt in explicitly.
        ALLOW_UNSAFE = os.getenv("ALLOW_UNSAFE_TORCH_LOAD", "0") == "1"
        if not ALLOW_UNSAFE:
            raise RuntimeError(
                "Unsafe torch.load blocked. "
                "Set ALLOW_UNSAFE_TORCH_LOAD=1 to enable loading full models."
            )

        torch.set_num_threads(1)
        torch.manual_seed(0)
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            pass

        # Phase 1 — PyTorch >=2.6 compat: weights_only defaults to True in 2.6+,
        # breaking legacy .pt/.tar files that contain full serialised modules.
        obj = torch.load(path, weights_only=False, map_location="cpu")

        if isinstance(obj, torch.nn.Module):
            self._model = obj
        elif isinstance(obj, dict):
            raise RuntimeError(
                "State dict detected. Full torch.nn.Module required."
            )
        else:
            raise RuntimeError(f"Unsupported torch object type: {type(obj)}")

        if not hasattr(self._model, "forward"):
            raise RuntimeError("Loaded torch object is not a valid model")

        self._model.eval()
        self._model_path = path

    def run(self, input: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not installed")
        tensor = torch.tensor(input, dtype=torch.float32)
        with torch.no_grad():
            out = self._model(tensor)
        return out.detach().cpu().numpy() if hasattr(out, "detach") else out

    def benchmark(self, path: str | None = None) -> dict[str, Any]:
        if path:
            self.load(path)
        if torch is None or np is None:
            return {"success": False, "error": "torch/numpy unavailable",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        if self._model is None:
            return {"success": False, "error": "Model not loaded",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        try:
            dummy = torch.randn(1, 3, 224, 224)
            for _ in range(_WARMUP_RUNS):
                with torch.no_grad():
                    self._model(dummy)
            rss_before = _rss()
            timings: list[float] = []
            wall = time.perf_counter()
            for _ in range(_BENCHMARK_RUNS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    self._model(dummy)
                timings.append((time.perf_counter() - t0) * 1000.0)
                if (time.perf_counter() - wall) >= _WALL_CAP_S:
                    break
            memory_mb = float(max(0, _rss() - rss_before)) / (1024.0 * 1024.0)
            avg = float(sum(timings) / len(timings))
            p95 = float(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)])
            return {"success": True, "latency_avg_ms": avg, "latency_p95_ms": p95,
                    "memory_mb": memory_mb, "benchmark_runs": len(timings), "error": None}
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb": None,
            }


# ── TFLITE ────────────────────────────────────────────────────────────────────

class TFLITE(BaseRuntime):
    def __init__(self, use_gpu_delegate: bool = False) -> None:
        super().__init__()
        self._use_gpu = use_gpu_delegate

    def check_available(self) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter  # noqa: F401
        except ImportError:
            import tensorflow as tf  # noqa: F401

    def load(self, path: str) -> None:
        if _Interpreter is None:
            raise RuntimeError("tflite_runtime or tensorflow not installed")
        delegates = []
        if self._use_gpu and _load_delegate is not None:
            try:
                delegates = [_load_delegate("libGLESv2.so")]
            except Exception:
                pass
        self._interp = _Interpreter(model_path=path,
                                    experimental_delegates=delegates or None)
        self._interp.allocate_tensors()
        self._model_path = path

    def run(self, input: Any) -> Any:
        details = self._interp.get_input_details()
        self._interp.set_tensor(details[0]["index"],
                                np.array(input, dtype=np.float32))
        self._interp.invoke()
        return [self._interp.get_tensor(d["index"])
                for d in self._interp.get_output_details()]

    def _build_dummy(self) -> dict[int, Any]:
        rng = np.random.default_rng(0)
        dummy: dict[int, Any] = {}
        for d in self._interp.get_input_details():
            shape = _resolve_shape(list(d.get("shape", [])))
            dtype = d.get("dtype", np.float32)
            if np.issubdtype(dtype, np.floating):
                dummy[int(d["index"])] = rng.random(shape).astype(dtype)
            else:
                dummy[int(d["index"])] = np.zeros(shape, dtype=dtype)
        return dummy

    def benchmark(self, path: str | None = None) -> dict[str, Any]:
        if path:
            self.load(path)
        if _Interpreter is None or np is None:
            return {"success": False, "error": "tflite/numpy unavailable",
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
        try:
            dummy = self._build_dummy()
            for _ in range(_WARMUP_RUNS):
                for idx, arr in dummy.items():
                    self._interp.set_tensor(idx, arr)
                self._interp.invoke()
            rss_before = _rss()
            timings: list[float] = []
            wall = time.perf_counter()
            for _ in range(_BENCHMARK_RUNS):
                for idx, arr in dummy.items():
                    self._interp.set_tensor(idx, arr)
                t0 = time.perf_counter()
                self._interp.invoke()
                timings.append((time.perf_counter() - t0) * 1000.0)
                if (time.perf_counter() - wall) >= _WALL_CAP_S:
                    break
            memory_mb = float(max(0, _rss() - rss_before)) / (1024.0 * 1024.0)
            avg = float(sum(timings) / len(timings))
            p95 = float(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)])
            return {"success": True, "latency_avg_ms": avg, "latency_p95_ms": p95,
                    "memory_mb": memory_mb, "benchmark_runs": len(timings), "error": None}
        except Exception as exc:
            return {"success": False, "error": str(exc),
                    "latency_avg_ms": None, "latency_p95_ms": None, "memory_mb": None}
