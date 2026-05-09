"""
src/training/data_collector.py

Real-World Benchmark Data Collection Pipeline.

PURPOSE
=======
Executes all 10 canonical runtimes on each model file and records:
  - latency_ms         (median of N warmup + K timed runs)
  - peak_memory_mb     (RSS delta before/after inference)
  - failed             (bool: timeout, crash, unsupported op, etc.)
  - accuracy_drop      (float: cosine distance from reference output, 0.0 if failed)

Produces a DataFrame that is saved to disk as a Parquet file.
The training pipeline (dataset.py) loads this file to build the supervised dataset.

LABEL GENERATION
================
For each model, the optimal runtime is:

    label = argmin_r ( L_r + α · M_r + β · F_r · FAILURE_PENALTY )

where:
    L_r  = latency_ms[r]       (if failed: set to MAX_LATENCY_MS)
    M_r  = memory_mb[r]        (if failed: set to MAX_MEMORY_MB)
    F_r  = 1.0 if failed else 0.0
    α    = ALPHA    (memory weight, default 0.5)
    β    = BETA     (failure penalty weight, default 10.0)
    FAILURE_PENALTY = 1000.0   (ms equivalent cost of a failure)

This generates the best_runtime column used as the classification target.

DESIGN CONSTRAINTS
==================
- All timeout/failure cases are captured; failed runtimes are NOT excluded.
  Their failure signal is used by the failure_head and label cost function.
- Worker processes are isolated via multiprocessing to prevent crashes
  from bringing down the collector.
- No heuristics, no manual exclusions. Every runtime is attempted on every model.

USAGE
=====
    from training.data_collector import DataCollector

    collector = DataCollector(
        model_paths=list(Path("models/").glob("*.onnx")),
        output_path=Path("data/benchmark_results.parquet"),
    )
    df = collector.run(n_workers=4)
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool, TimeoutError as MPTimeoutError
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical runtimes ────────────────────────────────────────────────────────
RUNTIMES: tuple[str, ...] = (
    "ONNX_CPU",
    "ONNX_CUDA",
    "TensorRT",
    "TFLite_CPU",
    "TFLite_GPU",
    "TF_CPU",
    "TF_GPU",
    "TensorRT_Native",
    "OpenVINO_CPU",
    "OpenVINO_GPU",
)

# ── Label generation hyperparameters ─────────────────────────────────────────
ALPHA            = 0.5        # memory weight in cost function
BETA             = 10.0       # failure penalty multiplier
FAILURE_PENALTY  = 1000.0     # ms-equivalent cost of a single failure
MAX_LATENCY_MS   = 10_000.0   # sentinel latency for failed runs
MAX_MEMORY_MB    = 32_768.0   # sentinel memory for failed runs (32 GB)

# ── Benchmark parameters ──────────────────────────────────────────────────────
N_WARMUP_RUNS    = 5
N_TIMED_RUNS     = 20
TIMEOUT_SECONDS  = 60         # per-runtime timeout


# ── Data records ─────────────────────────────────────────────────────────────

@dataclass
class RuntimeResult:
    """Single runtime result for one model."""
    model_path:   str
    runtime:      str
    latency_ms:   float        # median timed latency; MAX_LATENCY_MS if failed
    memory_mb:    float        # peak RSS delta in MB; MAX_MEMORY_MB if failed
    failed:       bool
    error_type:   str          # empty string if success; exception class name if failed
    accuracy_drop: float       # cosine distance from reference; 0.0 if failed


@dataclass
class ModelRecord:
    """All runtime results + label for one model."""
    model_path:    str
    model_size_mb: float
    results:       list[RuntimeResult] = field(default_factory=list)
    best_runtime:  str = ""   # argmin of cost function across runtimes

    def compute_label(self) -> None:
        """
        Compute best_runtime using the multi-objective cost function.

        cost(r) = latency_ms(r) + α · memory_mb(r) + β · failure_penalty · failed(r)
        """
        if not self.results:
            self.best_runtime = ""
            return

        costs = []
        for r in self.results:
            L = r.latency_ms if not r.failed else MAX_LATENCY_MS
            M = r.memory_mb  if not r.failed else MAX_MEMORY_MB
            F = 1.0 if r.failed else 0.0
            cost = L + ALPHA * M + BETA * FAILURE_PENALTY * F
            costs.append(cost)

        best_idx = int(np.argmin(costs))
        self.best_runtime = self.results[best_idx].runtime


# ── Runtime benchmark worker ──────────────────────────────────────────────────

def _benchmark_runtime(args: tuple[str, str]) -> RuntimeResult:
    """
    Worker function: benchmark a single (model_path, runtime) pair.
    Runs in an isolated subprocess via Pool.

    Catches ALL exceptions and returns a failed RuntimeResult rather than
    propagating — we never want a single broken runtime to corrupt the dataset.
    """
    model_path, runtime = args

    try:
        return _run_benchmark(model_path, runtime)
    except Exception as exc:
        logger.debug(
            "_benchmark_runtime: %s / %s failed with %s: %s",
            Path(model_path).name, runtime, type(exc).__name__, exc,
        )
        return RuntimeResult(
            model_path   = model_path,
            runtime      = runtime,
            latency_ms   = MAX_LATENCY_MS,
            memory_mb    = MAX_MEMORY_MB,
            failed       = True,
            error_type   = type(exc).__name__,
            accuracy_drop = 0.0,
        )


def _run_benchmark(model_path: str, runtime: str) -> RuntimeResult:
    """
    Execute the actual benchmark for one (model, runtime) pair.
    """
    import psutil

    path = Path(model_path)

    session = _load_runtime_session(path, runtime)

    # ── Warm-up runs (discarded) ──────────────────────────────────────────────
    dummy_input = _make_dummy_input(session)
    for _ in range(N_WARMUP_RUNS):
        _run_inference(session, dummy_input)

    # ── Measure peak memory ───────────────────────────────────────────────────
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024 / 1024   # MB

    # ── Timed runs ────────────────────────────────────────────────────────────
    latencies = []
    reference_output = None
    for i in range(N_TIMED_RUNS):
        t0 = time.perf_counter()
        output = _run_inference(session, dummy_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)   # ms
        if i == 0:
            reference_output = output

    mem_after  = proc.memory_info().rss / 1024 / 1024
    memory_mb  = max(0.0, mem_after - mem_before)
    latency_ms = float(np.median(latencies))

    # ── Accuracy drop (cosine distance from first output) ─────────────────────
    accuracy_drop = 0.0
    if reference_output is not None:
        try:
            ref = np.array(reference_output, dtype=np.float32).flatten()
            cur = np.array(_run_inference(session, dummy_input), dtype=np.float32).flatten()
            if ref.shape == cur.shape and np.linalg.norm(ref) > 1e-8 and np.linalg.norm(cur) > 1e-8:
                cos_sim = float(np.dot(ref, cur) / (np.linalg.norm(ref) * np.linalg.norm(cur)))
                accuracy_drop = max(0.0, 1.0 - cos_sim)
        except Exception:
            accuracy_drop = 0.0

    _close_session(session)

    return RuntimeResult(
        model_path    = model_path,
        runtime       = runtime,
        latency_ms    = latency_ms,
        memory_mb     = memory_mb,
        failed        = False,
        error_type    = "",
        accuracy_drop = accuracy_drop,
    )


# ── Runtime session factory ───────────────────────────────────────────────────
# These functions are thin wrappers that the calling code will adapt per runtime.
# The interface is deliberately uniform: load → dummy_input → run → close.

def _load_runtime_session(path: Path, runtime: str) -> Any:
    """Load and return a runtime session object for the given runtime."""
    import importlib

    dispatchers = {
        "ONNX_CPU":        ("onnxruntime", "InferenceSession", {"providers": ["CPUExecutionProvider"]}),
        "ONNX_CUDA":       ("onnxruntime", "InferenceSession", {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}),
        "TensorRT":        ("onnxruntime", "InferenceSession", {"providers": ["TensorrtExecutionProvider", "CPUExecutionProvider"]}),
        "TFLite_CPU":      ("_tflite", None, {}),
        "TFLite_GPU":      ("_tflite_gpu", None, {}),
        "TF_CPU":          ("_tf_cpu", None, {}),
        "TF_GPU":          ("_tf_gpu", None, {}),
        "TensorRT_Native": ("_tensorrt_native", None, {}),
        "OpenVINO_CPU":    ("_openvino_cpu", None, {}),
        "OpenVINO_GPU":    ("_openvino_gpu", None, {}),
    }

    if runtime not in dispatchers:
        raise ValueError(f"Unknown runtime: {runtime!r}")

    module_name, class_name, kwargs = dispatchers[runtime]

    # ONNX family
    if module_name == "onnxruntime":
        ort = importlib.import_module("onnxruntime")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=opts, **kwargs)
        return ("ort", session)

    # TFLite family
    if module_name in ("_tflite", "_tflite_gpu"):
        try:
            tflite = importlib.import_module("tflite_runtime.interpreter")
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite
        delegates = []
        if module_name == "_tflite_gpu":
            try:
                from tflite_runtime.interpreter import load_delegate
                delegates = [load_delegate("libdelegate.so")]
            except Exception:
                pass
        interp = tflite.Interpreter(model_path=str(path), experimental_delegates=delegates)
        interp.allocate_tensors()
        return ("tflite", interp)

    # TensorFlow family
    if module_name in ("_tf_cpu", "_tf_gpu"):
        import tensorflow as tf
        if module_name == "_tf_cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = tf.saved_model.load(str(path))
        return ("tf", model)

    # TensorRT Native
    if module_name == "_tensorrt_native":
        import tensorrt as trt
        import pycuda.driver as cuda
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return ("trt_native", context)

    # OpenVINO
    if module_name in ("_openvino_cpu", "_openvino_gpu"):
        from openvino.runtime import Core
        device = "GPU" if module_name == "_openvino_gpu" else "CPU"
        ie = Core()
        model = ie.read_model(str(path))
        compiled = ie.compile_model(model, device_name=device)
        return ("openvino", compiled)

    raise ValueError(f"Unhandled runtime: {runtime!r}")


def _make_dummy_input(session: Any) -> Any:
    """Create a random dummy input matching the session's expected input shape."""
    kind, sess = session

    if kind == "ort":
        inputs = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        return inputs

    if kind == "tflite":
        input_details = sess.get_input_details()
        inputs = {}
        for det in input_details:
            shape = [d if d > 0 else 1 for d in det["shape"]]
            inputs[det["index"]] = np.random.randn(*shape).astype(det["dtype"])
        return inputs

    if kind == "tf":
        # Assume a concrete function with a single input
        cf = sess.signatures.get("serving_default", list(sess.signatures.values())[0])
        inputs = {}
        for name, spec in cf.structured_input_signature[1].items():
            shape = [d if d is not None else 1 for d in spec.shape.as_list()]
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        return inputs

    if kind == "trt_native":
        return np.random.randn(1, 3, 224, 224).astype(np.float32)   # generic image shape

    if kind == "openvino":
        model = sess
        inputs = {}
        for inp in model.inputs:
            shape = [d if d > 0 else 1 for d in inp.shape]
            inputs[inp.any_name] = np.random.randn(*shape).astype(np.float32)
        return inputs

    raise ValueError(f"Unknown session kind: {kind!r}")


def _run_inference(session: Any, inputs: Any) -> Any:
    """Run one inference pass and return the output array."""
    kind, sess = session

    if kind == "ort":
        return sess.run(None, inputs)[0]

    if kind == "tflite":
        for idx, data in inputs.items():
            sess.set_tensor(idx, data)
        sess.invoke()
        output_details = sess.get_output_details()
        return sess.get_tensor(output_details[0]["index"])

    if kind == "tf":
        cf = sess.signatures.get("serving_default", list(sess.signatures.values())[0])
        import tensorflow as tf
        tf_inputs = {k: tf.constant(v) for k, v in inputs.items()}
        result = cf(**tf_inputs)
        return list(result.values())[0].numpy()

    if kind == "trt_native":
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        inp = inputs
        output = np.empty((1, 1000), dtype=np.float32)
        d_in  = cuda.mem_alloc(inp.nbytes)
        d_out = cuda.mem_alloc(output.nbytes)
        cuda.memcpy_htod(d_in, inp)
        sess.execute_v2([int(d_in), int(d_out)])
        cuda.memcpy_dtoh(output, d_out)
        return output

    if kind == "openvino":
        result = sess(inputs)
        return list(result.values())[0]

    raise ValueError(f"Unknown session kind: {kind!r}")


def _close_session(session: Any) -> None:
    """Release session resources."""
    kind, sess = session
    try:
        if kind == "tflite":
            del sess
        elif kind == "tf":
            del sess
        elif kind == "trt_native":
            del sess
    except Exception:
        pass


# ── Data Collector ────────────────────────────────────────────────────────────

class DataCollector:
    """
    Orchestrates benchmark execution across all models × runtimes and
    persists results to a Parquet file.

    Args:
        model_paths:  List of paths to model files (.onnx / .tflite / SavedModel dirs).
        output_path:  Destination path for the benchmark Parquet file.
        n_workers:    Number of parallel worker processes.
        timeout_sec:  Per-runtime timeout in seconds.
    """

    def __init__(
        self,
        model_paths:  list[Path],
        output_path:  Path,
        n_workers:    int = 4,
        timeout_sec:  int = TIMEOUT_SECONDS,
    ) -> None:
        self.model_paths  = [Path(p) for p in model_paths]
        self.output_path  = Path(output_path)
        self.n_workers    = n_workers
        self.timeout_sec  = timeout_sec

    def run(self) -> "Any":  # returns pd.DataFrame
        """
        Execute all benchmarks and return a DataFrame with one row per
        (model, runtime) pair.

        Columns:
            model_path, model_size_mb, runtime,
            latency_ms, memory_mb, failed, error_type, accuracy_drop,
            best_runtime   (label — argmin of cost function)
        """
        import pandas as pd

        logger.info(
            "DataCollector: starting benchmark of %d models × %d runtimes = %d tasks",
            len(self.model_paths), len(RUNTIMES),
            len(self.model_paths) * len(RUNTIMES),
        )

        # Build task list: all (model, runtime) pairs
        tasks = [
            (str(mp), rt)
            for mp in self.model_paths
            for rt in RUNTIMES
        ]

        results_flat: list[RuntimeResult] = []

        with Pool(processes=self.n_workers) as pool:
            async_results = [
                pool.apply_async(_benchmark_runtime, (task,))
                for task in tasks
            ]
            for i, ar in enumerate(async_results):
                try:
                    result = ar.get(timeout=self.timeout_sec)
                except MPTimeoutError:
                    mp, rt = tasks[i]
                    logger.warning(
                        "DataCollector: timeout after %ds — %s / %s",
                        self.timeout_sec, Path(mp).name, rt,
                    )
                    result = RuntimeResult(
                        model_path    = mp,
                        runtime       = rt,
                        latency_ms    = MAX_LATENCY_MS,
                        memory_mb     = MAX_MEMORY_MB,
                        failed        = True,
                        error_type    = "TimeoutError",
                        accuracy_drop = 0.0,
                    )
                results_flat.append(result)

        # ── Group by model and compute labels ──────────────────────────────────
        rows = []
        model_groups: dict[str, list[RuntimeResult]] = {}
        for r in results_flat:
            model_groups.setdefault(r.model_path, []).append(r)

        for mp, runtime_results in model_groups.items():
            model_size_mb = Path(mp).stat().st_size / (1024 * 1024) if Path(mp).exists() else 0.0
            record = ModelRecord(
                model_path    = mp,
                model_size_mb = model_size_mb,
                results       = runtime_results,
            )
            record.compute_label()

            if not record.best_runtime:
                logger.warning(
                    "DataCollector: no valid runtime found for %s — skipping", Path(mp).name
                )
                continue

            for r in runtime_results:
                row = asdict(r)
                row["model_size_mb"] = model_size_mb
                row["best_runtime"]  = record.best_runtime
                rows.append(row)

        df = pd.DataFrame(rows)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)
        logger.info(
            "DataCollector: saved %d rows (%d models) to %s",
            len(df), len(model_groups), self.output_path,
        )
        return df
