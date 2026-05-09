"""
core/runtime/tf_evaluator.py

TensorFlow runtime benchmark for .pb model files.
No ONNX conversion. No imports from services or api layers.

CONTRACT (identical to evaluate_onnx_runtime / evaluate_torch_runtime):
    {
        success:         bool
        latency_avg_ms:  float | None   (None iff success=False)
        latency_p95_ms:  float | None   (None iff success=False)
        memory_mb:       float | None   (None iff success=False)
        error:           str   | None   (None iff success=True)
        benchmark_runs:  int
        stress_test:     dict
    }

LOADING STRATEGY (two-pass, automatic):
    1. Try tf.saved_model.load() on the .pb file's parent directory.
       Many .pb files are the protobuf blob inside a SavedModel directory;
       loading the parent recovers the full SavedModel metadata.
    2. If that fails, fall back to loading as a TF1 frozen graph via
       tf.compat.v1:
           graph_def = tf.compat.v1.GraphDef()
           graph_def.ParseFromString(open(model_path, 'rb').read())
           tf.import_graph_def(graph_def, name='')
       Both strategies are attempted before the function gives up.

INPUT SHAPE:
    SavedModel path  — NHWC shapes are probed in order until one runs.
    Frozen graph path — Placeholder op shape is read directly from the graph;
                        dynamic dims are resolved via _resolve_tf_shape();
                        fallback: (1, 224, 224, 3) NHWC.
    The canonical NCHW probe (1, 3, 224, 224) used in the other evaluators
    is transposed to NHWC (1, 224, 224, 3) for TF.

DEVICE CONTROL:
    use_gpu=False → SavedModel: tf.device('/CPU:0') context wraps every call.
                  → Frozen graph: ConfigProto(device_count={'GPU': 0}).
    use_gpu=True  → No restriction; TF selects the best available device.

INVARIANTS:
    - Never raises — all exceptions are caught and surfaced via error key.
    - No imports from services or api layers.
    - No IO side-effects beyond the model file read.
    - No global mutable state.
    - Frozen graph tf.compat.v1.Session is always closed (try/finally).
    - stress_test schema is identical to evaluator.py _DEFAULT_STRESS.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Optional imports (never raise at module level) ────────────────────────────

try:
    import tensorflow as tf  # type: ignore[import]
except ImportError:
    tf = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import]
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


# ── Benchmark constants ───────────────────────────────────────────────────────

_WARMUP_RUNS    = 3
_BENCHMARK_RUNS = 10
_WALL_CAP_S     = 2.0

# TF conventionally uses NHWC (batch, height, width, channels).
# The canonical NCHW probe (1, 3, 224, 224) from other evaluators is
# transposed to NHWC (1, 224, 224, 3) here as the fallback.
_FALLBACK_SHAPE_NHWC: tuple[int, ...] = (1, 224, 224, 3)

# Shapes probed in order for the SavedModel path (NHWC).
# Mirrors the NCHW ordering used in evaluator._PROBE_SHAPES (the ONNX evaluator).
_NHWC_PROBE_SHAPES: list[tuple[int, ...]] = [
    (1, 224, 224, 3),
    (1, 640, 640, 3),
    (1, 256, 256, 3),
    (1, 28,  28,  1),
    (1, 512),
]

# Op types that are never treated as terminal output nodes.
_NON_TERMINAL_OP_TYPES: frozenset[str] = frozenset({
    "Const", "Variable", "VariableV2", "VarHandleOp",
    "Identity",   # pass-through — not a real output boundary
    "NoOp", "Assign", "SaveV2", "RestoreV2",
    "IsVariableInitialized", "Assert",
})

_DEFAULT_STRESS: dict[str, Any] = {
    "enabled":          False,
    "runs":             0,
    "latency_avg_ms":   None,
    "latency_p95_ms":   None,
    "peak_memory_mb":   None,
    "memory_growth_mb": None,
    "memory_stability": "STABLE",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rss_bytes() -> int:
    """Return current process RSS in bytes, or 0 if psutil is unavailable."""
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            pass
    return 0


def _resolve_tf_shape(
    raw_dims: list[Any],
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Materialise a list of TF dimension values (int, None, or -1) into a
    concrete shape tuple.

    Rules:
        index 0 (batch)  → always 1
        known positive d → kept as-is
        None / <= 0      → corresponding fallback value, or 224 if out of range
    """
    if not raw_dims:
        return fallback
    resolved: list[int] = []
    for i, d in enumerate(raw_dims):
        if i == 0:
            resolved.append(1)
        elif isinstance(d, int) and d > 0:
            resolved.append(d)
        elif i < len(fallback):
            resolved.append(int(fallback[i]))
        else:
            resolved.append(224)
    return tuple(resolved)


def _detect_frozen_io(
    graph: "Any",
) -> "tuple[str | None, list[str], tuple[int, ...]]":
    """
    Scan a frozen tf.Graph for input Placeholders and terminal output ops.

    Input detection:
        Ops of type ``Placeholder`` are the graph's external inputs.
        Preference: first Placeholder whose output dtype is float32.
        Fallback: first Placeholder of any dtype.

    Output detection:
        Ops whose output tensors are not consumed as inputs by any other
        op in the graph (excluding noise types in _NON_TERMINAL_OP_TYPES).
        If none are found, the last non-const op is used as a last resort.

    Returns:
        (input_tensor_name, output_tensor_names, resolved_input_shape)
        input_tensor_name is None when no Placeholder exists (load failure).
    """
    ops = list(graph.get_operations())

    # Build the complete set of tensor names that feed at least one other op.
    consumed: set[str] = set()
    for op in ops:
        for inp_tensor in op.inputs:
            consumed.add(inp_tensor.name)

    placeholder_ops: list[Any] = []
    terminal_ops:    list[Any] = []

    for op in ops:
        if op.type == "Placeholder":
            placeholder_ops.append(op)
        elif (
            op.type not in _NON_TERMINAL_OP_TYPES
            and op.outputs
            and all(t.name not in consumed for t in op.outputs)
        ):
            terminal_ops.append(op)

    # Select input: prefer first float32 Placeholder
    chosen_input_op: "Any | None" = None
    for op in placeholder_ops:
        try:
            if op.outputs and op.outputs[0].dtype.name == "float32":
                chosen_input_op = op
                break
        except Exception:
            continue
    if chosen_input_op is None and placeholder_ops:
        chosen_input_op = placeholder_ops[0]

    if chosen_input_op is None:
        return None, [], _FALLBACK_SHAPE_NHWC

    # Input tensor name (e.g. "input:0", "x:0")
    input_tensor = chosen_input_op.outputs[0]
    input_name   = input_tensor.name

    # Resolve input shape: read from TensorShape, resolve dynamic dims
    try:
        raw_dims: list[Any] = [
            (d.value if hasattr(d, "value") else d)
            for d in input_tensor.shape.dims
        ] if input_tensor.shape.dims is not None else []
    except Exception:
        raw_dims = []
    input_shape = _resolve_tf_shape(raw_dims, _FALLBACK_SHAPE_NHWC)

    # Collect output tensor names from terminal ops
    output_names: list[str] = []
    for op in terminal_ops:
        for t in op.outputs:
            output_names.append(t.name)

    # Last-resort fallback: final non-noise op's first output
    if not output_names:
        for op in reversed(ops):
            if op.type not in _NON_TERMINAL_OP_TYPES and op.outputs:
                output_names = [op.outputs[0].name]
                break

    return input_name, output_names, input_shape


def _close_session(session: "Any") -> None:
    """Close a tf.compat.v1.Session safely; silently ignore all errors."""
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


def _run_stress(
    run_fn: "Any",
    dummy:  "Any",
    runs:   int,
) -> dict[str, Any]:
    """
    Execute *runs* sequential inference calls and collect memory growth.

    Returns _DEFAULT_STRESS on any exception — never raises.
    Schema is identical to the _DEFAULT_STRESS in evaluator.py and
    evaluator.py (the ONNX evaluator) so onnx_benchmark_service.py's field extraction
    is compatible without modification.
    """
    if runs <= 0:
        return _DEFAULT_STRESS

    try:
        rss_snapshots: list[int]   = []
        timings_ms:    list[float] = []

        wall_start = time.perf_counter()
        for _ in range(runs):
            rss_snapshots.append(_rss_bytes())
            t0 = time.perf_counter()
            run_fn(dummy)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            if (time.perf_counter() - wall_start) >= _WALL_CAP_S * 3:
                break  # hard wall: 3× the normal cap for stress runs

        if not timings_ms:
            return _DEFAULT_STRESS

        avg_ms    = float(sum(timings_ms) / len(timings_ms))
        sorted_ms = sorted(timings_ms)
        p95_idx   = max(0, int(len(sorted_ms) * 0.95) - 1)
        p95_ms    = float(sorted_ms[p95_idx])

        growth_mb = 0.0
        if len(rss_snapshots) >= 2:
            growth_mb = float(
                max(0, rss_snapshots[-1] - rss_snapshots[0])
            ) / (1024.0 * 1024.0)

        peak_mb = (
            float(max(rss_snapshots) - rss_snapshots[0]) / (1024.0 * 1024.0)
            if len(rss_snapshots) >= 2 else 0.0
        )

        return {
            "enabled":          True,
            "runs":             len(timings_ms),
            "latency_avg_ms":   avg_ms,
            "latency_p95_ms":   p95_ms,
            "peak_memory_mb":   round(peak_mb, 3),
            "memory_growth_mb": round(growth_mb, 3),
            "memory_stability": "STABLE" if growth_mb < 1.0 else "UNSTABLE",
        }

    except Exception as stress_exc:
        logger.warning(
            "tf_stress_failed",
            extra={
                "event":  "tf_stress_failed",
                "reason": str(stress_exc),
            },
        )
        return _DEFAULT_STRESS


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_tf_runtime(
    model_path:  str,
    use_gpu:     bool = False,
    stress_runs: int  = 0,
) -> dict[str, Any]:
    """
    Benchmark a TensorFlow .pb model file natively — no ONNX conversion.

    Two loading strategies are attempted automatically:
        1. tf.saved_model.load() on the parent directory (SavedModel format).
        2. tf.compat.v1 frozen graph loading (legacy frozen-graph .pb).
    The first strategy that succeeds is used; both failing returns success=False.

    Args:
        model_path:  Path to a .pb file (SavedModel blob or frozen graph).
        use_gpu:     False → force CPU execution.
                     True  → allow GPU if TF can find one.
        stress_runs: Extra sequential forward passes after the main benchmark
                     for memory stability analysis.  0 disables stress tests.

    Returns:
        dict with keys:
            success         bool
            latency_avg_ms  float | None
            latency_p95_ms  float | None
            memory_mb       float | None
            error           str   | None
            benchmark_runs  int
            stress_test     dict
    """
    # ── Dependency guard ──────────────────────────────────────────────────────
    if tf is None:
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error": (
                "tensorflow is not installed; "
                "install with: pip install tensorflow"
            ),
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }
    if np is None:
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          "numpy is not installed; install with: pip install numpy",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

    # ── Mutable load-state: session must be closed on all exit paths ──────────
    session: "Any" = None

    try:
        # ── Stage 1: resolve model path ───────────────────────────────────────
        try:
            pb_path    = Path(model_path)
            parent_dir = str(pb_path.parent)
        except Exception as path_exc:
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          f"tf_invalid_path: {path_exc}",
                "benchmark_runs": 0,
                "stress_test":    _DEFAULT_STRESS,
            }

        # ── Stage 2: load the model (SavedModel → frozen graph fallback) ───────
        load_mode:   str                  = ""
        sm_model:    "Any"                = None
        graph:       "Any"                = None
        input_name:  "str | None"         = None
        output_names: list[str]           = []
        input_shape: tuple[int, ...]      = _FALLBACK_SHAPE_NHWC

        # 2a. Try SavedModel on the parent directory
        sm_load_error: str = ""
        try:
            sm_model  = tf.saved_model.load(parent_dir)
            load_mode = "saved_model"
        except Exception as sm_exc:
            sm_load_error = str(sm_exc)

        # 2b. Frozen graph fallback
        if load_mode != "saved_model":
            fg_load_error: str = ""
            try:
                graph    = tf.Graph()
                graph_def = tf.compat.v1.GraphDef()
                with open(model_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
                with graph.as_default():
                    # name='' imports with no prefix — tensor names match the
                    # original frozen graph exactly (e.g. "input:0" not "import/input:0")
                    tf.import_graph_def(graph_def, name="")

                input_name, output_names, input_shape = _detect_frozen_io(graph)

                if input_name is None:
                    raise RuntimeError(
                        "no Placeholder op found — "
                        "cannot identify a float32 input in the frozen graph"
                    )
                if not output_names:
                    raise RuntimeError(
                        "no terminal output ops found — "
                        "frozen graph has no detectable output nodes"
                    )

                # Session config: optionally hide all GPUs for CPU-only mode
                if use_gpu:
                    sess_cfg = tf.compat.v1.ConfigProto()
                else:
                    sess_cfg = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

                session   = tf.compat.v1.Session(graph=graph, config=sess_cfg)
                load_mode = "frozen_graph"

            except Exception as fg_exc:
                fg_load_error = str(fg_exc)

        if not load_mode:
            # Both strategies failed — surface both error messages
            err_msg = (
                f"tf_load_failed: "
                f"saved_model_error='{sm_load_error}'; "
                f"frozen_graph_error='{fg_load_error}'"
            )
            logger.error(
                "tf_benchmark_failed",
                extra={
                    "event":              "tf_benchmark_failed",
                    "model_path":         model_path,
                    "stage":              "load",
                    "saved_model_error":  sm_load_error,
                    "frozen_graph_error": fg_load_error,
                },
            )
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          err_msg,
                "benchmark_runs": 0,
                "stress_test":    _DEFAULT_STRESS,
            }

        # ── Stage 3: build run_fn and dummy input ─────────────────────────────

        if load_mode == "saved_model":
            # Probe NHWC shapes to find a working input for the SavedModel callable.
            # Direct call is tried first; TF Serving signature-based models that
            # require named kwargs are not probed — they should use the ONNX path.
            working_shape: "tuple[int, ...] | None" = None

            for shape in _NHWC_PROBE_SHAPES:
                try:
                    probe_dummy   = np.zeros(shape, dtype=np.float32)
                    probe_const   = tf.constant(probe_dummy, dtype=tf.float32)
                    if use_gpu:
                        sm_model(probe_const)
                    else:
                        with tf.device("/CPU:0"):
                            sm_model(probe_const)
                    working_shape = shape
                    break
                except Exception:
                    continue

            if working_shape is None:
                logger.error(
                    "tf_benchmark_failed",
                    extra={
                        "event":         "tf_benchmark_failed",
                        "model_path":    model_path,
                        "stage":         "input_probe",
                        "probed_shapes": [list(s) for s in _NHWC_PROBE_SHAPES],
                        "reason":        "all NHWC shapes failed direct call",
                    },
                )
                return {
                    "success":        False,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "memory_mb":      None,
                    "error": (
                        f"tf_input_probe_failed: none of {_NHWC_PROBE_SHAPES} "
                        "produced a successful forward pass via direct call; "
                        "signature-based SavedModels (TF Serving exports) "
                        "are not supported — use the ONNX path instead"
                    ),
                    "benchmark_runs": 0,
                    "stress_test":    _DEFAULT_STRESS,
                }

            input_shape = working_shape
            dummy       = np.zeros(input_shape, dtype=np.float32)

            # run_fn: device context wraps every call for CPU-only mode
            if use_gpu:
                def run_fn(d: np.ndarray) -> None:
                    sm_model(tf.constant(d, dtype=tf.float32))
            else:
                def run_fn(d: np.ndarray) -> None:
                    with tf.device("/CPU:0"):
                        sm_model(tf.constant(d, dtype=tf.float32))

        else:  # frozen_graph
            dummy = np.zeros(input_shape, dtype=np.float32)
            _sess        = session       # captured for closure; must not be None here
            _input_name  = input_name    # e.g. "input:0"
            _output_names = output_names  # e.g. ["softmax:0"]

            def run_fn(d: np.ndarray) -> None:
                _sess.run(_output_names, feed_dict={_input_name: d})

        # ── Stage 4: log benchmark start ──────────────────────────────────────
        logger.info(
            "tf_benchmark_start",
            extra={
                "event":          "tf_benchmark_start",
                "model_path":     model_path,
                "load_mode":      load_mode,
                "input_shape":    list(input_shape),
                "use_gpu":        use_gpu,
                "warmup_runs":    _WARMUP_RUNS,
                "benchmark_runs": _BENCHMARK_RUNS,
                "stress_runs":    stress_runs,
                **({"input_name":   input_name,
                    "output_names": output_names}
                   if load_mode == "frozen_graph" else {}),
            },
        )

        # ── Stage 5: warmup ───────────────────────────────────────────────────
        try:
            for _ in range(_WARMUP_RUNS):
                run_fn(dummy)
        except Exception as warmup_exc:
            logger.error(
                "tf_benchmark_failed",
                extra={
                    "event":      "tf_benchmark_failed",
                    "model_path": model_path,
                    "load_mode":  load_mode,
                    "stage":      "warmup",
                    "reason":     str(warmup_exc),
                },
            )
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          f"tf_warmup_failed: {warmup_exc}",
                "benchmark_runs": 0,
                "stress_test":    _DEFAULT_STRESS,
            }

        # ── Stage 6: timed benchmark ──────────────────────────────────────────
        try:
            rss_before  = _rss_bytes()
            timings_ms: list[float] = []
            wall_start  = time.perf_counter()

            for _ in range(_BENCHMARK_RUNS):
                t0 = time.perf_counter()
                run_fn(dummy)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
                if (time.perf_counter() - wall_start) >= _WALL_CAP_S:
                    break

            rss_after = _rss_bytes()

        except Exception as bench_exc:
            logger.error(
                "tf_benchmark_failed",
                extra={
                    "event":      "tf_benchmark_failed",
                    "model_path": model_path,
                    "load_mode":  load_mode,
                    "stage":      "benchmark",
                    "reason":     str(bench_exc),
                },
            )
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          f"tf_benchmark_run_failed: {bench_exc}",
                "benchmark_runs": 0,
                "stress_test":    _DEFAULT_STRESS,
            }

        # ── Stage 7: stress test (must run before session close) ──────────────
        stress_result = _run_stress(run_fn, dummy, stress_runs)

    finally:
        # Guarantee the tf.compat.v1.Session is always released.
        # The SavedModel path leaves session=None, so this is a no-op there.
        _close_session(session)

    # ── Stage 8: aggregate latency ────────────────────────────────────────────
    runs_completed = len(timings_ms)
    latency_avg_ms = float(sum(timings_ms) / runs_completed)
    sorted_ms      = sorted(timings_ms)
    p95_idx        = max(0, int(runs_completed * 0.95) - 1)
    latency_p95_ms = float(sorted_ms[p95_idx])

    # ── Stage 9: memory ───────────────────────────────────────────────────────
    memory_mb = float(max(0, rss_after - rss_before)) / (1024.0 * 1024.0)

    # ── Stage 10: structured completion log ───────────────────────────────────
    logger.info(
        "tf_benchmark_complete",
        extra={
            "event":            "tf_benchmark_complete",
            "model_path":       model_path,
            "load_mode":        load_mode,
            "input_shape":      list(input_shape),
            "latency_avg_ms":   round(latency_avg_ms, 3),
            "latency_p95_ms":   round(latency_p95_ms, 3),
            "memory_mb":        round(memory_mb, 2),
            "runs_completed":   runs_completed,
            "stress_runs":      stress_result.get("runs", 0),
            "memory_stability": stress_result.get("memory_stability", "STABLE"),
        },
    )

    return {
        "success":        True,
        "latency_avg_ms": latency_avg_ms,
        "latency_p95_ms": latency_p95_ms,
        "memory_mb":      memory_mb,
        "error":          None,
        "benchmark_runs": runs_completed,
        "stress_test":    stress_result,
    }
