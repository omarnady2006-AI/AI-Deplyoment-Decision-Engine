"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any
from uuid import uuid4

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

logger = logging.getLogger(__name__)

# ── Resource-limit constants ───────
_MIN_FREE_DISK_BYTES = 512 * 1024 * 1024        # 512 MB minimum free disk
_MAX_MODEL_LOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB max file size
_MAX_THREAD_POOL_WORKERS = 64

# ── Circuit-breaker — AFTER: encapsulated in a class, zero bare globals ──────
class _CircuitBreaker:
    """Per-instance circuit breaker — no module-level mutable state."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_seconds: float = 60.0,
    ) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._failure_count: int = 0
        self._last_reset: float = 0.0
        self._open: bool = False
        self._failure_threshold: int = failure_threshold
        self._reset_seconds: float = reset_seconds

    def record_failure(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._last_reset > self._reset_seconds:
                self._failure_count = 0
                self._last_reset = now
                self._open = False
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._open = True

    def is_open(self) -> bool:
        """Return True if circuit is open (reject request)."""
        with self._lock:
            now = time.monotonic()
            if self._open and (now - self._last_reset > self._reset_seconds):
                self._open = False
                self._failure_count = 0
                self._last_reset = now
            return self._open


# Module-level singleton — one per process, no bare mutable globals.
_CIRCUIT_BREAKER: _CircuitBreaker = _CircuitBreaker(
    failure_threshold=int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "5")),
    reset_seconds=float(os.environ.get("CIRCUIT_BREAKER_RESET_S", "60.0")),
)


def _circuit_breaker_record_failure() -> None:
    _CIRCUIT_BREAKER.record_failure()


def _circuit_breaker_check() -> bool:
    """Return True if circuit is open (reject request)."""
    return _CIRCUIT_BREAKER.is_open()


def _check_disk_space(path: str | None = None) -> tuple[bool, str]:
    """Return (ok, reason). Fails if free disk < _MIN_FREE_DISK_BYTES.

    RULE 17: defaults to INSTANCE_DIR instead of the global /tmp.
    """
    from src.core.paths import INSTANCE_DIR as _IDIR
    check_path = path or str(_IDIR)
    try:
        if psutil is None:
            return True, ""
        usage = psutil.disk_usage(check_path)
        if usage.free < _MIN_FREE_DISK_BYTES:
            return False, f"insufficient_disk: {usage.free} bytes free"
    except Exception as exc:
        return False, f"disk_check_error: {exc}"
    return True, ""


def _check_fd_pressure() -> tuple[bool, str]:
    """Return (ok, reason). Fails when open FDs exceed 80% of soft limit."""
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if psutil is None:
            return True, ""
        proc = psutil.Process()
        open_fds = proc.num_fds()
        if open_fds > soft * 0.8:
            return False, f"fd_exhaustion: {open_fds}/{soft} fds open"
    except Exception:
        logger.warning("fd_check_skipped: platform does not support resource limits", exc_info=True)
    return True, ""


# ── Framework Adapter ──────────────────────────────────────────────────────────
try:
    from framework_adapter import detect_framework as _fa_detect_framework
    from framework_adapter import run_inference as _fa_run_inference
    _FRAMEWORK_ADAPTER_AVAILABLE = True
except ImportError:
    try:
        from src.core.framework_adapter import detect_framework as _fa_detect_framework
        from src.core.framework_adapter import run_inference as _fa_run_inference
        _FRAMEWORK_ADAPTER_AVAILABLE = True
    except ImportError:
        _FRAMEWORK_ADAPTER_AVAILABLE = False
        _fa_detect_framework = None  # type: ignore[assignment]
        _fa_run_inference = None     # type: ignore[assignment]


# ── Inference support ──────────────────────────────────────────────────────────

def _build_input_feed(session: ort.InferenceSession, batch_size: int) -> dict[str, np.ndarray]:
    """
    Construct a zero-filled input feed from session metadata.
    No accumulation, no size estimation, no shape product loops.
    Dynamic dimensions replaced with 1 via inline conditional.
    """
    feed: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        raw_shape = list(getattr(model_input, "shape", []) or [])
        shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in raw_shape] or [1]
        shape[0] = batch_size
        if np is not None:
            feed[model_input.name] = np.zeros(shape, dtype=np.float32)
        else:
            n = 1
            for d in shape:
                n *= d
            import array as _arr
            feed[model_input.name] = _arr.array("f", [0.0] * n)
    return feed


def _seed_deterministic() -> None:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    try:
        import torch  # type: ignore
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    except Exception:
        logger.warning("exception_occurred", exc_info=True)


# ── Core measurement function ──────────────────────────────────────────────────

def _profile_with_session(
    model_path: str,
    intra_threads: int,
    warmup_runs: int,
    measured_runs: int,
    batch_size: int,
    hard_deadline: float | None = None,
    request_id: str = "",
) -> tuple[dict[str, Any], str]:
    """
    Session creation + inference execution + time measurement + psutil reads.
    Returns 6 raw signals. Zero computation beyond permitted time/cpu deltas.
    """
    if ort is None:
        return dict(_RAW_ZERO), ""
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1

    # cold_start: permitted time delta
    t0 = time.time()
    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    cold_start_ms = (time.time() - t0) * 1000.0

    input_feed = _build_input_feed(session, batch_size=batch_size)

    # Single warm-up run (not recorded)
    session.run(None, input_feed)

    # Direct psutil reads — rss and cpu captured before/after
    if psutil is not None:
        process = psutil.Process()
        rss_before = int(process.memory_info().rss)
        cpu_times_before = process.cpu_times()
    else:
        process = None
        rss_before = 0
        cpu_times_before = None
    warm_latencies: list[float] = []
    for _ in range(warmup_runs):
        session.run(None, input_feed)
    for _ in range(measured_runs):
        if hard_deadline is not None and time.monotonic() >= hard_deadline:
            break
        t0 = time.perf_counter()
        session.run(None, input_feed)
        # latency: permitted time delta
        warm_latencies.append((time.perf_counter() - t0) * 1000.0)

    if psutil is not None and process is not None:
        cpu_times_after = process.cpu_times()
        rss_after = int(process.memory_info().rss)
    else:
        cpu_times_after = None
        rss_after = 0

    try:
        profiling_file = session.end_profiling()
    except Exception:
        profiling_file = ""

    # 6 raw signals only.
    # cpu deltas are the ONLY permitted subtraction (explicit exception).
    # rss values are direct psutil reads — no arithmetic applied.
    return {
        "latency_samples_ms": [float(v) for v in warm_latencies],
        "rss_bytes_before":   int(rss_before),
        "rss_bytes_after":    int(rss_after),
        "cpu_time_user_s":    float(cpu_times_after.user - cpu_times_before.user) if cpu_times_after and cpu_times_before else 0.0,
        "cpu_time_system_s":  float(cpu_times_after.system - cpu_times_before.system) if cpu_times_after and cpu_times_before else 0.0,
        "cold_start_ms":      float(cold_start_ms),
    }, profiling_file


# ── Public API ─────────────────────────────────────────────────────────────────

_RAW_ZERO: dict[str, Any] = {
    "latency_samples_ms": [],
    "rss_bytes_before":   0,
    "rss_bytes_after":    0,
    "cpu_time_user_s":    0.0,
    "cpu_time_system_s":  0.0,
    "cold_start_ms":      0.0,
}


def profile_model_runtime(
    model_path: str,
    profiling_budget_ms: float = 800.0,
    deterministic: bool = True,
    previous_runtime: dict[str, Any] | None = None,
    request_id: str = "",
    production_validation: bool = False,
    profile: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Profile a model and return ONLY the 6 raw signals declared in
    signal_registry.py.  No derivation, no aggregation, no smoothing.

    Returns:
        {
            "latency_samples_ms": list[float],
            "rss_bytes_before":   int,
            "rss_bytes_after":    int,
            "cpu_time_user_s":    float,
            "cpu_time_system_s":  float,
            "cold_start_ms":      float,
        }
    """
    if not request_id:
        request_id = str(uuid4())

    # ── Guard: circuit breaker ─────────────────────────────────────────────────
    if _circuit_breaker_check():
        logger.warning("stage=profiling status=rejected reason=circuit_open")
        return dict(_RAW_ZERO)

    # ── Guard: FD exhaustion ───────────────────────────────────────────────────
    fd_ok, fd_reason = _check_fd_pressure()
    if not fd_ok:
        logger.warning("stage=profiling status=rejected reason=%s", fd_reason)
        return dict(_RAW_ZERO)

    # ── Guard: disk space ──────────────────────────────────────────────────────
    disk_ok, disk_reason = _check_disk_space()
    if not disk_ok:
        logger.warning("stage=profiling status=rejected reason=%s", disk_reason)
        return dict(_RAW_ZERO)

    # ── Guard: file size ───────────────────────────────────────────────────────
    try:
        model_file_size = os.path.getsize(model_path)
        if model_file_size > _MAX_MODEL_LOAD_BYTES:
            logger.info(
                "stage=profiling status=rejected reason=file_too_large size=%d",
                model_file_size,
            )
            return dict(_RAW_ZERO)
    except OSError:
        logger.warning("exception_occurred", exc_info=True)

    if deterministic:
        _seed_deterministic()

    # Fixed 1-second hard ceiling — no budget arithmetic
    hard_deadline = time.monotonic() + 1.0

    logger.info(
        "stage=profile status=start path=%s request_id=%s",
        model_path, request_id,
    )

    try:
        # ── Framework detection ────────────────────────────────────────────────
        _fw = "onnx"
        if _FRAMEWORK_ADAPTER_AVAILABLE and _fa_detect_framework is not None:
            try:
                _fw = _fa_detect_framework(model_path)
            except ValueError:
                _fw = "onnx"

        if _fw != "onnx":
            # Non-ONNX path via Framework Adapter.
            # Adapter returns memory_mb; no unit conversion permitted.
            # rss_bytes_after dropped for this path (Option B).
            try:
                latency_ms, _memory_mb = _fa_run_inference(model_path)  # type: ignore[misc]
            except Exception:
                _circuit_breaker_record_failure()
                logger.warning("stage=profiling status=failed reason=framework_adapter_error")
                return dict(_RAW_ZERO)

            return {
                "latency_samples_ms": [float(latency_ms)],
                "rss_bytes_before":   0,
                "rss_bytes_after":    0,
                "cpu_time_user_s":    0.0,
                "cpu_time_system_s":  0.0,
                "cold_start_ms":      0.0,
            }

        # ── ONNX path: direct psutil read for thread count ─────────────────────
        _cpu_cores = psutil.cpu_count(logical=False) if psutil is not None else None
        if _cpu_cores is None:
            _cpu_cores = 1
        intra_threads = _cpu_cores

        raw_signals, _ = _profile_with_session(
            model_path=model_path,
            intra_threads=intra_threads,
            warmup_runs=2,
            measured_runs=5,
            batch_size=1,
            hard_deadline=hard_deadline,
            request_id=request_id,
        )

        logger.info(
            "stage=profile status=complete samples=%d cold_start_ms=%.2f",
            len(raw_signals["latency_samples_ms"]),
            raw_signals["cold_start_ms"],
        )
        return raw_signals

    except Exception as exc:
        _circuit_breaker_record_failure()
        logger.exception("stage=profiling status=error: %s", exc)
        return dict(_RAW_ZERO)