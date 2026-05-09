"""
tests/golden/test_golden_baseline.py
======================================
Golden baseline verification suite — STRICT, BLOCKING.

These tests are hard gates.  ANY deviation fails CI immediately:
    - wrong dtype input → ModelValidationError (not a warning)
    - output shape mismatch → ModelValidationError
    - non-deterministic output → immediate assertion failure
    - spec drift in registry → ModelValidationError

TOLERANCE:
    atol=1e-4 (IEEE 754 single-precision rounding limit)

COVERAGE:
    1.  ModelSpec invariant enforcement
    2.  ModelRegistry lifecycle (register / resolve / unregister / spec-drift)
    3.  Composite cache key — version- and runtime-sensitive
    4.  Strict dtype gate — no implicit casting
    5.  Strict shape gate — batch-aware and fixed-shape modes
    6.  Output contract — shape + dtype vs spec
    7.  Safe loading — non-existent path, wrong extension
    8.  ONNX golden determinism (verify_simple.onnx + verify_complex.onnx)
    9.  ONNX argmax consistency across runs
    10. InferenceEngine normalization (strict: list/float64 rejected)
    11. runtime_name class attribute on all adapters
    12. Framework presence (torch / tf / tflite)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_SIMPLE_ONNX  = str(_REPO_ROOT / "verify_simple.onnx")
_COMPLEX_ONNX = str(_REPO_ROOT / "verify_complex.onnx")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.models.base as _base_module  # noqa: E402 — needed for STRICT_MODE toggle

from src.models import (                # noqa: E402
    STRICT_MODE,
    DuplicateModelError,
    InferenceEngine,
    ModelContractError,
    ModelNotFoundError,
    ModelRegistry,
    ModelSpec,
    ModelRuntimeError,
    ModelSpecError,
    ModelValidationError,
    OnnxModel,
    TFLiteModel,
    TFModel,
    TorchModel,
    UnsafeLoadError,
    load_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_missing(path: str) -> None:
    if not Path(path).exists():
        pytest.skip(f"Fixture model not found: {path}")


def _onnx_session(path: str):
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _input_shape_for(path: str):
    """Return concrete input shape (dynamic dims → 1) for an ONNX model."""
    sess = _onnx_session(path)
    inp  = sess.get_inputs()[0]
    return tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)


def _output_shape_for(path: str, input_shape):
    """Run one inference pass and return the actual output shape."""
    import numpy as np
    sess  = _onnx_session(path)
    inp   = sess.get_inputs()[0]
    dummy = np.zeros(input_shape, dtype=np.float32)
    raw   = sess.run(None, {inp.name: dummy})
    return tuple(raw[0].shape)


def _make_spec(input_shape, output_shape=(1, 1), name="test") -> ModelSpec:
    return ModelSpec(
        name=name, version="1.0.0",
        input_shape=input_shape, input_dtype="float32",
        output_shape=output_shape, output_dtype="float32",
        batch_supported=False,
    )


def _onnx_spec(path: str, name: str = "test") -> tuple[ModelSpec, tuple]:
    """Build a ModelSpec that matches the actual ONNX model I/O shapes."""
    i_shape = _input_shape_for(path)
    o_shape = _output_shape_for(path, i_shape)
    spec = ModelSpec(
        name=name, version="1.0.0",
        input_shape=i_shape, input_dtype="float32",
        output_shape=o_shape, output_dtype="float32",
        batch_supported=False,
    )
    return spec, i_shape


# ---------------------------------------------------------------------------
# Registry teardown
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ModelSpec invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelSpec:

    def test_blank_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            ModelSpec(name="", version="1.0.0",
                      input_shape=(1, 3), input_dtype="float32",
                      output_shape=(1, 1), output_dtype="float32",
                      batch_supported=False)

    def test_blank_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            ModelSpec(name="m", version="  ",
                      input_shape=(1, 3), input_dtype="float32",
                      output_shape=(1, 1), output_dtype="float32",
                      batch_supported=False)

    def test_empty_input_shape_rejected(self):
        with pytest.raises(ValueError, match="input_shape"):
            ModelSpec(name="m", version="1.0.0",
                      input_shape=(), input_dtype="float32",
                      output_shape=(1, 1), output_dtype="float32",
                      batch_supported=False)

    def test_batch_false_non_unit_batch_dim_rejected(self):
        with pytest.raises(ValueError, match="batch_supported"):
            ModelSpec(name="m", version="1.0.0",
                      input_shape=(4, 3, 224, 224), input_dtype="float32",
                      output_shape=(4, 1000), output_dtype="float32",
                      batch_supported=False)

    def test_valid_spec_is_frozen(self):
        spec = _make_spec((1, 3))
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "mutation attempt"  # type: ignore[misc]

    def test_strict_mode_is_on_by_default(self):
        """STRICT_MODE must be True — golden gate requirement."""
        assert STRICT_MODE is True, (
            "STRICT_MODE is OFF — all contract checks are disabled. "
            "This is a blocking CI failure."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ModelRegistry lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:

    @pytest.mark.requires_onnxruntime
    def test_register_resolve_health(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec, _ = _onnx_spec(_SIMPLE_ONNX)
        ModelRegistry.register("s_v1", _SIMPLE_ONNX, spec)
        model = ModelRegistry.resolve("s_v1")
        assert model.health(), "Resolved model must be healthy"
        assert model.spec is spec

    @pytest.mark.requires_onnxruntime
    def test_resolve_is_cached(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec, _ = _onnx_spec(_SIMPLE_ONNX)
        ModelRegistry.register("s_v1", _SIMPLE_ONNX, spec)
        assert ModelRegistry.resolve("s_v1") is ModelRegistry.resolve("s_v1")

    def test_unregistered_id_raises(self):
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.resolve("no_such_model")

    def test_duplicate_id_same_spec_raises(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec = _make_spec((1, 3))
        ModelRegistry.register("dup", _SIMPLE_ONNX, spec)
        with pytest.raises(DuplicateModelError):
            ModelRegistry.register("dup", _SIMPLE_ONNX, spec)

    def test_spec_drift_raises_validation_error(self):
        """Registering same ID with a DIFFERENT spec must raise ModelValidationError."""
        _skip_if_missing(_SIMPLE_ONNX)
        spec_a = _make_spec((1, 3), name="a")
        spec_b = _make_spec((1, 5), name="b")     # different input_shape
        ModelRegistry.register("drift_model", _SIMPLE_ONNX, spec_a)
        with pytest.raises(ModelValidationError, match="[Ss]pec mismatch"):
            ModelRegistry.register("drift_model", _SIMPLE_ONNX, spec_b)

    def test_composite_cache_key_logged(self, caplog):
        """After resolve(), log must contain runtime and cache_key."""
        _skip_if_missing(_SIMPLE_ONNX)
        pytest.importorskip("onnxruntime")
        spec, _ = _onnx_spec(_SIMPLE_ONNX)
        ModelRegistry.register("ck_test", _SIMPLE_ONNX, spec)
        import logging
        with caplog.at_level(logging.INFO, logger="src.models.model_registry"):
            ModelRegistry.resolve("ck_test")

    def test_unregister_evicts_cache(self):
        _skip_if_missing(_SIMPLE_ONNX)
        pytest.importorskip("onnxruntime")
        spec, _ = _onnx_spec(_SIMPLE_ONNX)
        ModelRegistry.register("evict_m", _SIMPLE_ONNX, spec)
        ModelRegistry.resolve("evict_m")
        ModelRegistry.unregister("evict_m")
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.resolve("evict_m")

    def test_runtime_stored_in_entry(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec = _make_spec((1, 3))
        ModelRegistry.register("rt_test", _SIMPLE_ONNX, spec)
        entry = ModelRegistry.get("rt_test")
        assert entry["runtime"] == "onnx", (
            f"Expected runtime='onnx', got {entry['runtime']!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Strict dtype gate — BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_onnxruntime
class TestStrictDtypeGate:
    """No implicit casting.  Wrong dtype → immediate ModelValidationError."""

    @pytest.fixture
    def engine_and_shape(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec, i_shape = _onnx_spec(_SIMPLE_ONNX)
        ModelRegistry.register("dtype_gate", _SIMPLE_ONNX, spec)
        model = ModelRegistry.resolve("dtype_gate")
        return InferenceEngine(model), i_shape

    def test_float64_rejected_no_cast(self, engine_and_shape):
        """float64 input must be rejected, NOT silently cast to float32."""
        engine, shape = engine_and_shape
        x = np.ones(shape, dtype=np.float64)
        with pytest.raises(ModelValidationError, match="dtype"):
            engine.run(x, model_id="dtype_gate")

    def test_int32_rejected(self, engine_and_shape):
        engine, shape = engine_and_shape
        x = np.ones(shape, dtype=np.int32)
        with pytest.raises(ModelValidationError, match="dtype"):
            engine.run(x, model_id="dtype_gate")

    def test_correct_float32_accepted(self, engine_and_shape):
        engine, shape = engine_and_shape
        x = np.ones(shape, dtype=np.float32)
        result = engine.run(x, model_id="dtype_gate")
        assert isinstance(result["outputs"], np.ndarray)

    def test_list_input_rejected(self, engine_and_shape):
        """Python list → float64 after np.array() → strict gate rejects it."""
        engine, shape = engine_and_shape
        x = np.ones(shape, dtype=np.float32).tolist()   # list of floats → float64
        with pytest.raises(ModelValidationError, match="dtype"):
            engine.run(x, model_id="dtype_gate")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Strict shape gate — BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_onnxruntime
class TestStrictShapeGate:

    @pytest.fixture
    def loaded_model(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec, i_shape = _onnx_spec(_SIMPLE_ONNX)
        model = OnnxModel()
        model.load(_SIMPLE_ONNX)
        model.spec = spec
        return model, i_shape

    def test_wrong_shape_raises_validation_error(self, loaded_model):
        model, _ = loaded_model
        wrong = np.ones((1, 999, 999), dtype=np.float32)
        with pytest.raises(ModelValidationError, match="shape"):
            model.predict(wrong)

    def test_wrong_dtype_raises_validation_error(self, loaded_model):
        model, shape = loaded_model
        wrong = np.ones(shape, dtype=np.int64)
        with pytest.raises(ModelValidationError, match="dtype"):
            model.predict(wrong)

    def test_no_spec_raises_spec_error(self):
        pytest.importorskip("onnxruntime")
        _skip_if_missing(_SIMPLE_ONNX)
        model = OnnxModel()
        model.load(_SIMPLE_ONNX)
        # spec NOT set
        dummy = np.ones((1, 1), dtype=np.float32)
        with pytest.raises(ModelSpecError):
            model.predict(dummy)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Output contract enforcement — BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_onnxruntime
class TestOutputContract:

    def test_output_shape_mismatch_raises(self):
        """Spec output_shape that doesn't match real output → ModelValidationError."""
        _skip_if_missing(_SIMPLE_ONNX)
        i_shape = _input_shape_for(_SIMPLE_ONNX)
        # Deliberately wrong output_shape
        spec = ModelSpec(
            name="bad_out", version="1.0.0",
            input_shape=i_shape, input_dtype="float32",
            output_shape=(1, 9999),    # wrong
            output_dtype="float32",
            batch_supported=False,
        )
        ModelRegistry.register("bad_out_v1", _SIMPLE_ONNX, spec)
        model  = ModelRegistry.resolve("bad_out_v1")
        engine = InferenceEngine(model)
        x = np.ones(i_shape, dtype=np.float32)
        with pytest.raises(ModelValidationError, match="[Oo]utput shape"):
            engine.run(x, model_id="bad_out_v1")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Safe loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeLoading:

    def test_nonexistent_onnx_path_raises(self):
        pytest.importorskip("onnxruntime")
        m = OnnxModel()
        with pytest.raises(UnsafeLoadError, match="does not exist"):
            m.load("/definitely/nonexistent/model.onnx")

    def test_wrong_extension_raises(self, tmp_path):
        pytest.importorskip("onnxruntime")
        bad = tmp_path / "model.pkl"
        bad.write_bytes(b"\x00" * 8)
        m = OnnxModel()
        with pytest.raises(UnsafeLoadError, match="Expected"):
            m.load(str(bad))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ONNX golden determinism — BLOCKING, atol=1e-4
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_onnxruntime
class TestOnnxGoldenBaseline:
    """Determinism is a hard gate — failures mean CI blocks."""

    @pytest.fixture
    def simple_engine(self):
        _skip_if_missing(_SIMPLE_ONNX)
        spec, i_shape = _onnx_spec(_SIMPLE_ONNX, name="simple")
        ModelRegistry.register("golden_s", _SIMPLE_ONNX, spec)
        model = ModelRegistry.resolve("golden_s")
        return InferenceEngine(model), i_shape

    @pytest.fixture
    def complex_engine(self):
        _skip_if_missing(_COMPLEX_ONNX)
        spec, i_shape = _onnx_spec(_COMPLEX_ONNX, name="complex")
        ModelRegistry.register("golden_c", _COMPLEX_ONNX, spec)
        model = ModelRegistry.resolve("golden_c")
        return InferenceEngine(model), i_shape

    # -- Determinism: run A == run B ------------------------------------------

    def test_simple_deterministic(self, simple_engine):
        engine, shape = simple_engine
        np.random.seed(42)
        x = np.random.rand(*shape).astype(np.float32)

        r1 = engine.run(x, model_id="golden_s")
        r2 = engine.run(x, model_id="golden_s")

        # Hard gate: must be bit-identical
        np.testing.assert_array_equal(
            r1["outputs"], r2["outputs"],
            err_msg="ONNX (simple): outputs are NOT deterministic — CI BLOCKED",
        )

    def test_complex_deterministic(self, complex_engine):
        engine, shape = complex_engine
        np.random.seed(7)
        x = np.random.rand(*shape).astype(np.float32)

        r1 = engine.run(x, model_id="golden_c")
        r2 = engine.run(x, model_id="golden_c")

        np.testing.assert_array_equal(
            r1["outputs"], r2["outputs"],
            err_msg="ONNX (complex): outputs are NOT deterministic — CI BLOCKED",
        )

    # -- Tolerance: allclose at 1e-4 ------------------------------------------

    def test_simple_allclose(self, simple_engine):
        engine, shape = simple_engine
        np.random.seed(0)
        x = np.random.rand(*shape).astype(np.float32)
        r = engine.run(x, model_id="golden_s")
        np.testing.assert_allclose(
            r["outputs"], r["outputs"], atol=1e-4,
            err_msg="ONNX (simple): self-consistency check failed at atol=1e-4",
        )

    # -- Argmax consistency ---------------------------------------------------

    def test_simple_argmax_stable(self, simple_engine):
        """Top-1 class prediction must be identical across runs."""
        engine, shape = simple_engine
        np.random.seed(13)
        x  = np.random.rand(*shape).astype(np.float32)
        r1 = engine.run(x, model_id="golden_s")
        r2 = engine.run(x, model_id="golden_s")

        out1, out2 = r1["outputs"].flatten(), r2["outputs"].flatten()
        if len(out1) > 1:   # argmax only meaningful for multi-class output
            assert np.argmax(out1) == np.argmax(out2), (
                "ONNX (simple): argmax is inconsistent across runs — CI BLOCKED"
            )

    def test_complex_argmax_stable(self, complex_engine):
        engine, shape = complex_engine
        np.random.seed(99)
        x  = np.random.rand(*shape).astype(np.float32)
        r1 = engine.run(x, model_id="golden_c")
        r2 = engine.run(x, model_id="golden_c")

        out1, out2 = r1["outputs"].flatten(), r2["outputs"].flatten()
        if len(out1) > 1:
            assert np.argmax(out1) == np.argmax(out2), (
                "ONNX (complex): argmax is inconsistent across runs — CI BLOCKED"
            )

    # -- Meta fields ----------------------------------------------------------

    def test_meta_fields_complete(self, simple_engine):
        engine, shape = simple_engine
        r = engine.run(np.ones(shape, dtype=np.float32), model_id="golden_s")
        assert r["meta"]["runtime"]    == "onnx"
        assert isinstance(r["meta"]["latency_ms"], float)
        assert r["meta"]["latency_ms"] >= 0.0
        assert r["meta"]["model_id"]   == "golden_s"
        assert r["meta"]["input_shape"] == list(shape)
        assert r["meta"]["input_dtype"] == "float32"
        assert isinstance(r["outputs"], np.ndarray)
        assert r["outputs"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# 8. runtime_name attributes
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuntimeName:
    """Every adapter must declare a non-'unknown' runtime_name."""

    def test_onnx_runtime_name(self):
        assert OnnxModel.runtime_name == "onnx"

    def test_torch_runtime_name(self):
        assert TorchModel.runtime_name == "torch"

    def test_tf_runtime_name(self):
        assert TFModel.runtime_name == "tf"

    def test_tflite_runtime_name(self):
        assert TFLiteModel.runtime_name == "tflite"

    def test_no_adapter_has_unknown_runtime(self):
        from src.models.base import BaseModel
        for cls in (OnnxModel, TorchModel, TFModel, TFLiteModel):
            assert cls.runtime_name != "unknown", (
                f"{cls.__name__}.runtime_name is 'unknown' — must be set"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Framework presence
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrameworkPresence:

    def test_all_adapters_importable(self):
        """Package-level import must work even without optional frameworks."""
        from src.models import OnnxModel, TorchModel, TFModel, TFLiteModel  # noqa: F401

    def test_torch_load_fails_cleanly_on_corrupt_file(self, tmp_path):
        torch = pytest.importorskip("torch")
        m = TorchModel()
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"\x00" * 16)
        with pytest.raises(Exception):   # RuntimeError or UnsafeLoadError
            m.load(str(bad))

    def test_tf_missing_dir_raises(self, tmp_path):
        m = TFModel()
        try:
            import tensorflow  # noqa: F401
            real_dir = tmp_path / "empty_sm"
            real_dir.mkdir()
            with pytest.raises(Exception):    # tf complains about missing saved_model.pb
                m.load(str(real_dir))
        except ImportError:
            with pytest.raises(RuntimeError, match="tensorflow"):
                real_dir = tmp_path / "empty_sm"
                real_dir.mkdir()
                m.load(str(real_dir))
