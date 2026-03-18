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
Unified Model Analysis Module

This is the SINGLE source of truth for all model analysis operations.
All model analysis logic in the repository must use this module.

Provides:
- Operator extraction
- Parameter counting (float tensors only, zero-dim handling, Constant nodes)
- Dynamic shape detection
- Input shape extraction
- Graph depth traversal (topological)
- Unsupported operator detection
"""


import logging
import multiprocessing
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.model_hash import compute_model_hash

logger = logging.getLogger(__name__)

# ── Security / resource limits ─────────────────────────────────────────────
_MAX_MODEL_FILE_BYTES  = 2 * 1024 * 1024 * 1024   # 2 GB hard ceiling
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB zip-bomb guard
_MAX_NODE_COUNT        = 1_000_000
_MAX_PARAM_COUNT       = 1_000_000_000
_MAX_GRAPH_DEPTH       = 100_000                   # PG2.2
_MAX_ZIP_FILES         = 1_000                     # PG3.2
_MAX_ZIP_RATIO         = 100                       # PG3.1 compression ratio ceiling
_MAX_PARSE_TIMEOUT_S   = 30                        # PG2.1 subprocess timeout
_MAX_PARSE_MEMORY_MB   = 1024                      # PG2.1 subprocess memory cap


_ONNX_BASELINE_OPS: frozenset[str] = frozenset({
    "Abs", "Acos", "Acosh", "Add", "And", "ArgMax", "ArgMin",
    "Asin", "Asinh", "Atan", "Atanh", "AveragePool",
    "BatchNormalization", "BitShift", "Cast", "CastLike", "Ceil",
    "Celu", "Clip", "Compress", "Concat", "ConcatFromSequence",
    "Constant", "ConstantOfShape", "Conv", "ConvInteger",
    "ConvTranspose", "Cos", "Cosh", "CumSum",
    "DepthToSpace", "DequantizeLinear", "Det", "Div", "Dropout",
    "DynamicQuantizeLinear", "Einsum", "Elu", "Equal", "Erf",
    "Exp", "Expand", "EyeLike", "Flatten", "Floor",
    "GRU", "Gather", "GatherElements", "GatherND", "Gemm",
    "GlobalAveragePool", "GlobalLpPool", "GlobalMaxPool",
    "Greater", "GreaterOrEqual", "GridSample", "GroupNormalization",
    "HardSigmoid", "Hardswish", "Identity", "If", "InstanceNormalization",
    "IsInf", "IsNaN", "LSTM", "LayerNormalization", "LeakyRelu",
    "Less", "LessOrEqual", "Log", "LogSoftmax", "Loop",
    "LpNormalization", "LpPool", "MatMul", "MatMulInteger",
    "Max", "MaxPool", "MaxUnpool", "Mean", "MeanVarianceNormalization",
    "Min", "Mod", "Mul", "Multinomial", "Neg", "NegativeLogLikelihoodLoss",
    "NonMaxSuppression", "NonZero", "Not", "OneHot", "Optional",
    "OptionalGetElement", "OptionalHasElement", "Or", "PRelu", "Pad",
    "Pow", "QLinearConv", "QLinearMatMul", "QuantizeLinear",
    "RNN", "RandomNormal", "RandomNormalLike", "RandomUniform",
    "RandomUniformLike", "Range", "Reciprocal", "ReduceL1", "ReduceL2",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
    "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
    "Relu", "Reshape", "Resize", "ReverseSequence", "RoiAlign",
    "Round", "Scan", "ScatterElements", "ScatterND", "Selu",
    "SequenceAt", "SequenceConstruct", "SequenceEmpty", "SequenceErase",
    "SequenceInsert", "SequenceLength", "Shape", "Shrink",
    "Sigmoid", "Sign", "Sin", "Sinh", "Size", "Slice", "Softmax",
    "Softplus", "Softsign", "SpaceToDepth", "Split", "SplitToSequence",
    "Sqrt", "Squeeze", "StringNormalizer", "Sub", "Sum", "Tan", "Tanh",
    "TfIdfVectorizer", "ThresholdedRelu", "Tile", "TopK", "Transpose",
    "Trilu", "Unique", "Unsqueeze", "Where", "Xor",
})

_FLOAT_DTYPES: frozenset[int] = frozenset({1, 10, 11, 16})


@dataclass
class TensorShape:
    name: str
    shape: list[int | str | None]
    dtype: str = "float32"


@dataclass
class ModelAnalysisResult:
    success: bool
    model_path: str
    file_hash: str = ""
    operator_count: int = 0
    operators: list[str] = field(default_factory=list)
    operator_counts: dict[str, int] = field(default_factory=dict)
    parameter_count: int = 0
    has_dynamic_shapes: bool = False
    input_count: int = 0
    output_count: int = 0
    input_shapes: dict[str, list[int | str | None]] = field(default_factory=dict)
    unsupported_ops: list[str] = field(default_factory=list)
    sequential_depth: int = 0
    ir_version: int = 0
    opset_version: int = 0
    error: str | None = None
    graph: Any = None


def _is_value_info_dynamic(vi: Any) -> bool:
    type_proto = getattr(vi, "type", None)
    if type_proto is None:
        return False
    tt = getattr(type_proto, "tensor_type", None)
    if tt is None:
        return False
    shape_proto = getattr(tt, "shape", None)
    if shape_proto is None:
        return True
    for dim in getattr(shape_proto, "dim", []):
        if getattr(dim, "dim_param", ""):
            return True
        try:
            if int(getattr(dim, "dim_value", 1)) <= 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _count_parameters_from_initializer(init: Any) -> int:
    if getattr(init, "data_type", -1) not in _FLOAT_DTYPES:
        return 0
    dims = list(getattr(init, "dims", []))
    if not dims:
        return 1
    if any(d <= 0 for d in dims):
        return 0
    product = 1
    for d in dims:
        product *= d
    return product


def _count_parameters_from_constant_node(node: Any) -> int:
    if getattr(node, "op_type", "") != "Constant":
        return 0
    total = 0
    for attr in getattr(node, "attribute", []):
        t = getattr(attr, "t", None)
        if t is None:
            continue
        if getattr(t, "data_type", -1) not in _FLOAT_DTYPES:
            continue
        dims = list(getattr(t, "dims", []))
        if not dims:
            total += 1
            continue
        if any(d <= 0 for d in dims):
            continue
        product = 1
        for d in dims:
            product *= d
        total += product
    return total


def _extract_input_shapes(graph: Any) -> dict[str, list[int | str | None]]:
    input_shapes: dict[str, list[int | str | None]] = {}
    for vi in getattr(graph, "input", []):
        name = getattr(vi, "name", "")
        type_proto = getattr(vi, "type", None)
        tt = getattr(type_proto, "tensor_type", None) if type_proto else None
        shape_proto = getattr(tt, "shape", None) if tt else None
        if shape_proto is None:
            input_shapes[name] = []
            continue
        shape: list[int | str | None] = []
        for dim in getattr(shape_proto, "dim", []):
            param = getattr(dim, "dim_param", "")
            if param:
                shape.append(param)
            else:
                try:
                    shape.append(int(dim.dim_value))
                except (TypeError, ValueError):
                    shape.append(None)
        input_shapes[name] = shape
    return input_shapes


def _compute_sequential_depth(graph: Any) -> int:
    output_to_node: dict[str, int] = {}
    for idx, node in enumerate(getattr(graph, "node", [])):
        for out in getattr(node, "output", []):
            if out:
                output_to_node[out] = idx
    
    graph_sources: set[str] = set()
    for vi in getattr(graph, "input", []):
        graph_sources.add(getattr(vi, "name", ""))
    for init in getattr(graph, "initializer", []):
        graph_sources.add(getattr(init, "name", ""))
    
    node_depth_cache: dict[int, int] = {}
    
    def _node_depth(idx: int, visiting: set[int]) -> int:
        if idx in node_depth_cache:
            return node_depth_cache[idx]
        if idx in visiting:
            return 0
        visiting = visiting | {idx}
        max_input_depth = 0
        nodes = list(getattr(graph, "node", []))
        for inp in getattr(nodes[idx], "input", []):
            if not inp or inp in graph_sources:
                continue
            parent_idx = output_to_node.get(inp)
            if parent_idx is None:
                continue
            d = _node_depth(parent_idx, visiting)
            if d > max_input_depth:
                max_input_depth = d
        depth = 1 + max_input_depth
        node_depth_cache[idx] = depth
        return depth
    
    sequential_depth = 0
    nodes = list(getattr(graph, "node", []))
    for idx in range(len(nodes)):
        d = _node_depth(idx, set())
        if d > sequential_depth:
            sequential_depth = d
    
    return sequential_depth


_CACHE_MAX_MODELS = 128
_LOADED_MODEL_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_VALIDATED_MODEL_PATHS: set[str] = set()


def _get_cached_model(model_path: str) -> Any | None:
    model = _LOADED_MODEL_CACHE.get(model_path)
    if model is None:
        return None
    _LOADED_MODEL_CACHE.move_to_end(model_path)
    return model


def _cache_model(model_path: str, model: Any) -> None:
    # PG6.2 – collision detection: same path but different file size → invalidate
    try:
        current_size = Path(model_path).stat().st_size
        if model_path in _MODEL_SIZE_CACHE:
            if _MODEL_SIZE_CACHE[model_path] != current_size:
                _LOADED_MODEL_CACHE.pop(model_path, None)
                _VALIDATED_MODEL_PATHS.discard(model_path)
                logger.warning(
                    "stage=cache status=invalidated reason=size_mismatch path=%s", model_path
                )
        _MODEL_SIZE_CACHE[model_path] = current_size
    except OSError as e:
        logger.warning("exception_occurred", exc_info=True)

    if model_path in _LOADED_MODEL_CACHE:
        _LOADED_MODEL_CACHE.move_to_end(model_path)
    _LOADED_MODEL_CACHE[model_path] = model
    while len(_LOADED_MODEL_CACHE) > _CACHE_MAX_MODELS:
        evicted_path, _ = _LOADED_MODEL_CACHE.popitem(last=False)
        _VALIDATED_MODEL_PATHS.discard(evicted_path)
        _MODEL_SIZE_CACHE.pop(evicted_path, None)


def _clear_model_cache(model_path: str) -> None:
    _LOADED_MODEL_CACHE.pop(model_path, None)
    _VALIDATED_MODEL_PATHS.discard(model_path)


def get_validated_cached_model(model_path: str) -> Any | None:
    """Return cached model only if it already passed validate_onnx_model()."""
    return None


# ── PG6.2: track file size alongside cached models for collision detection ──
_MODEL_SIZE_CACHE: dict[str, int] = {}


def _subprocess_validate_worker(model_path: str, result_queue: "multiprocessing.Queue") -> None:  # type: ignore[type-arg]
    """
    PG2.1 – Isolated subprocess worker: loads and checks ONNX model.
    Runs with optional memory rlimit. No return value except via queue.
    """
    try:
        try:
            import resource as _resource
            _mem = _MAX_PARSE_MEMORY_MB * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (_mem, _mem))
        except Exception as e:
            logger.warning("rlimit_unavailable_on_platform", exc_info=True)

        import onnx
        from onnx.checker import check_model

        model = onnx.load(model_path)

        # Block external data refs
        for _init in getattr(getattr(model, "graph", None), "initializer", []) or []:
            if getattr(_init, "data_location", 0) == 1:
                result_queue.put({"ok": False, "reason": "external_data_reference"})
                return

        graph = model.graph
        if not graph or not graph.node:
            result_queue.put({"ok": False, "reason": "empty_graph"})
            return

        # PG2.2 – graph complexity
        node_count = len(graph.node)
        if node_count > _MAX_NODE_COUNT:
            result_queue.put({"ok": False, "reason": f"graph_too_large: {node_count} nodes"})
            return

        # PG2.2 – graph depth limit (protects against exponential depth traversal)
        try:
            from collections import deque
            output_to_node: dict = {}
            for idx, node in enumerate(graph.node):
                for out in node.output:
                    if out:
                        output_to_node[out] = idx
            init_names = {init.name for init in graph.initializer}
            input_names = {vi.name for vi in graph.input}
            depth_cache: dict = {}

            def _depth(idx: int, stack: set) -> int:
                if idx in depth_cache:
                    return depth_cache[idx]
                if idx in stack:
                    return 0
                stack = stack | {idx}
                nd = graph.node[idx]
                d = 1 + max(
                    (_depth(output_to_node[inp], stack)
                     for inp in nd.input
                     if inp and inp not in init_names and inp not in input_names
                     and inp in output_to_node),
                    default=0,
                )
                depth_cache[idx] = d
                return d

            max_depth = 0
            for i in range(node_count):
                max_depth = max(max_depth, _depth(i, set()))
                if max_depth > _MAX_GRAPH_DEPTH:
                    result_queue.put({"ok": False, "reason": f"graph_too_deep: {max_depth}"})
                    return
        except Exception as e:
            logger.warning("depth_guard_skipped", exc_info=True)

        check_model(model)
        result_queue.put({"ok": True, "reason": None})
    except Exception as exc:
        result_queue.put({"ok": False, "reason": str(exc)})


def _run_subprocess_validation(model_path: str) -> tuple[bool, str | None]:
    """
    PG2.1 – Run ONNX validation in an isolated subprocess with hard timeout.
    PG8.1 – Always join and close to prevent zombie processes.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: "multiprocessing.Queue" = ctx.Queue()  # type: ignore[type-arg]
    p = ctx.Process(target=_subprocess_validate_worker, args=(model_path, result_queue))
    p.start()
    p.join(timeout=_MAX_PARSE_TIMEOUT_S)

    if p.is_alive():
        p.kill()
        p.join()   # PG8.1 – no zombies
        p.close()
        logger.info("stage=validate status=failed reason=parse_timeout")
        return False, "parse_timeout: validation exceeded time limit"

    p.close()  # PG8.1

    try:
        result = result_queue.get_nowait()
    except Exception:
        return False, "parse_failed: subprocess returned no result"

    if result.get("ok"):
        return True, None
    return False, result.get("reason", "parse_failed")


def _detect_real_format(path: str) -> str:
    """Detect file format from magic bytes — never trust the extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if header[:2] == b"PK":
            return "pytorch_zip"
        # ONNX / TensorFlow protobuf magic bytes
        if header[:2] in (b"\x08\x01", b"\x0a\x0d", b"\x08\x06", b"\x08\x07",
                          b"\x0a\x02", b"\x0a\x06"):
            return "onnx_or_tf"
        # Bare protobuf: first byte is typically a field tag (0x08..0x7a)
        if header and (0x08 <= header[0] <= 0x7a):
            return "onnx_or_tf"
        return "unknown"
    except Exception:
        return "unknown"


def _check_zip_bomb(path: str) -> tuple[bool, str]:
    """Return (safe, reason). Guards against zip bombs, excessive files, and path traversal."""
    try:
        with zipfile.ZipFile(path) as z:
            entries = z.infolist()

            # PG3.2 – file count limit
            if len(entries) > _MAX_ZIP_FILES:
                return False, f"zip_too_many_files: {len(entries)} > {_MAX_ZIP_FILES}"

            compressed_size = max(Path(path).stat().st_size, 1)
            total_uncompressed = 0

            for entry in entries:
                # PG3.3 – path traversal protection
                if ".." in entry.filename or entry.filename.startswith("/"):
                    return False, f"zip_path_traversal: {entry.filename!r}"

                total_uncompressed += entry.file_size

                # PG3 original – absolute size guard
                if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                    return False, f"zip_bomb: uncompressed {total_uncompressed} exceeds limit"

            # PG3.1 – compression ratio guard
            ratio = total_uncompressed / compressed_size
            if ratio > _MAX_ZIP_RATIO:
                return False, f"zip_bomb_ratio: ratio {ratio:.1f} > {_MAX_ZIP_RATIO}"

    except zipfile.BadZipFile:
        pass  # not a zip archive — this is expected for non-zip model formats
    except Exception as exc:
        return False, f"zip_read_error: {exc}"
    return True, ""


def validate_onnx_model(model_path: str) -> tuple[bool, str | None, Any | None]:
    """
    Validate ONNX model structure for upload security.

    SINGLE location for onnx.load() validation.
    Caches the loaded model for later analysis.

    Returns:
        (is_valid, error_message, loaded_model)
    """
    logger.info("stage=validate status=start path=%s", model_path)
    try:
        import onnx

        # ── 1. File-size guard (before any parsing) ───────────────────────
        try:
            file_size = Path(model_path).stat().st_size
        except OSError as exc:
            logger.info("stage=validate status=failed reason=file_stat_error")
            return False, f"Cannot stat model file: {exc}", None

        if file_size > _MAX_MODEL_FILE_BYTES:
            logger.info("stage=validate status=failed reason=file_too_large size=%d", file_size)
            return False, f"Model file too large: {file_size} bytes", None

        # ── 2. Format detection (no extension trust) ──────────────────────
        real_fmt = _detect_real_format(model_path)
        if real_fmt == "pytorch_zip":
            # Could be a PyTorch zip — reject; zip-bomb check still applies
            safe, reason = _check_zip_bomb(model_path)
            if not safe:
                logger.info("stage=validate status=failed reason=%s", reason)
                return False, reason, None
            logger.info("stage=validate status=failed reason=pytorch_format_rejected")
            return False, "PyTorch format not supported; upload an ONNX file", None
        if real_fmt == "unknown":
            logger.info("stage=validate status=failed reason=unknown_format")
            return False, "Unrecognised model format; upload a valid ONNX file", None

        # ── 3. Zip-bomb guard (ONNX .onnx files can be zip-compressed) ───
        safe, reason = _check_zip_bomb(model_path)
        if not safe:
            logger.info("stage=validate status=failed reason=%s", reason)
            return False, reason, None

        # ── 4. Subprocess-isolated parse + check (PG2.1) ──────────────────
        #    Runs onnx.load + onnx.checker in a sandboxed child process.
        #    Kills on timeout or memory breach → no zombie (PG8.1).
        parse_ok, parse_reason = _run_subprocess_validation(model_path)
        if not parse_ok:
            logger.info("stage=validate status=failed reason=%s", parse_reason)
            return False, parse_reason, None

        # ── 5. Main-process load (safe: subprocess already validated) ──────
        #    Belt-and-suspenders parse timing guard (PG4.1).
        _t0 = time.perf_counter()
        model = onnx.load(model_path)
        if (time.perf_counter() - _t0) > _MAX_PARSE_TIMEOUT_S:
            logger.info("stage=validate status=failed reason=main_parse_timeout")
            return False, "parse_timeout: main process load exceeded limit", None

        # ── 6. Block external data references ─────────────────────────────
        for initializer in getattr(getattr(model, "graph", None), "initializer", []) or []:
            if getattr(initializer, "data_location", 0) == 1:  # EXTERNAL
                logger.info("stage=validate status=failed reason=external_data_reference")
                return False, "Model references external data files; not allowed", None

        # PG6.3 – only cache validated, non-invalid results
        _cache_model(model_path, model)
        _VALIDATED_MODEL_PATHS.add(model_path)
        logger.info("stage=validate status=passed path=%s", model_path)
        return True, None, model

    except Exception as e:
        logger.info("stage=validate status=failed reason=%s", str(e))
        return False, f"Invalid ONNX model: {str(e)}", None


def analyze_model(
    model_path: str,
    model_obj: Any | None = None,
    file_hash: str | None = None,
) -> ModelAnalysisResult:
    """
    Analyze an ONNX model file and return comprehensive analysis.
    
    This is the SINGLE canonical implementation for model analysis.
    All code in the repository must use this function.
    
    Args:
        model_path: Path to the ONNX model file
        model_obj: Optional pre-loaded ONNX model to avoid redundant loading
        
    Returns:
        ModelAnalysisResult with all extracted metrics
    """
    logger.info("stage=analyze status=start path=%s", model_path)
    try:
        import onnx

        if model_obj is not None:
            model = model_obj
        elif model_path in _LOADED_MODEL_CACHE:
            model = _get_cached_model(model_path)
        else:
            # Re-run security gate before any load outside the validation path
            is_valid, reason, _ = validate_onnx_model(model_path)
            if not is_valid:
                return ModelAnalysisResult(
                    success=False,
                    model_path=model_path,
                    error=f"invalid_model: {reason}",
                )
            # Model was cached by validate_onnx_model
            model = _get_cached_model(model_path) or onnx.load(model_path)
            _cache_model(model_path, model)
        
        graph = model.graph

        # PG2.2 – enforce graph depth limit before expensive analysis
        node_count_raw = len(graph.node)
        if node_count_raw > _MAX_NODE_COUNT:
            return ModelAnalysisResult(
                success=False,
                model_path=model_path,
                error=f"graph_too_large: {node_count_raw} nodes",
            )

        operator_counts: dict[str, int] = {}
        for node in graph.node:
            op_type = node.op_type
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1
        operators: list[str] = list(operator_counts.keys())
        
        unsupported_ops: list[str] = sorted(
            op for op in operators if op not in _ONNX_BASELINE_OPS
        )
        
        ir_version: int = getattr(model, "ir_version", 0)
        opset_version: int = 0
        if hasattr(model, "opset_import") and model.opset_import:
            opset_version = getattr(model.opset_import[0], "version", 0)
        
        parameter_count: int = 0
        
        for init in graph.initializer:
            parameter_count += _count_parameters_from_initializer(init)
        
        for node in graph.node:
            parameter_count += _count_parameters_from_constant_node(node)
        
        # Clamp to prevent integer overflow in downstream scoring (Patch 7.1)
        parameter_count = min(parameter_count, _MAX_PARAM_COUNT)
        node_count_clamped = min(len(graph.node), _MAX_NODE_COUNT)
        
        has_dynamic_shapes: bool = False
        for collection_name in ("input", "output", "value_info"):
            for vi in getattr(graph, collection_name, []) or []:
                if _is_value_info_dynamic(vi):
                    has_dynamic_shapes = True
                    break
            if has_dynamic_shapes:
                break
        
        input_shapes = _extract_input_shapes(graph)
        
        model_hash = file_hash if file_hash is not None else compute_model_hash(model_path)
        
        sequential_depth = _compute_sequential_depth(graph)

        # PG2.2 – hard depth limit
        if sequential_depth > _MAX_GRAPH_DEPTH:
            return ModelAnalysisResult(
                success=False,
                model_path=model_path,
                error=f"graph_too_deep: {sequential_depth}",
            )

        result = ModelAnalysisResult(
            success=True,
            model_path=model_path,
            file_hash=model_hash,
            operator_count=node_count_clamped,
            operators=sorted(operators),
            operator_counts=operator_counts,
            parameter_count=parameter_count,
            has_dynamic_shapes=has_dynamic_shapes,
            input_count=len(graph.input),
            output_count=len(graph.output),
            input_shapes=input_shapes,
            unsupported_ops=unsupported_ops,
            sequential_depth=sequential_depth,
            ir_version=ir_version,
            opset_version=opset_version,
            graph=graph,
        )
        
        REQUIRED_FIELDS = {
            "model_path", "file_hash", "operator_count", "operators",
            "operator_counts", "parameter_count", "has_dynamic_shapes",
            "input_shapes", "unsupported_ops", "sequential_depth",
            "input_count", "output_count", "success"
        }
        result_dict = {
            "model_path": result.model_path,
            "file_hash": result.file_hash,
            "operator_count": result.operator_count,
            "operators": result.operators,
            "operator_counts": result.operator_counts,
            "parameter_count": result.parameter_count,
            "has_dynamic_shapes": result.has_dynamic_shapes,
            "input_shapes": result.input_shapes,
            "unsupported_ops": result.unsupported_ops,
            "sequential_depth": result.sequential_depth,
            "input_count": result.input_count,
            "output_count": result.output_count,
            "success": result.success,
        }
        missing = REQUIRED_FIELDS - set(result_dict.keys())
        if missing:
            raise RuntimeError(f"ModelAnalysisResult missing required fields: {missing}")
        
        logger.info("stage=analyze status=complete path=%s operators=%d params=%d",
                    model_path, node_count_clamped, parameter_count)
        return result
    except Exception as e:
        logger.info("stage=analyze status=failed reason=%s", str(e))
        return ModelAnalysisResult(
            success=False,
            model_path=model_path,
            error=str(e),
        )


def detect_unsupported_ops(model_path: str) -> list[str]:
    """Detect unsupported operators in a model."""
    result = analyze_model(model_path)
    return result.unsupported_ops if result.success else []


def detect_dynamic_shapes(model_path: str) -> bool:
    """Check if model has dynamic shapes."""
    result = analyze_model(model_path)
    return result.has_dynamic_shapes if result.success else False


def count_parameters(model_path: str) -> int:
    """Count float parameters in a model."""
    result = analyze_model(model_path)
    return result.parameter_count if result.success else 0


def get_operator_counts(model_path: str) -> dict[str, int]:
    """Get operator counts for a model."""
    result = analyze_model(model_path)
    return result.operator_counts if result.success else {}


def get_input_shapes(model_path: str) -> dict[str, list[int | str | None]]:
    """Get input shapes for a model."""
    result = analyze_model(model_path)
    return result.input_shapes if result.success else {}


def get_sequential_depth(model_path: str) -> int:
    """Get sequential depth of model graph."""
    result = analyze_model(model_path)
    return result.sequential_depth if result.success else 0


def build_adversarial_test_models(output_dir: str) -> dict[str, str]:
    """
    Build minimal ONNX graphs that isolate specific signals for causality validation.

    Returns a mapping of case name -> absolute model path.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}

    def _save_model(name: str, model: Any) -> str:
        path = out_dir / f"{name}.onnx"
        onnx.save(model, str(path))
        created[name] = str(path.resolve())
        return str(path.resolve())

    # Case A: huge parameters, few ops
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2048])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2048])
    w_arr = np.ones((2048, 2048), dtype=np.float32)
    w = numpy_helper.from_array(w_arr, name="W")
    node = helper.make_node("MatMul", ["x", "W"], ["y"])
    graph = helper.make_graph([node], "case_a_huge_params_few_ops", [x], [y], initializer=[w])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("case_a_huge_params_few_ops", model)

    # Case B: many ops, tiny parameters (deep ReLU chain, no initializers)
    chain_len = 220
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info(f"relu_{chain_len}", TensorProto.FLOAT, [1, 4])
    nodes = []
    prev = "x"
    for i in range(1, chain_len + 1):
        cur = f"relu_{i}"
        nodes.append(helper.make_node("Relu", [prev], [cur]))
        prev = cur
    graph = helper.make_graph(nodes, "case_b_many_ops_tiny_params", [x], [y])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("case_b_many_ops_tiny_params", model)

    # Case C: deep graph, low compute (Identity chain)
    depth_len = 260
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info(f"id_{depth_len}", TensorProto.FLOAT, [1, 2])
    nodes = []
    prev = "x"
    for i in range(1, depth_len + 1):
        cur = f"id_{i}"
        nodes.append(helper.make_node("Identity", [prev], [cur]))
        prev = cur
    graph = helper.make_graph(nodes, "case_c_deep_low_compute", [x], [y])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("case_c_deep_low_compute", model)

    # Case D: unsupported operator only
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("CustomUnsupportedOp", ["x"], ["y"])
    graph = helper.make_graph([node], "case_d_unsupported_only", [x], [y])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("case_d_unsupported_only", model)

    # Tiny synthetic graph
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    bias = numpy_helper.from_array(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), name="bias")
    node = helper.make_node("Add", ["x", "bias"], ["y"])
    graph = helper.make_graph([node], "tiny_synthetic", [x], [y], initializer=[bias])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("tiny_synthetic", model)

    # Empty graph (for realism checks)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([], "empty_graph", [x], [y])
    model = helper.make_model(graph, producer_name="causality_validation")
    _save_model("empty_graph", model)

    return created
