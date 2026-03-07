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


from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.model_hash import compute_model_hash


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
    if model_path in _LOADED_MODEL_CACHE:
        _LOADED_MODEL_CACHE.move_to_end(model_path)
    _LOADED_MODEL_CACHE[model_path] = model
    while len(_LOADED_MODEL_CACHE) > _CACHE_MAX_MODELS:
        evicted_path, _ = _LOADED_MODEL_CACHE.popitem(last=False)
        _VALIDATED_MODEL_PATHS.discard(evicted_path)


def _clear_model_cache(model_path: str) -> None:
    _LOADED_MODEL_CACHE.pop(model_path, None)
    _VALIDATED_MODEL_PATHS.discard(model_path)


def get_validated_cached_model(model_path: str) -> Any | None:
    """Return cached model only if it already passed validate_onnx_model()."""
    return None


def validate_onnx_model(model_path: str) -> tuple[bool, str | None, Any | None]:
    """
    Validate ONNX model structure for upload security.
    
    SINGLE location for onnx.load() validation.
    Caches the loaded model for later analysis.
    
    Returns:
        (is_valid, error_message, loaded_model)
    """
    try:
        import onnx
        from onnx.checker import check_model
        
        model = onnx.load(model_path)
        _cache_model(model_path, model)
        
        if not model.graph or not model.graph.node:
            return False, "ONNX model contains no computational nodes", None
        
        check_model(model)
        _VALIDATED_MODEL_PATHS.add(model_path)
        return True, None, model
        
    except Exception as e:
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
    try:
        import onnx
        
        if model_obj is not None:
            model = model_obj
        elif model_path in _LOADED_MODEL_CACHE:
            model = _get_cached_model(model_path)
        else:
            model = onnx.load(model_path)
            _cache_model(model_path, model)
        
        graph = model.graph
        
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
        
        result = ModelAnalysisResult(
            success=True,
            model_path=model_path,
            file_hash=model_hash,
            operator_count=len(graph.node),
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
        
        return result
    except Exception as e:
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
