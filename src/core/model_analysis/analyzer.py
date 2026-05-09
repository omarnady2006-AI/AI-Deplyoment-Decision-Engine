from __future__ import annotations
"""Copyright (c) 2026 Omar Nady — Deployment Decision Engine. See LICENSE."""
"""core/model_analysis/analyzer.py — analyze_model() and convenience wrappers."""
import logging
from typing import Any

from src.core.model_hash import compute_model_hash
from src.core.model_analysis.constants import _MAX_NODE_COUNT, _MAX_PARAM_COUNT, _MAX_GRAPH_DEPTH, _ONNX_BASELINE_OPS
from src.core.model_analysis.result import ModelAnalysisResult, _REQUIRED_ANALYSIS_FIELDS
from src.core.model_analysis.cache import _get_cached_model, _cache_model
from src.core.model_analysis.validator import validate_onnx_model
from src.core.model_analysis.parser import (
    _is_value_info_dynamic, _count_parameters_from_initializer,
    _count_parameters_from_constant_node, _extract_input_shapes,
    _compute_sequential_depth,
)

logger = logging.getLogger(__name__)


def analyze_model(
    model_path: str,
    model_obj: Any | None = None,
    file_hash: str | None = None,
) -> ModelAnalysisResult:
    """Analyze an ONNX model. Single canonical implementation."""
    logger.info("stage=analyze status=start path=%s", model_path)
    try:
        import onnx
        if model_obj is not None:
            model = model_obj
        else:
            _cached = _get_cached_model(model_path)
            if _cached is not None:
                model = _cached
            else:
                is_valid, reason, _ = validate_onnx_model(model_path)
                if not is_valid:
                    return ModelAnalysisResult(success=False, model_path=model_path, error=f"invalid_model: {reason}")
                _after = _get_cached_model(model_path)
                model = _after if _after is not None else onnx.load(model_path)
                _cache_model(model_path, model)

        graph = model.graph
        node_count_raw = len(graph.node)
        if node_count_raw > _MAX_NODE_COUNT:
            return ModelAnalysisResult(success=False, model_path=model_path, error=f"graph_too_large: {node_count_raw} nodes")

        operator_counts: dict[str, int] = {}
        for node in graph.node:
            op_type = node.op_type
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1
        operators = list(operator_counts.keys())
        unsupported_ops = sorted(op for op in operators if op not in _ONNX_BASELINE_OPS)

        ir_version = getattr(model, "ir_version", 0)
        opset_version = 0
        if hasattr(model, "opset_import") and model.opset_import:
            opset_version = getattr(model.opset_import[0], "version", 0)

        parameter_count = 0
        for init in graph.initializer:
            parameter_count += _count_parameters_from_initializer(init)
        for node in graph.node:
            parameter_count += _count_parameters_from_constant_node(node)
        parameter_count = min(parameter_count, _MAX_PARAM_COUNT)
        node_count_clamped = min(len(graph.node), _MAX_NODE_COUNT)

        has_dynamic_shapes = False
        for coll in ("input","output","value_info"):
            for vi in getattr(graph, coll, []) or []:
                if _is_value_info_dynamic(vi):
                    has_dynamic_shapes = True
                    break
            if has_dynamic_shapes:
                break

        input_shapes = _extract_input_shapes(graph)
        model_hash = file_hash if file_hash is not None else compute_model_hash(model_path)
        sequential_depth = _compute_sequential_depth(graph)

        if sequential_depth > _MAX_GRAPH_DEPTH:
            return ModelAnalysisResult(success=False, model_path=model_path, error=f"graph_too_deep: {sequential_depth}")

        result = ModelAnalysisResult(
            success=True, model_path=model_path, file_hash=model_hash,
            operator_count=node_count_clamped, operators=sorted(operators),
            operator_counts=operator_counts, parameter_count=parameter_count,
            has_dynamic_shapes=has_dynamic_shapes, input_count=len(graph.input),
            output_count=len(graph.output), input_shapes=input_shapes,
            unsupported_ops=unsupported_ops, sequential_depth=sequential_depth,
            ir_version=ir_version, opset_version=opset_version, graph=graph,
        )
        missing = _REQUIRED_ANALYSIS_FIELDS - vars(result).keys()
        if missing:
            raise RuntimeError(f"ModelAnalysisResult missing fields: {missing}")
        logger.info("stage=analyze status=complete path=%s operators=%d params=%d",
                    model_path, node_count_clamped, parameter_count)
        return result
    except Exception as e:
        logger.info("stage=analyze status=failed reason=%s", str(e))
        return ModelAnalysisResult(success=False, model_path=model_path, error=str(e))


# ── Convenience wrappers ──────────────────────────────────────────────────────

def detect_unsupported_ops(model_path: str) -> list[str]:
    r = analyze_model(model_path); return r.unsupported_ops if r.success else []

def detect_dynamic_shapes(model_path: str) -> bool:
    r = analyze_model(model_path); return r.has_dynamic_shapes if r.success else False

def count_parameters(model_path: str) -> int:
    r = analyze_model(model_path); return r.parameter_count if r.success else 0

def get_operator_counts(model_path: str) -> dict[str, int]:
    r = analyze_model(model_path); return r.operator_counts if r.success else {}

def get_input_shapes(model_path: str) -> dict[str, list[int | str | None]]:
    r = analyze_model(model_path); return r.input_shapes if r.success else {}

def get_sequential_depth(model_path: str) -> int:
    r = analyze_model(model_path); return r.sequential_depth if r.success else 0
