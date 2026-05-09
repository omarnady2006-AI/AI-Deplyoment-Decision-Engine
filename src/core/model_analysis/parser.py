from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady — Deployment Decision Engine.
See LICENSE in the project root for full terms.
"""
"""core/model_analysis/parser.py — pure ONNX graph parsing functions."""
from typing import Any
from src.core.model_analysis.constants import _FLOAT_DTYPES


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
    nodes = list(getattr(graph, "node", []))
    node_depth_cache: dict[int, int] = {}
    _visiting: set[int] = set()

    def _node_depth(idx: int) -> int:
        if idx in node_depth_cache:
            return node_depth_cache[idx]
        if idx in _visiting:
            return 0
        _visiting.add(idx)
        max_input_depth = 0
        for inp in getattr(nodes[idx], "input", []):
            if not inp or inp in graph_sources:
                continue
            parent_idx = output_to_node.get(inp)
            if parent_idx is None:
                continue
            d = _node_depth(parent_idx)
            if d > max_input_depth:
                max_input_depth = d
        depth = 1 + max_input_depth
        node_depth_cache[idx] = depth
        _visiting.discard(idx)
        return depth

    sequential_depth = 0
    for idx in range(len(nodes)):
        d = _node_depth(idx)
        if d > sequential_depth:
            sequential_depth = d
    return sequential_depth
