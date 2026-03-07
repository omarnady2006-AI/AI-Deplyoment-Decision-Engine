"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.decision import DeploymentDecision, make_decision
from src.core.rule import run_rules
from src.core.runtime import RuntimeName
from src.core.model_analysis import analyze_model
from src.rules.unsupported_operator import UnsupportedOperatorRule


@dataclass(frozen=True)
class RuntimeValidationResult:
    model_path: str
    runtime: RuntimeName
    predicted_decision: DeploymentDecision
    executed: bool
    success: bool
    error_message: str | None
    failing_variants: list[str] | None


def _run_with_timeout(function: Any, timeout_seconds: float) -> Any:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(function)
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _build_facts_from_analysis(analysis) -> dict[str, object]:
    facts: dict[str, object] = {}
    
    operators = analysis.operators if analysis.success else []
    op_types = set(operators)
    
    facts["model.has_non_max_suppression"] = "NonMaxSuppression" in op_types
    facts["model.has_resize"] = "Resize" in op_types
    facts["model.has_conv_transpose"] = "ConvTranspose" in op_types
    facts["model.uses_layer_normalization"] = "LayerNormalization" in op_types
    facts["model.uses_batch_normalization"] = "BatchNormalization" in op_types
    facts["operator_counts"] = analysis.operator_counts if analysis.success else {}
    facts["parameter_count"] = analysis.parameter_count if analysis.success else 0
    facts["model.has_dynamic_shapes"] = analysis.has_dynamic_shapes if analysis.success else False
    return facts


def _predict_decision(model_path: str, runtime: RuntimeName) -> DeploymentDecision:
    analysis = analyze_model(model_path)
    facts = _build_facts_from_analysis(analysis)
    diagnostics = run_rules([UnsupportedOperatorRule()], facts, runtime)
    return make_decision(diagnostics).decision


def _is_finite_outputs(outputs: object) -> tuple[bool, str | None]:
    try:
        import numpy as np
    except Exception as error:
        return False, f"missing dependency numpy: {error}"

    if isinstance(outputs, (list, tuple)):
        tensors = list(outputs)
    else:
        tensors = [outputs]

    for index, output in enumerate(tensors):
        array = np.asarray(output)
        if not np.issubdtype(array.dtype, np.number):
            return False, f"non-numeric output dtype at index {index}: {array.dtype}"
        if not np.all(np.isfinite(array)):
            return False, f"non-finite output values (NaN/Inf) at index {index}"
    return True, None


def _onnx_dtype_to_numpy(type_name: str) -> object | None:
    try:
        import numpy as np
    except Exception:
        return None

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float32,
        "tensor(double)": np.float32,
        "tensor(int8)": np.int32,
        "tensor(int16)": np.int32,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int32,
        "tensor(uint8)": np.int32,
        "tensor(uint16)": np.int32,
        "tensor(uint32)": np.int32,
        "tensor(uint64)": np.int32,
        "tensor(bool)": np.int32,
    }
    return mapping.get(type_name)


def _make_shape_with_variation(raw_shape: list[object], run_index: int) -> tuple[int, ...]:
    cycle = (1, 2, 3)
    cycle_value = cycle[run_index % len(cycle)]
    shape: list[int] = []
    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(cycle_value)
    return tuple(shape)


def _normalize_minimal_shape(raw_shape: list[object]) -> tuple[int, ...]:
    shape: list[int] = []
    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(1)
    return tuple(shape)


def _generate_domain_variants(
    input_spec: list[tuple[object, list[object], object]],
) -> list[tuple[str, list[tuple[object, tuple[int, ...], object]]]]:
    base_specs: list[tuple[object, tuple[int, ...], object, list[object]]] = []
    for key, raw_shape, dtype in input_spec:
        base_specs.append((key, _normalize_minimal_shape(raw_shape), dtype, raw_shape))

    variants: list[tuple[str, list[tuple[object, tuple[int, ...], object]]]] = []
    nominal_specs = [(key, shape, dtype) for key, shape, dtype, _ in base_specs]
    variants.append(("nominal", nominal_specs))

    def _make_variant(name: str) -> list[tuple[object, tuple[int, ...], object]]:
        result: list[tuple[object, tuple[int, ...], object]] = []
        for key, base_shape, dtype, raw_shape in base_specs:
            shape = list(base_shape)

            if name in {"small", "large", "skinny"} and len(shape) >= 2:
                h_index = len(shape) - 2
                w_index = len(shape) - 1
                if name == "small":
                    shape[h_index] = max(1, shape[h_index] // 2)
                    shape[w_index] = max(1, shape[w_index] // 2)
                elif name == "large":
                    shape[h_index] = min(1024, max(1, shape[h_index] * 2))
                    shape[w_index] = min(1024, max(1, shape[w_index] * 2))
                elif name == "skinny":
                    if shape[w_index] > 1:
                        shape[w_index] = 1
                    elif shape[h_index] > 1:
                        shape[h_index] = 1

            if name == "batch_spike" and len(shape) >= 1:
                first_raw = raw_shape[0] if raw_shape else None
                if not isinstance(first_raw, int) or first_raw <= 0 or first_raw == 1:
                    shape[0] = 4

            result.append((key, tuple(shape), dtype))
        return result

    variants.append(("small", _make_variant("small")))
    variants.append(("large", _make_variant("large")))
    variants.append(("skinny", _make_variant("skinny")))
    variants.append(("batch_spike", _make_variant("batch_spike")))
    return variants


def _extract_output_shapes(outputs: object) -> tuple[tuple[int, ...], ...]:
    try:
        import numpy as np
    except Exception:
        return tuple()

    if isinstance(outputs, (list, tuple)):
        tensors = list(outputs)
    else:
        tensors = [outputs]
    return tuple(tuple(np.asarray(output).shape) for output in tensors)


def _run_stability_loop(
    run_once: Any, probe_name: str, warmup_runs: int = 2, test_runs: int = 5
) -> tuple[bool, str | None]:
    total_runs = warmup_runs + test_runs
    timeout_seconds = 10.0
    start_time = time.monotonic()

    baseline_duration: float | None = None
    baseline_shapes: tuple[tuple[int, ...], ...] | None = None

    for run_index in range(total_runs):
        elapsed = time.monotonic() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            return False, f"{probe_name} stability probe timed out after 10 seconds"

        run_started = time.monotonic()
        try:
            outputs = _run_with_timeout(lambda: run_once(run_index), remaining)
        except concurrent.futures.TimeoutError:
            return False, f"{probe_name} stability probe timed out after 10 seconds"
        except Exception as error:
            if run_index >= warmup_runs:
                return False, f"runtime crashed after warmup: {error}"
            return False, str(error)

        run_duration = time.monotonic() - run_started
        if baseline_duration is None:
            baseline_duration = run_duration
        elif baseline_duration > 0 and run_duration > baseline_duration * 4.0:
            return False, "execution time grew more than 4x from first run"

        finite, finite_error = _is_finite_outputs(outputs)
        if not finite:
            return False, finite_error

        output_shapes = _extract_output_shapes(outputs)
        if baseline_shapes is None:
            baseline_shapes = output_shapes
        elif output_shapes != baseline_shapes:
            return False, "shape changes across runs"

    return True, None


def _run_domain_stress_validation(
    input_spec: list[tuple[object, list[object], object]],
    run_once_for_variant: Any,
    probe_name: str,
) -> tuple[bool, str | None, list[str] | None]:
    variants = _generate_domain_variants(input_spec)
    nominal_passed = False
    failing_variants: list[str] = []

    for variant_name, variant_inputs in variants:
        success, error = _run_stability_loop(
            lambda run_index: run_once_for_variant(run_index, variant_inputs),
            f"{probe_name}:{variant_name}",
            warmup_runs=1,
            test_runs=2,
        )
        if variant_name == "nominal":
            if not success:
                return False, error, ["nominal"]
            nominal_passed = True
            continue

        if nominal_passed and not success:
            failing_variants.append(variant_name)

    if failing_variants:
        return (
            False,
            "UNSTABLE_DOMAIN: fails outside nominal input domain",
            failing_variants,
        )
    return True, None, None


def _run_stability_probe(
    session_or_engine: object, runtime: RuntimeName
) -> tuple[bool, str | None, list[str] | None]:
    try:
        import numpy as np
    except Exception as error:
        return False, f"missing dependency numpy: {error}", None

    if runtime == RuntimeName.ONNXRUNTIME:
        session = session_or_engine
        input_metas = list(session.get_inputs())
        input_specs: list[tuple[str, list[object], object]] = []
        for meta in input_metas:
            dtype = _onnx_dtype_to_numpy(meta.type)
            if dtype is None:
                return False, f"unsupported ONNX input dtype: {meta.type}", None
            input_specs.append((meta.name, list(meta.shape), dtype))

        def _run_once(run_index: int, concrete_specs: list[tuple[object, tuple[int, ...], object]]) -> object:
            inputs: dict[str, object] = {}
            for input_index, (name, shape, dtype) in enumerate(concrete_specs):
                if np.issubdtype(dtype, np.floating):
                    seed = 1000 + run_index * 97 + input_index
                    rng = np.random.default_rng(seed)
                    data = rng.normal(0.0, 1e-3, size=shape).astype(np.float32)
                elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                    value = (run_index % 3) + 1
                    data = np.full(shape, value, dtype=np.int32)
                else:
                    raise RuntimeError(f"unsupported ONNX input dtype: {dtype}")
                inputs[name] = data
            return session.run(None, inputs)

        domain_success, domain_error, failing_variants = _run_domain_stress_validation(
            input_specs, _run_once, "onnxruntime"
        )
        if not domain_success:
            return False, domain_error, failing_variants

        nominal_variant = _generate_domain_variants(input_specs)[0][1]
        success, error = _run_stability_loop(
            lambda run_index: _run_once(run_index, nominal_variant), "onnxruntime"
        )
        return success, error, None

    if runtime == RuntimeName.TFLITE:
        interpreter = session_or_engine
        input_details = interpreter.get_input_details()
        input_specs: list[tuple[int, list[object], object]] = []
        for detail in input_details:
            dtype = detail.get("dtype")
            if dtype is None:
                return False, "missing TFLite input dtype", None

            if np.issubdtype(dtype, np.floating):
                tensor_dtype = np.float32
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                tensor_dtype = np.int32
            else:
                return False, f"unsupported TFLite input dtype: {dtype}", None

            shape_source = detail.get("shape_signature")
            if shape_source is None:
                shape_source = detail.get("shape")
            input_specs.append((int(detail["index"]), list(shape_source), tensor_dtype))

        output_indices = [int(detail["index"]) for detail in interpreter.get_output_details()]

        def _run_once(run_index: int, concrete_specs: list[tuple[object, tuple[int, ...], object]]) -> object:
            for input_index, (tensor_index, shape, tensor_dtype) in enumerate(concrete_specs):
                if np.issubdtype(tensor_dtype, np.floating):
                    seed = 2000 + run_index * 97 + input_index
                    rng = np.random.default_rng(seed)
                    data = rng.normal(0.0, 1e-3, size=shape).astype(np.float32)
                elif np.issubdtype(tensor_dtype, np.integer) or np.issubdtype(
                    tensor_dtype, np.bool_
                ):
                    value = (run_index % 3) + 1
                    data = np.full(shape, value, dtype=np.int32)
                else:
                    raise RuntimeError(f"unsupported TFLite input dtype: {tensor_dtype}")
                interpreter.set_tensor(tensor_index, data)
            interpreter.invoke()
            return [interpreter.get_tensor(index) for index in output_indices]

        domain_success, domain_error, failing_variants = _run_domain_stress_validation(
            input_specs, _run_once, "tflite"
        )
        if not domain_success:
            return False, domain_error, failing_variants

        nominal_variant = _generate_domain_variants(input_specs)[0][1]
        success, error = _run_stability_loop(
            lambda run_index: _run_once(run_index, nominal_variant), "tflite"
        )
        return success, error, None

    if runtime == RuntimeName.TENSORRT:
        engine = session_or_engine
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except Exception as error:
            return False, f"missing dependency for TensorRT execution: {error}", None

        try:
            context = engine.create_execution_context()
            if context is None:
                return False, "failed to create TensorRT execution context", None

            num_bindings = int(engine.num_bindings)

            input_specs: list[tuple[int, list[object], object]] = []
            for index in range(num_bindings):
                if not engine.binding_is_input(index):
                    continue
                trt_dtype = engine.get_binding_dtype(index)
                np_dtype = trt.nptype(trt_dtype)
                if np.issubdtype(np_dtype, np.floating):
                    tensor_dtype = np.float32
                elif np.issubdtype(np_dtype, np.integer) or np.issubdtype(np_dtype, np.bool_):
                    tensor_dtype = np.int32
                else:
                    return False, f"unsupported TensorRT binding dtype: {trt_dtype}", None
                input_specs.append((index, list(engine.get_binding_shape(index)), tensor_dtype))

            def _run_once(run_index: int, concrete_specs: list[tuple[object, tuple[int, ...], object]]) -> object:
                bindings: list[int] = [0] * num_bindings
                allocated_buffers: list[object] = []
                output_meta: list[tuple[tuple[int, ...], object, object]] = []

                for index in range(num_bindings):
                    raw_shape = list(engine.get_binding_shape(index))
                    matched = None
                    for input_index, input_shape, _ in concrete_specs:
                        if int(input_index) == index:
                            matched = input_shape
                            break
                    shape = matched if matched is not None else _normalize_minimal_shape(raw_shape)
                    if any(dim <= 0 for dim in shape):
                        shape = tuple(1 for _ in shape)

                    if any(dim < 0 for dim in raw_shape):
                        context.set_binding_shape(index, shape)
                        shape = tuple(context.get_binding_shape(index))

                    trt_dtype = engine.get_binding_dtype(index)
                    np_dtype = trt.nptype(trt_dtype)
                    if np.issubdtype(np_dtype, np.floating):
                        seed = 3000 + run_index * 97 + index
                        rng = np.random.default_rng(seed)
                        host = rng.normal(0.0, 1e-3, size=shape).astype(np.float32)
                        tensor_dtype = np.float32
                    elif np.issubdtype(np_dtype, np.integer) or np.issubdtype(
                        np_dtype, np.bool_
                    ):
                        value = (run_index % 3) + 1
                        host = np.full(shape, value, dtype=np.int32)
                        tensor_dtype = np.int32
                    else:
                        raise RuntimeError(
                            f"unsupported TensorRT binding dtype: {trt_dtype}"
                        )

                    device = cuda.mem_alloc(host.nbytes)
                    allocated_buffers.append(device)
                    bindings[index] = int(device)
                    cuda.memcpy_htod(device, host)

                    if not engine.binding_is_input(index):
                        output_meta.append((shape, tensor_dtype, device))

                if not bool(context.execute_v2(bindings)):
                    raise RuntimeError("tensorrt execute_v2 returned false")

                outputs = []
                for shape, dtype, device in output_meta:
                    host_output = np.empty(shape, dtype=dtype)
                    cuda.memcpy_dtoh(host_output, device)
                    outputs.append(host_output)
                return outputs

            domain_success, domain_error, failing_variants = _run_domain_stress_validation(
                input_specs, _run_once, "tensorrt"
            )
            if not domain_success:
                return False, domain_error, failing_variants

            nominal_variant = _generate_domain_variants(input_specs)[0][1]
            success, error = _run_stability_loop(
                lambda run_index: _run_once(run_index, nominal_variant), "tensorrt"
            )
            return success, error, None
        except Exception as error:
            return False, str(error), None

    return False, f"unsupported runtime: {runtime.value}", None


def _validate_onnxruntime(
    model_path: str,
) -> tuple[bool, bool, str | None, list[str] | None]:
    try:
        import onnxruntime as ort
    except Exception as error:
        return False, False, f"missing dependency onnxruntime: {error}", None

    try:
        session = ort.InferenceSession(model_path)
    except Exception as error:
        return True, False, str(error), None

    success, error, failing_variants = _run_stability_probe(
        session, RuntimeName.ONNXRUNTIME
    )
    if not success and error is not None and "unsupported ONNX input dtype" in error:
        return False, False, error, None
    return True, success, error, failing_variants


def _validate_tensorrt(
    model_path: str,
) -> tuple[bool, bool, str | None, list[str] | None]:
    try:
        import tensorrt as trt
    except Exception as error:
        return False, False, f"missing dependency tensorrt: {error}", None

    try:
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        onnx_bytes = Path(model_path).read_bytes()

        if not parser.parse(onnx_bytes):
            errors: list[str] = []
            for index in range(parser.num_errors):
                errors.append(str(parser.get_error(index)))
            parse_error = "; ".join(errors) if errors else "unknown parser error"
            return True, False, parse_error, None

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit") and hasattr(
            trt, "MemoryPoolType"
        ):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
        else:
            config.max_workspace_size = 1 << 28

        built = None
        if hasattr(builder, "build_serialized_network"):
            built = builder.build_serialized_network(network, config)
        elif hasattr(builder, "build_engine"):
            built = builder.build_engine(network, config)
        else:
            return True, False, "tensorrt build API unavailable", None

        if built is None:
            return True, False, "engine build failed", None

        if hasattr(trt, "IHostMemory") and isinstance(built, trt.IHostMemory):
            runtime_obj = trt.Runtime(logger)
            engine = runtime_obj.deserialize_cuda_engine(built)
            if engine is None:
                return True, False, "failed to deserialize TensorRT engine", None
        else:
            engine = built

        success, error, failing_variants = _run_stability_probe(
            engine, RuntimeName.TENSORRT
        )
        if (
            not success
            and error is not None
            and "unsupported TensorRT binding dtype" in error
        ):
            return False, False, error, None
        return True, success, error, failing_variants
    except Exception as error:
        return True, False, str(error), None


def _validate_tflite(
    model_path: str,
) -> tuple[bool, bool, str | None, list[str] | None]:
    interpreter_cls = None
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter

        interpreter_cls = TFLiteInterpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter as TensorFlowLiteInterpreter

            interpreter_cls = TensorFlowLiteInterpreter
        except Exception as error:
            return (
                False,
                False,
                f"missing dependency tflite_runtime/tensorflow: {error}",
                None,
            )

    try:
        interpreter = interpreter_cls(model_path=model_path)
        interpreter.allocate_tensors()
        success, error, failing_variants = _run_stability_probe(
            interpreter, RuntimeName.TFLITE
        )
        if not success and error is not None and "unsupported TFLite input dtype" in error:
            return False, False, error, None
        return True, success, error, failing_variants
    except Exception as error:
        return True, False, str(error), None


def _validate_runtime_execution(
    model_path: str, runtime: RuntimeName
) -> tuple[bool, bool, str | None, list[str] | None]:
    if runtime == RuntimeName.ONNXRUNTIME:
        return _validate_onnxruntime(model_path)
    if runtime == RuntimeName.TENSORRT:
        return _validate_tensorrt(model_path)
    if runtime == RuntimeName.TFLITE:
        return _validate_tflite(model_path)
    return False, False, f"unsupported runtime: {runtime.value}", None


def validate_runtime(model_path: str, runtime: RuntimeName) -> RuntimeValidationResult:
    try:
        predicted_decision = _predict_decision(model_path, runtime)
    except Exception:
        predicted_decision = DeploymentDecision.UNSUPPORTED

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_validate_runtime_execution, model_path, runtime)
        try:
            executed, success, error_message, failing_variants = future.result(timeout=12)
        except concurrent.futures.TimeoutError:
            executed = False
            success = False
            error_message = "validation timed out"
            failing_variants = None
    except Exception as error:
        executed = False
        success = False
        error_message = str(error)
        failing_variants = None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return RuntimeValidationResult(
        model_path=model_path,
        runtime=runtime,
        predicted_decision=predicted_decision,
        executed=executed,
        success=success,
        error_message=error_message,
        failing_variants=failing_variants,
    )
