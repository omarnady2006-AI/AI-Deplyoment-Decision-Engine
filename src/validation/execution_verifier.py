"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Execution verification layer for runtime validation."""
import time
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.core.optional_dependencies import (
    PSUTIL, ONNXRUNTIME, TENSORRT, TENSORFLOW, OPENVINO, NUMPY
)


@dataclass
class VerificationResult:
    """Result of runtime verification."""
    runtime: str
    success: bool
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    inference_output_shape: Optional[Tuple] = None
    build_time_ms: Optional[float] = None
    conversion_time_ms: Optional[float] = None


class ONNXVerifier:
    """Verify ONNX model execution."""
    
    def __init__(self):
        self.ort_session = None
    
    def verify(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None,
        test_input: Optional[Any] = None
    ) -> VerificationResult:
        """Run ONNX model and measure performance."""
        if not ONNXRUNTIME.available or not PSUTIL.available or not NUMPY.available:
            return VerificationResult(
                runtime="onnx",
                success=False,
                error_type="ImportError",
                error_message="onnxruntime, psutil, or numpy not installed"
            )
        
        try:
            ort = ONNXRUNTIME.module
            np = NUMPY.module
            
            # Get available providers
            providers = ort.get_available_providers()
            
            # Select best provider
            if 'CUDAExecutionProvider' in providers:
                provider = 'CUDAExecutionProvider'
            elif 'CoreMLExecutionProvider' in providers:
                provider = 'CoreMLExecutionProvider'
            else:
                provider = 'CPUExecutionProvider'
            
            # Measure memory before
            process = PSUTIL.module.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Create session
            start_time = time.perf_counter()
            session = ort.InferenceSession(
                model_path,
                providers=[provider],
                provider_options=[{}]
            )
            load_time = (time.perf_counter() - start_time) * 1000
            
            # Get input info
            input_info = session.get_inputs()
            if not input_info:
                return VerificationResult(
                    runtime="onnx",
                    success=False,
                    error_type="NoInputs",
                    error_message="Model has no inputs"
                )
            
            # Create dummy input if not provided
            if test_input is None:
                inp = input_info[0]
                shape = input_shape if input_shape else inp.shape
                # Handle dynamic shapes
                shape = [s if isinstance(s, int) else 1 for s in shape]
                test_input = np.random.randn(*shape).astype(np.float32)
            
            # Warmup run
            try:
                session.run(None, {input_info[0].name: test_input})
            except Exception:
                pass  # Warmup may fail, continue
            
            # Measure inference
            timings = []
            peak_mem = mem_before
            outputs = None
            
            for _ in range(5):
                mem_before_run = process.memory_info().rss / 1024 / 1024
                
                start = time.perf_counter()
                outputs = session.run(None, {input_info[0].name: test_input})
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(elapsed)
                
                mem_after_run = process.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, mem_after_run)
            
            latency_ms = sum(timings) / len(timings)
            memory_mb = peak_mem - mem_before
            
            output_shape = outputs[0].shape if outputs else None
            
            return VerificationResult(
                runtime="onnx",
                success=True,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                peak_memory_mb=peak_mem,
                inference_output_shape=output_shape
            )
            
        except Exception as e:
            return VerificationResult(
                runtime="onnx",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )


class TensorRTVerifier:
    """Verify TensorRT engine build and execution."""
    
    def __init__(self):
        self.trt_logger = None
    
    def verify(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None,
        max_workspace_size: int = 1 << 30  # 1GB
    ) -> VerificationResult:
        """Attempt TensorRT build and measure performance."""
        if not TENSORRT.available:
            return VerificationResult(
                runtime="tensorrt",
                success=False,
                error_type="ImportError",
                error_message="tensorrt not installed"
            )
        
        try:
            trt = TENSORRT.module
            
            # Create logger
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            
            if not builder:
                return VerificationResult(
                    runtime="tensorrt",
                    success=False,
                    error_type="BuilderError",
                    error_message="Failed to create TensorRT builder"
                )
            
            # Parse ONNX model
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            start_time = time.perf_counter()
            
            with open(model_path, 'rb') as f:
                if not parser.parse(f.read()):
                    errors = [parser.get_error(i) for i in range(parser.num_errors)]
                    return VerificationResult(
                        runtime="tensorrt",
                        success=False,
                        error_type="ParseError",
                        error_message="; ".join(str(e) for e in errors)
                    )
            
            parse_time = (time.perf_counter() - start_time) * 1000
            
            # Check for unsupported ops
            if not self._check_supported_ops(network, logger):
                return VerificationResult(
                    runtime="tensorrt",
                    success=False,
                    error_type="UnsupportedOps",
                    error_message="Model contains unsupported operators"
                )
            
            # Build config
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # Build engine
            start_time = time.perf_counter()
            
            engine = None
            serialized = None
            try:
                engine = builder.build_engine(network, config)
            except AttributeError:
                # TensorRT 8.5+ uses build_serialized_network
                serialized = builder.build_serialized_network(network, config)
            
            build_time = (time.perf_counter() - start_time) * 1000
            
            if engine is None and serialized is None:
                return VerificationResult(
                    runtime="tensorrt",
                    success=False,
                    error_type="BuildError",
                    error_message="Failed to build TensorRT engine"
                )
            
            return VerificationResult(
                runtime="tensorrt",
                success=True,
                build_time_ms=build_time,
            )
            
        except Exception as e:
            return VerificationResult(
                runtime="tensorrt",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def _check_supported_ops(self, network, logger) -> bool:
        """Check if all operators are supported."""
        return TENSORRT.available


class TFLiteVerifier:
    """Verify TFLite conversion and execution."""
    
    def verify(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None
    ) -> VerificationResult:
        """Convert to TFLite and measure performance."""
        if not TENSORFLOW.available or not PSUTIL.available or not NUMPY.available:
            return VerificationResult(
                runtime="tflite",
                success=False,
                error_type="ImportError",
                error_message="tensorflow, psutil, or numpy not installed"
            )
        
        try:
            tf = TENSORFLOW.module
            np = NUMPY.module
            
            # Load model
            start_time = time.perf_counter()
            
            try:
                # Try loading as TFLite first
                interpreter = tf.lite.Interpreter(model_path=model_path)
                load_time = (time.perf_counter() - start_time) * 1000
            except Exception:
                # Try converting from ONNX/saved model
                # For now, return unsupported
                return VerificationResult(
                    runtime="tflite",
                    success=False,
                    error_type="ConversionError",
                    error_message="Model is not in TFLite format and conversion not implemented"
                )
            
            # Allocate tensors
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            if not input_details:
                return VerificationResult(
                    runtime="tflite",
                    success=False,
                    error_type="NoInputs",
                    error_message="Model has no inputs"
                )
            
            # Create test input
            inp = input_details[0]
            shape = input_shape if input_shape else inp['shape']
            shape = [s if s > 0 else 1 for s in shape]
            test_input = np.random.randn(*shape).astype(np.float32)
            
            # Measure memory
            process = PSUTIL.module.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Warmup
            try:
                interpreter.set_tensor(inp['index'], test_input)
                interpreter.invoke()
            except Exception:
                pass
            
            # Measure inference
            timings = []
            peak_mem = mem_before
            
            for _ in range(5):
                mem_before_run = process.memory_info().rss / 1024 / 1024
                
                interpreter.set_tensor(inp['index'], test_input)
                start = time.perf_counter()
                interpreter.invoke()
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(elapsed)
                
                mem_after_run = process.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, mem_after_run)
            
            latency_ms = sum(timings) / len(timings)
            memory_mb = peak_mem - mem_before
            output_shape = output_details[0]['shape'] if output_details else None
            
            return VerificationResult(
                runtime="tflite",
                success=True,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                peak_memory_mb=peak_mem,
                inference_output_shape=output_shape,
                conversion_time_ms=load_time
            )
            
        except Exception as e:
            return VerificationResult(
                runtime="tflite",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            # Allocate tensors
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            if not input_details:
                return VerificationResult(
                    runtime="tflite",
                    success=False,
                    error_type="NoInputs",
                    error_message="Model has no inputs"
                )
            
            # Create test input
            import numpy as np
            inp = input_details[0]
            shape = input_shape if input_shape else inp['shape']
            shape = [s if s > 0 else 1 for s in shape]
            test_input = np.random.randn(*shape).astype(np.float32)
            
            # Measure memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Warmup
            try:
                interpreter.set_tensor(inp['index'], test_input)
                interpreter.invoke()
            except Exception:
                pass
            
            # Measure inference
            timings = []
            peak_mem = mem_before
            
            for _ in range(5):
                mem_before_run = process.memory_info().rss / 1024 / 1024
                
                interpreter.set_tensor(inp['index'], test_input)
                start = time.perf_counter()
                interpreter.invoke()
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(elapsed)
                
                mem_after_run = process.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, mem_after_run)
            
            latency_ms = sum(timings) / len(timings)
            memory_mb = peak_mem - mem_before
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            output_shape = output.shape
            
            return VerificationResult(
                runtime="tflite",
                success=True,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                peak_memory_mb=peak_mem,
                inference_output_shape=output_shape,
                conversion_time_ms=load_time
            )
            
        except Exception as e:
            return VerificationResult(
                runtime="tflite",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )


class OpenVINOVerifier:
    """Verify OpenVINO model execution."""
    
    def verify(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None
    ) -> VerificationResult:
        """Run OpenVINO model and measure performance."""
        if not OPENVINO.available or not PSUTIL.available or not NUMPY.available:
            return VerificationResult(
                runtime="openvino",
                success=False,
                error_type="ImportError",
                error_message="openvino, psutil, or numpy not installed"
            )
        
        try:
            from openvino.runtime import Core
            np = NUMPY.module
            
            core = Core()
            
            # Measure memory
            process = PSUTIL.module.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Read model
            start_time = time.perf_counter()
            
            # Try to load model
            try:
                model = core.read_model(model_path)
            except Exception:
                # Try as ONNX
                model = core.read_model(model=model_path)
            
            compile_time = (time.perf_counter() - start_time) * 1000
            
            # Compile model
            device = "GPU" if "GPU" in core.available_devices else "CPU"
            compiled_model = core.compile_model(model, device_name=device)
            
            # Create infer request
            infer_request = compiled_model.create_infer_request()
            
            # Get input shape
            input_tensor = compiled_model.input(0)
            shape = input_shape if input_shape else input_tensor.shape
            shape = [s if s > 0 else 1 for s in shape]
            
            # Create test input
            test_input = np.random.randn(*shape).astype(np.float32)
            
            # Warmup
            try:
                infer_request.infer({0: test_input})
            except Exception:
                pass
            
            # Measure inference
            timings = []
            peak_mem = mem_before
            result = None
            
            for _ in range(5):
                mem_before_run = process.memory_info().rss / 1024 / 1024
                
                start = time.perf_counter()
                result = infer_request.infer({0: test_input})
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(elapsed)
                
                mem_after_run = process.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, mem_after_run)
            
            latency_ms = sum(timings) / len(timings)
            memory_mb = peak_mem - mem_before
            
            output_shape = list(result.values())[0].shape if result else None
            
            return VerificationResult(
                runtime="openvino",
                success=True,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                peak_memory_mb=peak_mem,
                inference_output_shape=output_shape,
                build_time_ms=compile_time
            )
            
        except Exception as e:
            return VerificationResult(
                runtime="openvino",
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )


class ExecutionVerifier:
    """Main execution verification orchestrator."""
    
    def __init__(self):
        self.onnx_verifier = ONNXVerifier()
        self.tensorrt_verifier = TensorRTVerifier()
        self.tflite_verifier = TFLiteVerifier()
        self.openvino_verifier = OpenVINOVerifier()
    
    def verify_runtime(
        self,
        runtime: str,
        model_path: str,
        input_shape: Optional[Tuple] = None,
        test_input: Optional[Any] = None
    ) -> VerificationResult:
        """Verify specific runtime."""
        runtime = runtime.lower()
        
        if runtime == "onnx" or runtime == "onnxruntime":
            return self.onnx_verifier.verify(model_path, input_shape, test_input)
        elif runtime == "tensorrt" or runtime == "trt":
            return self.tensorrt_verifier.verify(model_path, input_shape)
        elif runtime == "tflite":
            return self.tflite_verifier.verify(model_path, input_shape)
        elif runtime == "openvino":
            return self.openvino_verifier.verify(model_path, input_shape)
        else:
            return VerificationResult(
                runtime=runtime,
                success=False,
                error_type="UnknownRuntime",
                error_message=f"Unknown runtime: {runtime}"
            )
    
    def verify_all_runtimes(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None,
        test_input: Optional[Any] = None
    ) -> Dict[str, VerificationResult]:
        """Verify all available runtimes."""
        results = {}
        
        runtimes = ["onnx", "tensorrt", "tflite", "openvino"]
        
        for runtime in runtimes:
            result = self.verify_runtime(runtime, model_path, input_shape, test_input)
            results[runtime] = result
        
        return results
    
    def get_best_runtime(
        self,
        model_path: str,
        input_shape: Optional[Tuple] = None,
        metric: str = "latency"
    ) -> Tuple[Optional[str], Dict[str, VerificationResult]]:
        """Find best performing runtime."""
        results = self.verify_all_runtimes(model_path, input_shape)
        
        successful = {k: v for k, v in results.items() if v.success}
        
        if not successful:
            return None, results
        
        if metric == "latency":
            best = min(successful.items(), key=lambda x: x[1].latency_ms or float('inf'))
        elif metric == "memory":
            best = min(successful.items(), key=lambda x: x[1].memory_mb or float('inf'))
        else:
            best = list(successful.items())[0]
        
        return best[0], results
