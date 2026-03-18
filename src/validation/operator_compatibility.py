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
Operator compatibility verification before deployment.

This module uses the canonical analyze_model() from src.core.model_analysis.
It does NOT load ONNX models directly.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass

from src.core.model_analysis import analyze_model, ModelAnalysisResult
from src.core.optional_dependencies import ONNXRUNTIME, TENSORRT, TENSORFLOW, OPENVINO


logger = logging.getLogger(__name__)


@dataclass
class CompatibilityIssue:
    severity: str
    category: str
    operator: Optional[str]
    message: str
    suggestion: Optional[str]


@dataclass
class CompatibilityResult:
    runtime: str
    compatible: bool
    issues: List[CompatibilityIssue]
    supported_ops: Set[str]
    unsupported_ops: Set[str]
    can_compile: bool
    can_infer: bool


class ONNXCompatibilityChecker:
    def __init__(self):
        self.supported_onnx_ops = self._get_supported_onnx_ops()
    
    def check_compatibility(
        self,
        model_path: str,
        runtime: str = "onnxruntime"
    ) -> CompatibilityResult:
        runtime = runtime.lower()
        
        analysis = analyze_model(model_path)
        
        if not analysis.success:
            return CompatibilityResult(
                runtime=runtime,
                compatible=False,
                issues=[CompatibilityIssue(
                    severity="error",
                    category="load_error",
                    operator=None,
                    message=f"Failed to analyze model: {analysis.error}",
                    suggestion="Verify model file is valid ONNX format"
                )],
                supported_ops=set(),
                unsupported_ops=set(),
                can_compile=False,
                can_infer=False
            )
        
        if runtime in ["onnx", "onnxruntime"]:
            return self._check_onnxruntime_compatibility(analysis, model_path)
        elif runtime in ["tensorrt", "trt"]:
            return self._check_tensorrt_compatibility(analysis, model_path)
        elif runtime == "openvino":
            return self._check_openvino_compatibility(analysis, model_path)
        elif runtime == "tflite":
            return self._check_tflite_compatibility(analysis, model_path)
        else:
            return CompatibilityResult(
                runtime=runtime,
                compatible=False,
                issues=[CompatibilityIssue(
                    severity="error",
                    category="unknown_runtime",
                    operator=None,
                    message=f"Unknown runtime: {runtime}",
                    suggestion="Use onnxruntime, tensorrt, openvino, or tflite"
                )],
                supported_ops=set(),
                unsupported_ops=set(),
                can_compile=False,
                can_infer=False
            )
    
    def _check_onnxruntime_compatibility(
        self,
        analysis: ModelAnalysisResult,
        model_path: str
    ) -> CompatibilityResult:
        issues: List[CompatibilityIssue] = []
        supported_ops: Set[str] = set()
        unsupported_ops: Set[str] = set()
        
        if analysis.ir_version < 7:
            issues.append(CompatibilityIssue(
                severity="warning",
                category="version",
                operator=None,
                message=f"Model IR version {analysis.ir_version} is old",
                suggestion="Consider exporting with newer ONNX opset"
            ))
        
        if analysis.opset_version < 11:
            issues.append(CompatibilityIssue(
                severity="warning",
                category="opset",
                operator=None,
                message=f"Opset version {analysis.opset_version} may have limited support",
                suggestion="Use opset 11 or higher for best compatibility"
            ))
        
        for op_type in analysis.operators:
            if op_type in self.supported_onnx_ops:
                supported_ops.add(op_type)
            else:
                unsupported_ops.add(op_type)
                issues.append(CompatibilityIssue(
                    severity="warning",
                    category="unsupported_op",
                    operator=op_type,
                    message=f"Operator {op_type} may not be supported",
                    suggestion=f"Check if {op_type} has a custom implementation"
                ))
        
        if analysis.has_dynamic_shapes:
            issues.append(CompatibilityIssue(
                severity="info",
                category="dynamic_shape",
                operator=None,
                message="Model has dynamic shapes",
                suggestion="Dynamic shapes are supported but may impact performance"
            ))
        
        can_infer = False
        can_compile = False
        if ONNXRUNTIME.available:
            try:
                ort = ONNXRUNTIME.module
                sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                can_infer = True
                can_compile = True
            except Exception as e:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="session_error",
                    operator=None,
                    message=f"Failed to create inference session: {str(e)}",
                    suggestion="Check model validity and operator support"
                ))
        else:
            issues.append(CompatibilityIssue(
                severity="error",
                category="import_error",
                operator=None,
                message="onnxruntime not installed",
                suggestion="Install onnxruntime package"
            ))
        
        compatible = can_infer and len(unsupported_ops) == 0
        
        return CompatibilityResult(
            runtime="onnxruntime",
            compatible=compatible,
            issues=issues,
            supported_ops=supported_ops,
            unsupported_ops=unsupported_ops,
            can_compile=can_compile,
            can_infer=can_infer
        )
    
    def _check_tensorrt_compatibility(
        self,
        analysis: ModelAnalysisResult,
        model_path: str
    ) -> CompatibilityResult:
        issues: List[CompatibilityIssue] = []
        supported_ops: Set[str] = set()
        unsupported_ops: Set[str] = set()
        
        trt_supported = {
            'Conv', 'Relu', 'MaxPool', 'AveragePool', 'Add', 'Mul', 'Concat',
            'Gemm', 'BatchNormalization', 'Dropout', 'Flatten', 'Reshape',
            'Transpose', 'Sigmoid', 'Tanh', 'Softmax', 'LRN', 'Split', 'Clip',
            'LeakyRelu', 'Selu', 'Elu', 'PRelu', 'Identity', 'MatMul', 'Cast',
            'Squeeze', 'Unsqueeze', 'Pad', 'Slice', 'Resize', 'ReduceMean',
            'ReduceSum', 'ReduceMax', 'ReduceMin', 'GlobalAveragePool',
            'GlobalMaxPool', 'ConvTranspose', 'HardSigmoid', 'Gather', 'Scatter'
        }
        
        for op_type in analysis.operators:
            if op_type in trt_supported:
                supported_ops.add(op_type)
            else:
                unsupported_ops.add(op_type)
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="unsupported_op",
                    operator=op_type,
                    message=f"TensorRT does not support {op_type}",
                    suggestion=f"Replace {op_type} with supported equivalent or use plugin"
                ))
        
        if analysis.has_dynamic_shapes:
            issues.append(CompatibilityIssue(
                severity="error",
                category="dynamic_shape",
                operator=None,
                message="TensorRT has limited dynamic shape support",
                suggestion="Fix input shapes or use shape tensors"
            ))
        
        can_compile = False
        can_infer = False
        if TENSORRT.available:
            try:
                trt = TENSORRT.module
                
                logger_obj = trt.Logger(trt.Logger.WARNING)
                builder = trt.Builder(logger_obj)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, logger_obj)
                
                with open(model_path, 'rb') as f:
                    can_parse = parser.parse(f.read())
                
                if can_parse:
                    can_compile = True
                    can_infer = True
                else:
                    errors = [parser.get_error(i) for i in range(parser.num_errors)]
                    for err in errors[:3]:
                        issues.append(CompatibilityIssue(
                            severity="error",
                            category="parse_error",
                            operator=None,
                            message=f"TensorRT parse error: {err}",
                            suggestion="Fix unsupported operators or shapes"
                        ))
            except Exception as e:
                logger.exception("TensorRT parsing failed")
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="parse_error",
                    operator=None,
                    message=f"TensorRT parsing failed: {str(e)}",
                    suggestion="Check model compatibility"
                ))
        else:
            issues.append(CompatibilityIssue(
                severity="error",
                category="import_error",
                operator=None,
                message="TensorRT not installed",
                suggestion="Install tensorrt package"
            ))
        
        compatible = can_compile and len(unsupported_ops) == 0
        
        return CompatibilityResult(
            runtime="tensorrt",
            compatible=compatible,
            issues=issues,
            supported_ops=supported_ops,
            unsupported_ops=unsupported_ops,
            can_compile=can_compile,
            can_infer=can_infer
        )
    
    def _check_openvino_compatibility(
        self,
        analysis: ModelAnalysisResult,
        model_path: str
    ) -> CompatibilityResult:
        issues: List[CompatibilityIssue] = []
        supported_ops: Set[str] = set()
        unsupported_ops: Set[str] = set()
        
        supported_ops.update(analysis.operators)
        
        can_compile = False
        can_infer = False
        if OPENVINO.available:
            try:
                from openvino.runtime import Core
                core = Core()
                
                try:
                    core_model = core.read_model(model=model_path)
                    can_compile = True
                    can_infer = True
                except Exception as e:
                    issues.append(CompatibilityIssue(
                        severity="error",
                        category="load_error",
                        operator=None,
                        message=f"OpenVINO failed to load model: {str(e)}",
                        suggestion="Check operator compatibility"
                    ))
            except Exception as e:
                logger.exception("OpenVINO error")
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="load_error",
                    operator=None,
                    message=f"OpenVINO error: {str(e)}",
                    suggestion="Check installation and model"
                ))
        else:
            issues.append(CompatibilityIssue(
                severity="error",
                category="import_error",
                operator=None,
                message="OpenVINO not installed",
                suggestion="Install openvino package"
            ))
        
        compatible = can_compile
        
        return CompatibilityResult(
            runtime="openvino",
            compatible=compatible,
            issues=issues,
            supported_ops=supported_ops,
            unsupported_ops=unsupported_ops,
            can_compile=can_compile,
            can_infer=can_infer
        )
    
    def _check_tflite_compatibility(
        self,
        analysis: ModelAnalysisResult,
        model_path: str
    ) -> CompatibilityResult:
        issues: List[CompatibilityIssue] = []
        supported_ops: Set[str] = set()
        unsupported_ops: Set[str] = set()
        
        tflite_supported = {
            'Conv', 'Relu', 'MaxPool', 'AveragePool', 'Add', 'Mul', 'Concat',
            'Gemm', 'Reshape', 'Transpose', 'Sigmoid', 'Tanh', 'Softmax',
            'Split', 'Resize', 'ReduceMean', 'ReduceSum', 'DepthToSpace',
            'SpaceToDepth', 'Pad', 'Slice', 'StridedSlice', 'Pack', 'Unpack',
            'Gather', 'ScatterNd', 'OneHot', 'Log', 'Exp', 'Pow', 'Sqrt',
            'Rsqrt', 'Neg', 'Abs', 'Floor', 'Ceil', 'Round', 'Clip'
        }
        
        for op_type in analysis.operators:
            if op_type in tflite_supported:
                supported_ops.add(op_type)
            else:
                unsupported_ops.add(op_type)
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="unsupported_op",
                    operator=op_type,
                    message=f"TFLite does not support {op_type}",
                    suggestion=f"Replace {op_type} or implement custom op"
                ))
        
        can_compile = len(unsupported_ops) == 0
        can_infer = can_compile
        
        compatible = can_compile and len(unsupported_ops) == 0
        
        return CompatibilityResult(
            runtime="tflite",
            compatible=compatible,
            issues=issues,
            supported_ops=supported_ops,
            unsupported_ops=unsupported_ops,
            can_compile=can_compile,
            can_infer=can_infer
        )
    
    def _get_supported_onnx_ops(self) -> Set[str]:
        return {
            'Abs', 'Acos', 'Acosh', 'Add', 'And', 'ArgMax', 'ArgMin', 'Asin',
            'Asinh', 'Atan', 'Atanh', 'AveragePool', 'BatchNormalization',
            'Bernoulli', 'BitShift', 'Cast', 'CastLike', 'Ceil', 'Celu',
            'Clip', 'Compress', 'Concat', 'ConcatFromSequence', 'Constant',
            'ConstantOfShape', 'Conv', 'ConvInteger', 'ConvTranspose',
            'Cos', 'Cosh', 'CumSum', 'DepthToSpace', 'DequantizeLinear',
            'Det', 'Div', 'Dropout', 'DynamicQuantizeLinear', 'Einsum',
            'Elu', 'Equal', 'Erf', 'Exp', 'Expand', 'EyeLike', 'Flatten',
            'Floor', 'GRU', 'Gather', 'GatherElements', 'GatherND', 'Gemm',
            'GlobalAveragePool', 'GlobalLpPool', 'GlobalMaxPool', 'Gradient',
            'Greater', 'GreaterOrEqual', 'GridSample', 'HardSigmoid',
            'Hardmax', 'Identity', 'If', 'InstanceNormalization', 'IsInf',
            'IsNaN', 'LRN', 'LSTM', 'LayerNormalization', 'LeakyRelu', 'Less',
            'LessOrEqual', 'Log', 'LogSoftmax', 'Loop', 'MatMul', 'MatMulInteger',
            'Max', 'MaxPool', 'MaxRoiPool', 'MaxUnpool', 'Mean', 'MeanVarianceNormalization',
            'Min', 'Mod', 'Mul', 'Multinomial', 'Neg', 'NonMaxSuppression',
            'NonZero', 'Not', 'OneHot', 'Or', 'PRelu', 'Pad', 'Pow',
            'QLinearConv', 'QLinearMatMul', 'QuantizeLinear', 'RNN',
            'RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike',
            'Range', 'Reciprocal', 'ReduceL1', 'ReduceL2', 'ReduceLogSum',
            'ReduceLogSumExp', 'ReduceMax', 'ReduceMean', 'ReduceMin',
            'ReduceProd', 'ReduceSum', 'ReduceSumSquare', 'Relu', 'Reshape',
            'Resize', 'ReverseSequence', 'RoiAlign', 'Round', 'Scan', 'Scatter',
            'ScatterElements', 'ScatterND', 'Selu', 'SequenceAt', 'SequenceConstruct',
            'SequenceEmpty', 'SequenceErase', 'SequenceInsert', 'SequenceLength',
            'Shape', 'Shrink', 'Sigmoid', 'Sign', 'Sin', 'Sinh', 'Size', 'Slice',
            'Softmax', 'SoftmaxCrossEntropyLoss', 'Softplus', 'Softsign', 'SpaceToDepth',
            'Split', 'SplitToSequence', 'Sqrt', 'Squeeze', 'Sub', 'Sum', 'Tan',
            'Tanh', 'TfIdfVectorizer', 'ThresholdedRelu', 'Tile', 'TopK',
            'Transpose', 'Trilu', 'Unique', 'Unsqueeze', 'Upsample', 'Where',
            'Xor', 'ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper',
            'DictVectorizer', 'FeatureVectorizer', 'Imputer', 'LabelEncoder',
            'LinearClassifier', 'LinearRegressor', 'Normalizer', 'OneHotEncoder',
            'SVMClassifier', 'SVMRegressor', 'Scaler', 'TreeEnsembleClassifier',
            'TreeEnsembleRegressor', 'ZipMap'
        }


class CompatibilityVerifier:
    def __init__(self):
        self.onnx_checker = ONNXCompatibilityChecker()
    
    def verify_all_runtimes(
        self,
        model_path: str
    ) -> Dict[str, CompatibilityResult]:
        runtimes = ["onnxruntime", "tensorrt", "openvino", "tflite"]
        results = {}
        
        for runtime in runtimes:
            results[runtime] = self.onnx_checker.check_compatibility(model_path, runtime)
        
        return results
    
    def get_compatible_runtimes(
        self,
        model_path: str,
        require_compilation: bool = True
    ) -> List[str]:
        results = self.verify_all_runtimes(model_path)
        
        compatible = []
        for runtime, result in results.items():
            if require_compilation:
                if result.compatible and result.can_compile:
                    compatible.append(runtime)
            else:
                if result.compatible:
                    compatible.append(runtime)
        
        return compatible
    
    def get_best_runtime(
        self,
        model_path: str,
        device_class: str = "cpu"
    ) -> Optional[str]:
        compatible = self.get_compatible_runtimes(model_path)
        
        if not compatible:
            return None
        
        if device_class == "cuda":
            priority = ["tensorrt", "onnxruntime", "openvino", "tflite"]
        elif device_class == "gpu":
            priority = ["openvino", "onnxruntime", "tensorrt", "tflite"]
        else:
            priority = ["onnxruntime", "openvino", "tflite", "tensorrt"]
        
        for runtime in priority:
            if runtime in compatible:
                return runtime
        
        return compatible[0] if compatible else None
    
    def get_blocking_issues(
        self,
        model_path: str,
        runtime: str
    ) -> List[CompatibilityIssue]:
        result = self.onnx_checker.check_compatibility(model_path, runtime)
        return [i for i in result.issues if i.severity == "error"]
    
    def dry_run_compilation(
        self,
        model_path: str,
        runtime: str
    ) -> Tuple[bool, List[str]]:
        result = self.onnx_checker.check_compatibility(model_path, runtime)
        
        errors = []
        for issue in result.issues:
            if issue.severity == "error":
                errors.append(f"{issue.category}: {issue.message}")
        
        return result.can_compile, errors
