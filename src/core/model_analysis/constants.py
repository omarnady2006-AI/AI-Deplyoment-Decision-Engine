from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady — Deployment Decision Engine.
See LICENSE in the project root for full terms.
"""
"""core/model_analysis/constants.py — resource limits and operator tables."""

_MAX_MODEL_FILE_BYTES   = 2 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
_MAX_NODE_COUNT         = 1_000_000
_MAX_PARAM_COUNT        = 1_000_000_000
_MAX_GRAPH_DEPTH        = 100_000
_MAX_ZIP_FILES          = 1_000
_MAX_ZIP_RATIO          = 100
_MAX_PARSE_TIMEOUT_S    = 30
_MAX_PARSE_MEMORY_MB    = 1024

_FLOAT_DTYPES: frozenset[int] = frozenset({1, 10, 11, 16})

_ONNX_BASELINE_OPS: frozenset[str] = frozenset({
    "Abs","Acos","Acosh","Add","And","ArgMax","ArgMin","Asin","Asinh","Atan",
    "Atanh","AveragePool","BatchNormalization","BitShift","Cast","CastLike",
    "Ceil","Celu","Clip","Compress","Concat","ConcatFromSequence","Constant",
    "ConstantOfShape","Conv","ConvInteger","ConvTranspose","Cos","Cosh",
    "CumSum","DepthToSpace","DequantizeLinear","Det","Div","Dropout",
    "DynamicQuantizeLinear","Einsum","Elu","Equal","Erf","Exp","Expand",
    "EyeLike","Flatten","Floor","GRU","Gather","GatherElements","GatherND",
    "Gemm","GlobalAveragePool","GlobalLpPool","GlobalMaxPool","Greater",
    "GreaterOrEqual","GridSample","GroupNormalization","HardSigmoid",
    "Hardswish","Identity","If","InstanceNormalization","IsInf","IsNaN",
    "LSTM","LayerNormalization","LeakyRelu","Less","LessOrEqual","Log",
    "LogSoftmax","Loop","LpNormalization","LpPool","MatMul","MatMulInteger",
    "Max","MaxPool","MaxUnpool","Mean","MeanVarianceNormalization","Min","Mod",
    "Mul","Multinomial","Neg","NegativeLogLikelihoodLoss","NonMaxSuppression",
    "NonZero","Not","OneHot","Optional","OptionalGetElement",
    "OptionalHasElement","Or","PRelu","Pad","Pow","QLinearConv","QLinearMatMul",
    "QuantizeLinear","RNN","RandomNormal","RandomNormalLike","RandomUniform",
    "RandomUniformLike","Range","Reciprocal","ReduceL1","ReduceL2",
    "ReduceLogSum","ReduceLogSumExp","ReduceMax","ReduceMean","ReduceMin",
    "ReduceProd","ReduceSum","ReduceSumSquare","Relu","Reshape","Resize",
    "ReverseSequence","RoiAlign","Round","Scan","ScatterElements","ScatterND",
    "Selu","SequenceAt","SequenceConstruct","SequenceEmpty","SequenceErase",
    "SequenceInsert","SequenceLength","Shape","Shrink","Sigmoid","Sign","Sin",
    "Sinh","Size","Slice","Softmax","Softplus","Softsign","SpaceToDepth",
    "Split","SplitToSequence","Sqrt","Squeeze","StringNormalizer","Sub","Sum",
    "Tan","Tanh","TfIdfVectorizer","ThresholdedRelu","Tile","TopK","Transpose",
    "Trilu","Unique","Unsqueeze","Where","Xor",
})
