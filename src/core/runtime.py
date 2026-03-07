"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from enum import Enum


class RuntimeName(str, Enum):
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    TFLITE = "tflite"
