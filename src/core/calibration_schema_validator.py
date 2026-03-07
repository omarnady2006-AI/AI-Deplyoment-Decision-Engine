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
Calibration Schema Validator

Validates calibration JSON data against expected schema.
"""


from typing import Any


class CalibrationSchemaError(Exception):
    """Raised when calibration schema validation fails."""
    pass


def validate_calibration_schema(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate calibration JSON data against expected schema.
    
    Args:
        data: Calibration data dictionary
        
    Returns:
        The validated data dictionary
        
    Raises:
        CalibrationSchemaError: If validation fails
    """
    if not isinstance(data, dict):
        raise CalibrationSchemaError("Calibration data must be a dictionary")
    
    required_keys = ["environment", "benchmarks", "comparative_ranking"]
    for key in required_keys:
        if key not in data:
            raise CalibrationSchemaError(f"Missing required key: '{key}'")
    
    if not isinstance(data.get("environment"), dict):
        raise CalibrationSchemaError("'environment' must be a dictionary")
    
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise CalibrationSchemaError("'benchmarks' must be a dictionary")
    
    ranking = data.get("comparative_ranking")
    if not isinstance(ranking, dict):
        raise CalibrationSchemaError("'comparative_ranking' must be a dictionary")
    
    return data
