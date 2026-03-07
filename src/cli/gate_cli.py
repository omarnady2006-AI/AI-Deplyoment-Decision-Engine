"""
Deployment Gate Authority - Pre-Deployment Gate

This module provides a mandatory deployment gate that:
1. Evaluates models before deployment
2. Outputs machine-readable JSON results
3. Uses exit codes for CI/CD integration

Usage:
    python -m cli.gate --mode gate --model path/to/model.onnx --output-json result.json

Exit codes (advisory mode):
    0 - Analysis completed (all decisions)
    2 - Error during analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.core.pipeline import run_pipeline, PipelineResult
from src.core.model_hash import compute_model_hash
from src.core.decision import DeploymentDecision, make_decision
from src.core.decision_schema import (
    DecisionSchema,
    SchemaVersion,
    ScoringWeights,
)
from src.core.confidence import ConfidenceLevel
from src.core.device_profiler import DeviceProfiler
from src.diagnostics.report import Diagnostic, DiagnosticSeverity


EXIT_ALLOWED = 0
EXIT_BLOCKED = 0
EXIT_ERROR = 2


@dataclass
class GateResult:
    """Machine-readable gate result."""
    decision: str
    confidence: float
    confidence_level: str
    risk_level: str
    blocking_reasons: list[str]
    runtime_limits: dict[str, Any]
    model_hash: str
    device_fingerprint: str
    timestamp: str
    schema_version: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "risk_level": self.risk_level,
            "blocking_reasons": self.blocking_reasons,
            "runtime_limits": self.runtime_limits,
            "model_hash": self.model_hash,
            "device_fingerprint": self.device_fingerprint,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }


def analyze_model_for_gate(
    model_path: str,
    device_fingerprint: str,
) -> tuple[list[Diagnostic], dict[str, Any], str]:
    """
    Analyze a model and return diagnostics, constraints, and model hash.
    
    Uses the canonical pipeline module for all analysis.
    
    Returns:
        Tuple of (diagnostics, constraints_dict, model_hash)
    """
    diagnostics: list[Diagnostic] = []
    constraints: dict[str, Any] = {}
    
    pipeline_result = run_pipeline(model_path)
    model_hash = pipeline_result.model_hash
    
    if not pipeline_result.success:
        diagnostics.append(Diagnostic(
            id="analysis_failed",
            severity=DiagnosticSeverity.FAIL.value,
            title="Model Analysis Failed",
            message=f"Failed to analyze model: {pipeline_result.error}",
            suggestion="Check model file format and integrity",
        ))
        return diagnostics, constraints, model_hash
    
    analysis = pipeline_result.analysis
    if analysis is None:
        diagnostics.append(Diagnostic(
            id="analysis_failed",
            severity=DiagnosticSeverity.FAIL.value,
            title="Model Analysis Failed",
            message="Analysis returned no results",
            suggestion="Check model file format and integrity",
        ))
        return diagnostics, constraints, model_hash
    
    for op in analysis.unsupported_ops:
        diagnostics.append(Diagnostic(
            id=f"unsupported_op_{op}",
            severity=DiagnosticSeverity.FAIL.value,
            title=f"Unsupported Operation: {op}",
            message=f"Unsupported operation: {op}",
            suggestion=f"Replace {op} with supported alternative",
        ))
    
    if analysis.has_dynamic_shapes:
        diagnostics.append(Diagnostic(
            id="dynamic_shapes",
            severity=DiagnosticSeverity.WARN.value,
            title="Dynamic Shapes Detected",
            message="Model contains dynamic shapes",
            suggestion="Use static shapes for production deployment",
        ))
    
    if analysis.parameter_count > 50_000_000:
        diagnostics.append(Diagnostic(
            id="large_model",
            severity=DiagnosticSeverity.WARN.value,
            title="Large Model",
            message=f"Model has {analysis.parameter_count:,} parameters",
            suggestion="Consider model optimization or pruning",
        ))
    
    return diagnostics, constraints, model_hash


def compute_gate_decision(
    diagnostics: list[Diagnostic],
    constraints: dict[str, Any],
) -> tuple[str, float, str]:
    """
    Compute the gate decision from diagnostics and constraints.
    
    Returns:
        Tuple of (decision, confidence, risk_level)
    """
    decision_report = make_decision(diagnostics)
    decision = decision_report.decision
    
    fail_count = sum(1 for d in diagnostics if d.severity == "FAIL")
    warn_count = sum(1 for d in diagnostics if d.severity == "WARN")
    
    confidence_score = 1.0 - (fail_count * 0.5) - (warn_count * 0.1)
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    if decision in ("BLOCK", "UNSUPPORTED"):
        risk_level = "CRITICAL"
    elif decision in ("ALLOW_WITH_CONDITIONS", "SUPPORTED_WITH_WARNINGS"):
        if confidence_score < 0.5:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"
    else:
        if confidence_score < 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
    
    return decision, confidence_score, risk_level


def generate_gate_result(
    decision: str,
    confidence: float,
    risk_level: str,
    diagnostics: list[Diagnostic],
    constraints: dict[str, Any],
    model_hash: str,
    device_fingerprint: str,
) -> GateResult:
    """Generate the gate result object."""
    blocking_reasons = []
    for d in diagnostics:
        if d.severity == DiagnosticSeverity.FAIL.value:
            blocking_reasons.append(d.message)
    
    runtime_limits = {}
    if constraints:
        if "max_memory_mb" in constraints:
            runtime_limits["max_memory_mb"] = constraints["max_memory_mb"]
        if "max_latency_ms" in constraints:
            runtime_limits["max_latency_ms"] = constraints["max_latency_ms"]
    
    if decision in ("ALLOW_WITH_CONDITIONS", "SUPPORTED_WITH_WARNINGS"):
        runtime_limits.setdefault("max_runtime_seconds", 300)
        runtime_limits.setdefault("graceful_shutdown_timeout", 30)
    
    return GateResult(
        decision=decision,
        confidence=confidence,
        confidence_level=ConfidenceLevel.from_score(confidence).value if confidence else "UNKNOWN",
        risk_level=risk_level,
        blocking_reasons=blocking_reasons,
        runtime_limits=runtime_limits,
        model_hash=model_hash,
        device_fingerprint=device_fingerprint,
        timestamp=datetime.utcnow().isoformat() + "Z",
        schema_version=SchemaVersion.LATEST.value,
    )


def run_gate(
    model_path: str,
    output_json: Optional[str] = None,
) -> tuple[GateResult, int]:
    """
    Run the deployment gate on a model.
    
    Args:
        model_path: Path to the model file
        output_json: Optional path to write JSON result
        
    Returns:
        Tuple of (GateResult, exit_code)
    """
    profiler = DeviceProfiler()
    device_fingerprint = profiler.get_device_fingerprint()
    
    diagnostics, constraints, model_hash = analyze_model_for_gate(
        model_path, device_fingerprint
    )
    
    decision, confidence, risk_level = compute_gate_decision(diagnostics, constraints)
    
    result = generate_gate_result(
        decision=decision,
        confidence=confidence,
        risk_level=risk_level,
        diagnostics=diagnostics,
        constraints=constraints,
        model_hash=model_hash,
        device_fingerprint=device_fingerprint,
    )
    
    if output_json:
        output_path = Path(output_json)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
    
    exit_code = EXIT_ALLOWED
    
    return result, exit_code


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the gate CLI."""
    parser = argparse.ArgumentParser(
        description="Deployment Gate - Mandatory model evaluation before deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the model file to evaluate",
    )
    parser.add_argument(
        "--output-json", "-o",
        help="Path to write JSON result (machine-readable output)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args(argv)
    
    try:
        result, exit_code = run_gate(args.model, args.output_json)
        
        if args.verbose or not args.output_json:
            print(f"Decision: {result.decision}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Risk Level: {result.risk_level}")
            print(f"Model Hash: {result.model_hash}")
            print(f"Device: {result.device_fingerprint}")
            
            if result.blocking_reasons:
                print("\nBlocking Reasons:")
                for reason in result.blocking_reasons:
                    print(f"  - {reason}")
            
            if result.runtime_limits:
                print("\nRuntime Limits:")
                for key, value in result.runtime_limits.items():
                    print(f"  {key}: {value}")
        
        return exit_code
    
    except Exception as e:
        error_result = {
            "error": str(e),
            "decision": "ERROR",
            "confidence": 0.0,
            "risk_level": "UNKNOWN",
            "blocking_reasons": [f"Analysis error: {str(e)}"],
            "runtime_limits": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.write_text(json.dumps(error_result, indent=2))
        
        if args.verbose:
            print(f"Error: {e}", file=sys.stderr)
        
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
