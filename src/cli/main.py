from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from src.core.model_hash import compute_model_hash

# Support for multiple model formats
SUPPORTED_FORMATS = [".onnx", ".pb", ".pt", ".pth", ".h5", ".joblib", ".pkl", ".pickle"]


def _ensure_local_import_path() -> None:
    package_root = Path(__file__).resolve().parents[1]
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)


def _resolve_analyze_model() -> Callable[[str], object]:
    from src.core.model_analysis import analyze_model
    return analyze_model


def _extract_ops(analysis: object) -> set[str]:
    graph = getattr(analysis, "graph", None)
    nodes = getattr(graph, "node", None)
    if nodes is None:
        return set()
    op_types: set[str] = set()
    for node in nodes:
        op_type = getattr(node, "op_type", None)
        if isinstance(op_type, str):
            op_types.add(op_type)
    return op_types


def _extract_operator_counts(analysis: object) -> dict[str, int]:
    graph = getattr(analysis, "graph", None)
    nodes = getattr(graph, "node", None)
    counts: dict[str, int] = {}
    if nodes is None:
        return counts
    for node in nodes:
        op_type = getattr(node, "op_type", None)
        if not isinstance(op_type, str):
            continue
        counts[op_type] = counts.get(op_type, 0) + 1
    return counts


def _build_facts(analysis: object) -> dict[str, object]:
    facts: dict[str, object] = {}

    if isinstance(analysis, dict):
        facts.update(analysis)
    else:
        analysis_facts = getattr(analysis, "facts", None)
        if isinstance(analysis_facts, dict):
            facts.update(analysis_facts)

    op_types = _extract_ops(analysis)
    operator_counts = _extract_operator_counts(analysis)
    if op_types:
        facts["model.has_non_max_suppression"] = "NonMaxSuppression" in op_types
        facts["model.has_resize"] = "Resize" in op_types
        facts["model.has_conv_transpose"] = "ConvTranspose" in op_types
        facts["model.uses_layer_normalization"] = "LayerNormalization" in op_types
        facts["model.uses_batch_normalization"] = "BatchNormalization" in op_types
        facts["model.has_conv"] = "Conv" in op_types or "ConvTranspose" in op_types
        facts["model.has_attention"] = (
            "Attention" in op_types or "MultiHeadAttention" in op_types
        )

    for key in (
        "model.has_non_max_suppression",
        "model.has_resize",
        "model.has_conv_transpose",
        "model.uses_layer_normalization",
        "model.uses_batch_normalization",
        "model.has_conv",
        "model.has_attention",
    ):
        facts[key] = bool(facts.get(key, False))

    facts["operator_counts"] = operator_counts
    
    parameter_count = getattr(analysis, "parameter_count", 0) or 0
    has_dynamic_shapes = getattr(analysis, "has_dynamic_shapes", False) or False
    
    facts["parameter_count"] = parameter_count
    facts["model.has_dynamic_shapes"] = has_dynamic_shapes
    facts["sequential_depth_estimate"] = sum(operator_counts.values())
    if "parameter_scale_class" not in facts:
        if parameter_count >= 25_000_000:
            facts["parameter_scale_class"] = "large"
        elif parameter_count >= 8_000_000:
            facts["parameter_scale_class"] = "medium"
        else:
            facts["parameter_scale_class"] = "small"

    return facts


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deploycheck",
        description="Analyze a model and recommend the best deployment runtime.",
    )
    parser.add_argument("model_path", help="Path to an ONNX model file.")
    parser.add_argument(
        "--profile",
        dest="profile_path",
        help="Path to input profile JSON.",
        default=None,
    )
    parser.add_argument(
        "--show-trust",
        action="store_true",
        help="Show trust metrics in prediction output.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class PipelineOutput:
    model_name: str
    report: dict
    best_runtime: str
    decision: str
    recommendation_status: str
    fail_count: int
    warn_count: int


def _count_severity(diagnostics: list, severity: str) -> int:
    return sum(
        1
        for diagnostic in diagnostics
        if getattr(diagnostic, "severity", None) == severity
    )


def _load_input_profile(profile_path: str | None):
    if profile_path is None:
        return None
    from src.core.input_profile import input_profile_from_dict

    profile_payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(profile_payload, dict):
        raise ValueError("profile JSON must be an object")
    return input_profile_from_dict(profile_payload)


def run_pipeline(model_path: str, profile_path: str | None = None) -> PipelineOutput:
    _ensure_local_import_path()

    from src.core.calibration_history import (
        CalibrationRecord,
        compute_calibration_stats,
        compute_model_bucket,
        compute_runtime_priors,
        load_history,
        record_calibration,
    )
    from src.core.action_plan import generate_action_plan
    from src.core.analysis_reliability import evaluate_analysis_reliability
    from src.core.advisor_report import AdvisorInput, generate_advisor_report
    from src.core.confidence import DecisionConfidence, compute_confidence
    from src.core.confidence_calibration import adjust_confidence_with_history
    from src.core.decision import DecisionReport
    from src.core.deployment_profile import predict_deployment_profile
    from src.core.fix_suggester import suggest_fixes
    from src.core.global_decision import compute_global_decision
    from src.core.model_summary import RuntimeAssessment, summarize_model
    from src.core.perf_behavior import classify_execution_behavior
    from src.core.perf_risk import PerfRiskReport
    from src.core.perf_risk import estimate_inference_risk
    from src.core.runtime import RuntimeName
    from src.core.runtime_selector import recommend_runtime
    from src.core.recommendation_policy import apply_recommendation_policy
    from src.rules.unsupported_operator import UnsupportedOperatorRule
    from src.validation.interpret_validation import interpret_validation_result
    from src.validation.runtime_validator import validate_runtime

    analyze_model = _resolve_analyze_model()
    analysis = analyze_model(model_path)
    facts = _build_facts(analysis)
    input_profile = _load_input_profile(profile_path)
    calibration_history_path = str(Path(".deploycheck_calibration_history.json"))
    model_hash = compute_model_hash(model_path)

    def _validation_outcome(validation_result) -> Literal["SUCCESS", "FAIL", "SLOW"]:
        message = (validation_result.error_message or "").lower()
        if "execution time grew more than 4x" in message or (
            "slow" in message and "stability" in message
        ):
            return "SLOW"
        if validation_result.executed and validation_result.success:
            return "SUCCESS"
        return "FAIL"

    def _confidence_level(score: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
        if score >= 0.80:
            return "HIGH"
        if score >= 0.50:
            return "MEDIUM"
        return "LOW"

    rules = [UnsupportedOperatorRule()]
    calibration_summary = summarize_model(assessments=[], facts=facts)
    model_bucket = compute_model_bucket(calibration_summary, facts)
    constraints_by_runtime = {}
    validations_by_runtime = {}
    for runtime in RuntimeName:
        validation_result = validate_runtime(model_path, runtime)
        validations_by_runtime[runtime] = validation_result
        record_calibration(
            CalibrationRecord(
                model_hash=model_hash,
                runtime=runtime,
                predicted_decision=validation_result.predicted_decision,
                actual_outcome=_validation_outcome(validation_result),
                timestamp=time.time(),
                bucket=model_bucket,
            ),
            calibration_history_path,
        )
        constraints_by_runtime[runtime] = interpret_validation_result(
            validation_result, input_profile
        )
    calibration_stats = compute_calibration_stats(
        load_history(calibration_history_path)
    )
    runtime_priors = compute_runtime_priors(load_history(calibration_history_path))

    def _build_reports_and_assessments(
        recommendation_obj,
        reliability_score: float | None = None,
    ) -> tuple[dict[RuntimeName, DecisionReport], list[RuntimeAssessment]]:
        reports: dict[RuntimeName, DecisionReport] = {}
        for evaluation in recommendation_obj.evaluations:
            confidence = compute_confidence(
                decision=evaluation.decision,
                diagnostics=evaluation.diagnostics,
                constraints=evaluation.constraints,
                validation=validations_by_runtime[evaluation.runtime],
                reliability_score=reliability_score,
            )
            adjusted_score = adjust_confidence_with_history(
                confidence_score=confidence.score,
                runtime=evaluation.runtime,
                stats=calibration_stats,
            )
            if abs(adjusted_score - confidence.score) > 1e-12:
                reasons = list(confidence.reasons)
                reasons.append("empirical calibration adjustment applied")
                confidence = DecisionConfidence(
                    score=adjusted_score,
                    level=_confidence_level(adjusted_score),
                    reasons=reasons,
                )
            reports[evaluation.runtime] = DecisionReport(
                decision=evaluation.decision,
                diagnostics=evaluation.diagnostics,
                constraints=evaluation.constraints,
                confidence=confidence,
            )

        assessment_items: list[RuntimeAssessment] = []
        for evaluation in recommendation_obj.evaluations:
            report = reports[evaluation.runtime]
            confidence = report.confidence
            if confidence is None:
                continue
            assessment_items.append(
                RuntimeAssessment(
                    runtime=evaluation.runtime,
                    decision=evaluation.decision,
                    constraints=evaluation.constraints,
                    confidence=confidence,
                    utility_score=evaluation.utility_score,
                    validation=validations_by_runtime[evaluation.runtime],
                )
            )
        return reports, assessment_items

    def _with_behavior(
        report: PerfRiskReport, bound_value: str, gpu_scaling_value: str
    ) -> PerfRiskReport:
        explanation = list(report.explanation)
        explanation.append(
            f"Execution behavior indicates {bound_value} with GPU scaling {gpu_scaling_value}."
        )
        return PerfRiskReport(
            estimated_latency_class=report.estimated_latency_class,
            memory_pressure=report.memory_pressure,
            parallelism=report.parallelism,
            realtime_cpu_viable=report.realtime_cpu_viable,
            realtime_gpu_viable=report.realtime_gpu_viable,
            explanation=explanation,
        )

    recommendation = recommend_runtime(
        rules,
        facts,
        constraints_by_runtime=constraints_by_runtime,
        validations_by_runtime=validations_by_runtime,
        runtime_priors=runtime_priors,
    )
    facts["recommended_runtime"] = recommendation.best_runtime.value
    decision_reports, assessments = _build_reports_and_assessments(recommendation)
    model_summary = summarize_model(assessments=assessments, facts=facts)
    perf_risk = estimate_inference_risk(
        facts=facts, model_summary=model_summary, assessments=assessments
    )
    execution_behavior = classify_execution_behavior(
        facts=facts, perf_risk=perf_risk, model_summary=model_summary
    )
    analysis_reliability = evaluate_analysis_reliability(
        facts=facts,
        perf_risk=perf_risk,
        execution_behavior=execution_behavior,
        validations=validations_by_runtime,
    )
    perf_risk = _with_behavior(
        perf_risk, execution_behavior.bound, execution_behavior.gpu_scaling
    )

    recommendation = recommend_runtime(
        rules,
        facts,
        constraints_by_runtime=constraints_by_runtime,
        validations_by_runtime=validations_by_runtime,
        gpu_scaling=execution_behavior.gpu_scaling,
        reliability_level=analysis_reliability.level,
        reliability_score=analysis_reliability.score,
        runtime_priors=runtime_priors,
    )
    facts["recommended_runtime"] = recommendation.best_runtime.value
    decision_reports, assessments = _build_reports_and_assessments(
        recommendation, reliability_score=analysis_reliability.score
    )
    model_summary = summarize_model(assessments=assessments, facts=facts)
    perf_risk = estimate_inference_risk(
        facts=facts, model_summary=model_summary, assessments=assessments
    )
    execution_behavior = classify_execution_behavior(
        facts=facts, perf_risk=perf_risk, model_summary=model_summary
    )
    analysis_reliability = evaluate_analysis_reliability(
        facts=facts,
        perf_risk=perf_risk,
        execution_behavior=execution_behavior,
        validations=validations_by_runtime,
    )
    perf_risk = _with_behavior(
        perf_risk, execution_behavior.bound, execution_behavior.gpu_scaling
    )
    global_decision = compute_global_decision(recommendation)
    confidence_by_runtime = {
        runtime: decision_reports[runtime].confidence
        for runtime in RuntimeName
        if decision_reports[runtime].confidence is not None
    }
    recommendation_policy = apply_recommendation_policy(
        recommendation=recommendation,
        confidence=confidence_by_runtime,
        reliability=analysis_reliability,
        perf_risk=perf_risk,
        validations=validations_by_runtime,
    )
    deployment_profile = predict_deployment_profile(
        summary=model_summary, assessments=assessments
    )

    fix_reports = {
        runtime: suggest_fixes(runtime, decision_reports[runtime].diagnostics)
        for runtime in RuntimeName
    }
    action_plan = generate_action_plan(
        recommendation=recommendation,
        decision_reports=decision_reports,
        fix_reports=fix_reports,
        deployment_profile=deployment_profile,
        model_summary=model_summary,
    )

    selected_runtime = (
        recommendation_policy.final_runtime
        if recommendation_policy.final_runtime is not None
        else recommendation.best_runtime
    )
    selected_decision_report = decision_reports[selected_runtime]
    fail_count = _count_severity(selected_decision_report.diagnostics, "FAIL")
    warn_count = _count_severity(selected_decision_report.diagnostics, "WARN")

    decision_status = "APPROVED"
    if selected_decision_report.decision in ("BLOCK", "UNSUPPORTED"):
        decision_status = "REJECTED"
    elif selected_decision_report.decision in ("ALLOW_WITH_CONDITIONS", "SUPPORTED_WITH_WARNINGS"):
        decision_status = "CONDITIONAL_APPROVAL"

    rejection_reasons = []
    conditional_reasons = []
    for d in selected_decision_report.diagnostics:
        if d.severity == "FAIL":
            rejection_reasons.append(d.message)
        elif d.severity == "WARN":
            conditional_reasons.append(d.message)

    best_eval = None
    for ev in recommendation.evaluations:
        if ev.runtime == selected_runtime:
            best_eval = ev
            break

    operator_count_val = facts.get("operator_count", 0)
    parameter_count_val = facts.get("parameter_count", 0)
    unsupported_ops_val = facts.get("unsupported_ops", [])
    
    advisor_input = AdvisorInput(
        model_path=model_path,
        model_hash=model_hash,
        selected_runtime=selected_runtime.value,
        decision_status=decision_status,
        confidence=getattr(selected_decision_report.confidence, 'score', 0.5) if selected_decision_report.confidence else 0.5,
        predicted_latency_ms=None,
        predicted_memory_mb=None,
        operator_count=int(operator_count_val) if isinstance(operator_count_val, (int, float)) else 0,
        parameter_count=int(parameter_count_val) if isinstance(parameter_count_val, (int, float)) else 0,
        has_dynamic_shapes=bool(facts.get("model.has_dynamic_shapes", False)),
        unsupported_ops=list(unsupported_ops_val) if isinstance(unsupported_ops_val, (list, tuple)) else [],
        warnings=[d.message for d in selected_decision_report.diagnostics if d.severity == "WARN"],
        rejection_reasons=rejection_reasons,
        conditional_reasons=conditional_reasons,
    )
    report = generate_advisor_report(advisor_input)

    return PipelineOutput(
        model_name=Path(model_path).name,
        report=report,
        best_runtime=(
            recommendation_policy.final_runtime.value
            if recommendation_policy.final_runtime is not None
            else "n/a"
        ),
        decision=selected_decision_report.decision,
        recommendation_status=recommendation_policy.status,
        fail_count=fail_count,
        warn_count=warn_count,
    )


def _enforce_decision_gate(result: PipelineOutput, model_path: str) -> tuple[bool, str]:
    """
    Compute advisory deployment recommendation.
    
    Returns:
        (allowed, reason): Always allowed=True in advisory mode, plus recommendation text.
    """
    from src.core.decision import DeploymentDecision
    
    # Map decision string to enum
    try:
        decision = DeploymentDecision(result.decision)
    except ValueError:
        decision = DeploymentDecision.UNSUPPORTED
    
    # Advisory recommendation logic
    if decision == DeploymentDecision.UNSUPPORTED:
        reason = (
            f"ADVISORY HIGH RISK: Model '{result.model_name}' is predicted NOT RUNNABLE.\n"
            f"Decision: {result.decision}\n"
            f"Failures: {result.fail_count}\n"
            f"Warnings: {result.warn_count}\n"
            f"Recommendation: do not deploy until issues are fixed."
        )
        return True, reason
    
    elif decision == DeploymentDecision.SUPPORTED_WITH_WARNINGS:
        reason = (
            f"ADVISORY CONDITIONAL: Model '{result.model_name}' is RUNNABLE WITH CONDITIONS.\n"
            f"Decision: {result.decision}\n"
            f"Failures: {result.fail_count}\n"
            f"Warnings: {result.warn_count}\n"
            f"Recommendation: run with close monitoring and guardrails."
        )
        return True, reason
    
    elif decision == DeploymentDecision.SUPPORTED:
        reason = (
            f"ADVISORY LOW RISK: Model '{result.model_name}' is predicted RUNNABLE.\n"
            f"Decision: {result.decision}\n"
            f"Failures: {result.fail_count}\n"
            f"Warnings: {result.warn_count}\n"
            f"Recommendation: proceed with standard runtime monitoring."
        )
        return True, reason
    
    else:
        reason = (
            f"ADVISORY UNKNOWN: Unknown decision '{result.decision}'.\n"
            f"Recommendation: treat deployment as high risk and review diagnostics."
        )
        return True, reason


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_local_import_path()
    
    # Standard pipeline with advisory gate
    try:
        result = run_pipeline(args.model_path, profile_path=args.profile_path)
        print(result.report)
        
        # ADVISORY GATE: Emit recommendation before execution
        allowed, enforcement_reason = _enforce_decision_gate(result, args.model_path)
        
        print("\n" + "="*70)
        print("DEPLOYMENT ADVISORY DECISION")
        print("="*70)
        print(enforcement_reason)
        print("="*70)
        
        # Advisory mode does not block execution.
        
        # Show trust metrics if requested
        if args.show_trust:
            from src.core.ood_detection import create_ood_detector
            from src.core.device_profiler import DeviceProfiler
            from src.core.decision_schema import create_prediction_output
            
            try:
                detector = create_ood_detector()
                profiler = DeviceProfiler()
                device_profile = profiler.get_current_profile()
                
                if device_profile is None:
                    print("Warning: Could not get device profile for trust analysis")
                    return 1
                
                # Get OOD result
                from src.core.ood_detection import ModelArchitectureFeatures, DeviceFingerprintFeatures
                model_features = ModelArchitectureFeatures()
                device_features = DeviceFingerprintFeatures(
                    cpu_cores=getattr(device_profile, 'cpu_cores', 0) or 0,
                    cpu_freq_mhz=getattr(device_profile, 'cpu_freq_mhz', 0.0) or 0.0,
                    gpu_memory_mb=getattr(device_profile, 'gpu_memory_mb', 0) or 0,
                    system_memory_mb=getattr(device_profile, 'system_memory_mb', 0) or 0,
                    device_class=getattr(device_profile, 'device_class', 'unknown') or 'unknown',
                    platform=getattr(device_profile, 'platform', 'unknown') or 'unknown',
                )
                
                ood_result = detector.detect(
                    model_features=model_features,
                    device_features=device_features,
                    runtime="onnxruntime",
                    precision="fp32",
                )
                
                # Load validation metrics
                import json
                from pathlib import Path
                
                dev_acc = 0.0
                fam_acc = 0.0
                
                device_report_path = Path("evaluation_reports/device_report.json")
                if device_report_path.exists():
                    with open(device_report_path) as f:
                        dev_data = json.load(f)
                        dev_acc = dev_data.get("mean_accuracy", 0.0)
                
                family_report_path = Path("evaluation_reports/family_report.json")
                if family_report_path.exists():
                    with open(family_report_path) as f:
                        fam_data = json.load(f)
                        fam_acc = fam_data.get("mean_accuracy", 0.0)
                
                # Create prediction output
                pred_output = create_prediction_output(
                    raw_probability=0.8,  # Placeholder
                    trust_level=ood_result.trust_level,
                    ood_score=ood_result.ood_score,
                    empirical_support_count=ood_result.empirical_support_count,
                    generalization_risk=1.0 - ood_result.is_in_distribution,
                    device_generalization_accuracy=dev_acc,
                    family_generalization_accuracy=fam_acc,
                    confidence_downgrade_factor=ood_result.confidence_downgrade_factor,
                )
                
                print("\n" + "="*50)
                print("TRUST METRICS")
                print("="*50)
                print(str(pred_output))
                print("="*50)
                
            except Exception as e:
                print(f"\nWarning: Could not compute trust metrics: {e}", file=sys.stderr)
        
        # Return based on recommendation status (but only if allowed through gate)
        if result.recommendation_status == "RECOMMEND":
            return 0
        if result.recommendation_status == "UNCERTAIN":
            return 4
        return 5
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
