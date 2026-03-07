from __future__ import annotations

from src.core.rule import Rule, RuleResult
from src.core.runtime import RuntimeName
from src.diagnostics.report import Diagnostic


class UnsupportedOperatorRule(Rule):
    id = "unsupported_operator"

    _RUNTIME_RULES: dict[
        RuntimeName, dict[str, tuple[str, str, str]]
    ] = {
        RuntimeName.TENSORRT: {
            "model.has_non_max_suppression": (
                "NonMaxSuppression",
                "FAIL",
                "replace with plugin",
            ),
            "model.has_resize": (
                "Resize",
                "WARN",
                "export static resize",
            ),
            "model.has_conv_transpose": (
                "ConvTranspose",
                "WARN",
                "replace with supported upsampling block",
            ),
            "model.uses_layer_normalization": (
                "LayerNormalization",
                "WARN",
                "fuse or rewrite layer normalization",
            ),
        },
        RuntimeName.TFLITE: {
            "model.has_conv_transpose": (
                "ConvTranspose",
                "FAIL",
                "replace with supported upsampling block",
            ),
            "model.uses_batch_normalization": (
                "BatchNormalization",
                "WARN",
                "fold batchnorm",
            ),
            "model.has_resize": (
                "Resize",
                "WARN",
                "export static resize",
            ),
        },
        RuntimeName.ONNXRUNTIME: {},
    }

    def evaluate(self, facts: dict[str, object], runtime: RuntimeName) -> RuleResult:
        diagnostics: list[Diagnostic] = []
        runtime_rules = self._RUNTIME_RULES.get(runtime, {})

        violations: list[tuple[str, str, str]] = []
        for fact_key, (operator_name, severity, suggestion) in runtime_rules.items():
            is_present = bool(facts.get(fact_key, False))
            if is_present:
                violations.append((operator_name, severity, suggestion))

        for operator_name, severity, suggestion in sorted(
            violations, key=lambda item: item[0]
        ):
            title = (
                f"Unsupported operator for {runtime.value}"
                if severity == "FAIL"
                else "Operator may fallback to slower implementation"
            )
            diagnostics.append(
                Diagnostic(
                    id=f"{self.id}.{runtime.value}.{operator_name.lower()}",
                    severity=severity,
                    title=title,
                    message=f"{operator_name} detected for runtime {runtime.value}.",
                    suggestion=suggestion,
                )
            )

        if not diagnostics:
            diagnostics.append(
                Diagnostic(
                    id=f"{self.id}.{runtime.value}.pass",
                    severity="PASS",
                    title=f"No unsupported operators for {runtime.value}",
                    message=(
                        f"Checked operator compatibility for runtime "
                        f"{runtime.value} with no blocking findings."
                    ),
                    suggestion=None,
                )
            )

        passed = not any(diagnostic.severity == "FAIL" for diagnostic in diagnostics)
        return RuleResult(rule_id=self.id, passed=passed, diagnostics=diagnostics)
