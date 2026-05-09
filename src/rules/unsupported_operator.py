from __future__ import annotations

from src.core.rule import Rule, RuleResult
from src.core.runtime import RuntimeName
from src.diagnostics.report import Diagnostic


class UnsupportedOperatorRule(Rule):
    id = "unsupported_operator"  # NOTE: class attribute; does not shadow built-in id()

    _RUNTIME_RULES: dict[
        RuntimeName, dict[str, tuple[str, str, str]]
    ] = {
        # ── Legacy / internal names (kept for backward compatibility) ──────────
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

        # ── ONNX execution providers ────────────────────────────────────────────
        # CPU and CUDA providers share the same ONNX op set — no restrictions.
        RuntimeName.ONNX_CPU:  {},
        RuntimeName.ONNX_CUDA: {},

        # ONNX-based TensorRT execution provider — same operator restrictions
        # as the native TensorRT runtime.
        RuntimeName.TRT: {
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

        # ── TFLite runtimes ─────────────────────────────────────────────────────
        # CPU and GPU delegates share the same TFLite op set.
        RuntimeName.TFLITE_CPU: {
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
        RuntimeName.TFLITE_GPU: {
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

        # ── TensorFlow (SavedModel / frozen-graph) runtimes ────────────────────
        # TF executes the full TF op set natively; BatchNorm is handled
        # transparently but NonMaxSuppression has GPU delegate caveats.
        RuntimeName.TF_CPU: {
            "model.has_non_max_suppression": (
                "NonMaxSuppression",
                "WARN",
                "verify tf.image.combined_non_max_suppression compatibility",
            ),
        },
        RuntimeName.TF_GPU: {
            "model.has_non_max_suppression": (
                "NonMaxSuppression",
                "WARN",
                "verify tf.image.combined_non_max_suppression GPU compatibility",
            ),
        },

        # ── Native TensorRT serialised-engine runtime ───────────────────────────
        RuntimeName.TRT_NATIVE: {
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

        # ── OpenVINO IR runtimes ────────────────────────────────────────────────
        # OpenVINO IR supports a broad op set; ConvTranspose maps to
        # ConvolutionBackpropData and is supported, but dynamic shapes and
        # some resize modes can be problematic.
        RuntimeName.OV_CPU: {
            "model.has_dynamic_shapes": (
                "DynamicShapes",
                "WARN",
                "reshape to static shapes before export for best performance",
            ),
            "model.has_resize": (
                "Resize",
                "WARN",
                "verify interpolation mode is supported in OpenVINO IR",
            ),
        },
        RuntimeName.OV_GPU: {
            "model.has_dynamic_shapes": (
                "DynamicShapes",
                "WARN",
                "reshape to static shapes before export for best performance",
            ),
            "model.has_resize": (
                "Resize",
                "WARN",
                "verify interpolation mode is supported in OpenVINO IR",
            ),
            "model.uses_batch_normalization": (
                "BatchNormalization",
                "WARN",
                "fold batchnorm before OpenVINO export for GPU performance",
            ),
        },
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
                        "Checked operator compatibility for runtime "
                        f"{runtime.value} with no blocking findings."
                    ),
                    suggestion=None,
                )
            )

        passed = not any(diagnostic.severity == "FAIL" for diagnostic in diagnostics)
        return RuleResult(rule_id=self.id, passed=passed, diagnostics=diagnostics)
