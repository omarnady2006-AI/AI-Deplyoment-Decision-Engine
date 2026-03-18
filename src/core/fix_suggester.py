"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.runtime import RuntimeName
from src.diagnostics.report import Diagnostic


@dataclass(frozen=True)
class SuggestedFix:
    runtime: RuntimeName
    rule_id: str
    operator: str | None
    explanation: str
    actions: list[str]


@dataclass(frozen=True)
class FixReport:
    runtime: RuntimeName
    fixes: list[SuggestedFix]


_OPERATOR_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "NonMaxSuppression": (
        "NonMaxSuppression is often not natively supported in optimized inference "
        "engines for all export patterns.",
        [
            "Replace NonMaxSuppression with a runtime-specific plugin implementation.",
            "Move post-processing NMS to application code after model inference.",
        ],
    ),
    "Resize": (
        "Resize can degrade performance or require fallback when dynamic shape or "
        "unsupported interpolation modes are used.",
        [
            "Export static resize dimensions and fixed scale factors.",
            "Replace unsupported resize modes with nearest or bilinear variants the runtime supports.",
        ],
    ),
    "ConvTranspose": (
        "ConvTranspose may be unsupported or mapped to slow fallback kernels "
        "depending on runtime and model shape configuration.",
        [
            "Replace ConvTranspose with upsample plus convolution blocks.",
            "Refactor decoder layers to use runtime-supported upsampling operators.",
        ],
    ),
    "BatchNormalization": (
        "BatchNormalization can trigger fallback or extra kernels when not folded "
        "into surrounding convolution layers.",
        [
            "Fold batch normalization into preceding convolution weights during export.",
            "Run graph optimization passes to fuse convolution and batch normalization.",
        ],
    ),
    "LayerNormalization": (
        "LayerNormalization may not have efficient native kernels in some runtimes.",
        [
            "Fuse layer normalization with adjacent operations where possible.",
            "Rewrite layer normalization into supported primitive operations.",
        ],
    ),
}


def _extract_rule_id(diagnostic_id: str) -> str:
    if "." in diagnostic_id:
        return diagnostic_id.split(".", 1)[0]
    return diagnostic_id


def _extract_operator(message: str) -> str | None:
    for operator in sorted(_OPERATOR_TEMPLATES.keys()):
        pattern = rf"\b{re.escape(operator)}\b"
        if re.search(pattern, message):
            return operator
    return None


def suggest_fixes(runtime: RuntimeName, diagnostics: list[Diagnostic]) -> FixReport:
    fail_diagnostics = sorted(
        (diagnostic for diagnostic in diagnostics if diagnostic.severity == "FAIL"),
        key=lambda diagnostic: (
            diagnostic.id,
            diagnostic.title,
            diagnostic.message,
            diagnostic.suggestion or "",
        ),
    )

    fixes: list[SuggestedFix] = []
    for diagnostic in fail_diagnostics:
        rule_id = _extract_rule_id(diagnostic.id)
        operator = _extract_operator(diagnostic.message)

        if operator is not None and operator in _OPERATOR_TEMPLATES:
            explanation, actions = _OPERATOR_TEMPLATES[operator]
        else:
            explanation = (
                "Runtime cannot execute this part of the model with current export settings."
            )
            actions = [
                "Inspect the failing operator and replace it with a runtime-supported alternative.",
                "Re-export the model with compatibility-focused optimization settings.",
            ]

        fixes.append(
            SuggestedFix(
                runtime=runtime,
                rule_id=rule_id,
                operator=operator,
                explanation=explanation,
                actions=actions,
            )
        )

    return FixReport(runtime=runtime, fixes=fixes)
