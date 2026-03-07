"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from src.core.runtime import RuntimeName
from src.diagnostics.report import Diagnostic


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    passed: bool
    diagnostics: list[Diagnostic]


class Rule(ABC):
    id: str

    @abstractmethod
    def evaluate(self, facts: dict[str, object], runtime: RuntimeName) -> RuleResult:
        """
        Evaluate rule against a set of model facts.
        Must be deterministic and side-effect free.
        """


def run_rules(
    rules: Sequence[Rule], facts: dict[str, object], runtime: RuntimeName
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for rule in rules:
        result = rule.evaluate(facts, runtime)
        diagnostics.extend(result.diagnostics)
    return diagnostics
