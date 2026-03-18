from dataclasses import dataclass
from enum import Enum
from typing import Literal


Severity = Literal["PASS", "WARN", "FAIL", "INFO"]


class DiagnosticSeverity(Enum):
    """Diagnostic severity levels for gate decisions."""
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"
    PASS = "PASS"


@dataclass(frozen=True)
class Diagnostic:
    id: str  # NOTE: dataclass field; does not shadow built-in id() in function scope
    severity: Severity
    title: str
    message: str
    suggestion: str | None = None


_SEVERITY_ORDER: dict[Severity, int] = {
    "FAIL": 0,
    "WARN": 1,
    "INFO": 2,
    "PASS": 3,
}


def format_report(diagnostics: list[Diagnostic]) -> str:
    sorted_diagnostics = sorted(
        diagnostics,
        key=lambda diagnostic: (
            _SEVERITY_ORDER[diagnostic.severity],
            diagnostic.id,
            diagnostic.title,
            diagnostic.message,
            diagnostic.suggestion or "",
        ),
    )

    sections: dict[Severity, list[Diagnostic]] = {
        "FAIL": [],
        "WARN": [],
        "INFO": [],
        "PASS": [],
    }
    for diagnostic in sorted_diagnostics:
        sections[diagnostic.severity].append(diagnostic)

    lines: list[str] = []
    for severity in ("FAIL", "WARN", "INFO", "PASS"):
        items = sections[severity]
        if not items:
            continue

        lines.append(f"=== {severity} ===")
        for diagnostic in items:
            lines.append(f"[{diagnostic.id}] {diagnostic.title}")
            lines.append(diagnostic.message)
            if diagnostic.suggestion:
                lines.append(f"Suggestion: {diagnostic.suggestion}")
            lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def has_blocking_issues(diagnostics: list[Diagnostic]) -> bool:
    return any(diagnostic.severity == "FAIL" for diagnostic in diagnostics)
