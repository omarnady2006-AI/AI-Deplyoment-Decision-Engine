"""
core/fix_suggester.py

Suggest concrete fixes for per-runtime diagnostics.
"""
from __future__ import annotations

from typing import Any


def suggest_fixes(runtime: Any, diagnostics: list) -> list:
    """
    Produce a list of fix suggestion dicts from per-runtime diagnostics.

    Each dict contains: runtime, issue, suggestion, severity.
    """
    rt_name = runtime.value if hasattr(runtime, "value") else str(runtime)
    fixes = []
    for d in diagnostics:
        sev = getattr(d, "severity", None)
        if sev not in ("FAIL", "WARN"):
            continue
        fixes.append({
            "runtime":    rt_name,
            "issue":      getattr(d, "title", getattr(d, "message", "unknown issue")),
            "suggestion": getattr(d, "suggestion", "Review diagnostic and resolve before deployment."),
            "severity":   sev,
        })
    return fixes
