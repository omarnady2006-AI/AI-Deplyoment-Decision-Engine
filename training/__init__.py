"""
training/

OFFLINE-ONLY PACKAGE — data generation and model training.

Rules enforced at import time:
  1. Nothing under training/ may be imported by anything under src/.
  2. If any training module is imported from a runtime (src.*) context,
     a RuntimeError is raised immediately.
  3. These modules exist solely for offline: dataset generation + model training.

Correct usage:
    python -m training.train_decision_model          # train and save model
    python -m training.data_generator               # generate dataset only
"""
import sys as _sys
import os as _os

# ── Runtime import guard ────────────────────────────────────────────────────
# Inspect the calling frame; if it originates from src.* code, abort.
def _check_runtime_import() -> None:
    """Raise if this package is imported by src.* runtime code."""
    # Walk the call stack looking for any src.* frame.
    depth = 0
    while True:
        try:
            frame = _sys._getframe(depth)
        except ValueError:
            break
        pkg  = frame.f_globals.get("__package__") or ""
        name = frame.f_globals.get("__name__") or ""
        if pkg.startswith("src") or name.startswith("src."):
            raise RuntimeError(
                "FORBIDDEN: training package must NOT be imported at runtime. "
                f"Found src.* caller in stack: name={name!r}, package={pkg!r}. "
                "Run training scripts directly: python -m training.train_decision_model"
            )
        depth += 1

if _os.environ.get("DDE_ALLOW_TRAINING_IMPORT") != "1":
    _check_runtime_import()
del _check_runtime_import
