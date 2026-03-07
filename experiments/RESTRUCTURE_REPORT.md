# RESTRUCTURE_REPORT.md
**Deployment Decision Engine — Repository Restructure**
Generated: 2026-03-07

---

## Final Verdict

```
SYSTEM_OPERATIONAL_AFTER_RESTRUCTURE
```

---

## 1. Phase 1 — File Classification

### CORE_ENGINE (required for runtime)
| File | Location |
|------|----------|
| `pipeline.py` | `src/core/` |
| `runtime_profiler.py` | `src/core/` |
| `model_analysis.py` | `src/core/` |
| `framework_adapter.py` | `src/core/` |
| `decision.py` | `src/core/` |
| `runtime.py` | `src/core/` |
| `runtime_selector.py` | `src/core/` |
| `confidence.py` | `src/core/` |
| `confidence_calibration.py` | `src/core/` |
| `risk_engine.py` | `src/core/` |
| `risk_based_decision.py` | `src/core/` |
| `model_hash.py` | `src/core/` |
| `persistence.py` | `src/core/` |
| `logging_config.py` | `src/core/` |
| `deployment_summary.py` | `src/core/` |
| `gpu_calibration.py` | `src/core/` |
| `benchmark_calibration.py` | `src/core/` |
| `calibration_history.py` | `src/core/` |
| `continuous_learning.py` | `src/core/` |
| `ground_truth_db.py` | `src/core/` |
| `ood_detection.py` | `src/core/` |
| `device_profiler.py` | `src/core/` |
| `decision_schema.py` | `src/core/` |
| `hardware_selector.py` | `src/core/` |
| `optional_dependencies.py` | `src/core/` |
| `perf_behavior.py` | `src/core/` |
| `perf_risk.py` | `src/core/` |
| `model_summary.py` | `src/core/` |
| `advisor_report.py` | `src/core/` |
| `action_plan.py` | `src/core/` |
| `global_decision.py` | `src/core/` |
| `recommendation_policy.py` | `src/core/` |
| `deployment_profile.py` | `src/core/` |
| `fix_suggester.py` | `src/core/` |
| `analysis_reliability.py` | `src/core/` |
| `decision_trace.py` | `src/core/` |
| `telemetry.py` | `src/core/` |
| `app_state.py` | `src/` |

### API_LAYER (required for runtime)
| File | Location |
|------|----------|
| `gui_app.py` | `src/api/` |
| `gui_routes.py` | `src/api/` |
| `gui_analysis_api.py` | `src/api/` |
| `gui_calibration_api.py` | `src/api/` |
| `gui_decision_api.py` | `src/api/` |
| `gui_state_api.py` | `src/api/` |

### VALIDATION (required for runtime)
| File | Location |
|------|----------|
| `execution_verifier.py` | `src/validation/` |
| `interpret_validation.py` | `src/validation/` |
| `operator_compatibility.py` | `src/validation/` |
| `runtime_validator.py` | `src/validation/` |

### UI_LAYER (required for runtime)
| File | Location |
|------|----------|
| `index.html` | `src/gui/static/` |
| `app.js` | `src/gui/static/` |
| `favicon.ico` | `src/gui/static/` |
| `logo.png` | `src/gui/static/` |

### UTILITIES (required for runtime)
| File | Location |
|------|----------|
| `unsupported_operator.py` | `src/rules/` |
| `report.py` | `src/diagnostics/` |
| `main.py` (entry) | `./` |
| `gui_app.py` (shim) | `./` |
| `run.bat` | `./` |
| `pyproject.toml` | `./` |

### EXPERIMENTAL / TESTING
| File | Classification |
|------|----------------|
| `run_experiment.py` | EXPERIMENTAL |
| `experiments/run_experiment.py` | EXPERIMENTAL (copy) |

### LEGACY / LSP
| File | Classification |
|------|----------------|
| `deployment_decision_engine/lsp/server.py` | LEGACY (LSP server, not core) |
| `deployment_decision_engine/__init__.py` | LEGACY |

---

## 2. Phase 2 — New Project Structure

```
deployment_decision_engine/
|
+-- src/                             <- Source code (unchanged)
|   +-- __init__.py
|   +-- app_state.py
|   +-- api/                         <- FastAPI server layer
|   |   +-- gui_app.py               <- FastAPI app entrypoint
|   |   +-- gui_routes.py
|   |   +-- gui_analysis_api.py
|   |   +-- gui_calibration_api.py
|   |   +-- gui_decision_api.py
|   |   +-- gui_state_api.py
|   +-- core/                        <- Engine (45 files)
|   |   +-- pipeline.py              <- Risk engine (332 KB)
|   |   +-- runtime_profiler.py      <- Runtime profiler (425 KB)
|   |   +-- model_analysis.py
|   |   +-- [... 42 other core modules]
|   +-- cli/                         <- CLI layer
|   |   +-- main.py
|   |   +-- gate_cli.py
|   +-- gui/                         <- UI layer
|   |   +-- static/
|   |       +-- index.html           <- Single-page app
|   |       +-- app.js
|   +-- validation/                  <- Validation modules
|   |   +-- execution_verifier.py
|   |   +-- interpret_validation.py
|   |   +-- operator_compatibility.py
|   |   +-- runtime_validator.py
|   +-- diagnostics/
|   |   +-- report.py
|   +-- rules/
|       +-- unsupported_operator.py
|
+-- experiments/                     <- Experiment scripts (NEW)
|   +-- run_experiment.py
|
+-- scripts/                         <- Helper scripts (NEW)
|   +-- start_server.py
|
+-- quarantine/                      <- Legacy/unused files (NEW)
|   +-- synthetic_models/
|   |   +-- oversized_501mb.onnx
|   |   +-- tmp_pad10.onnx
|   |   +-- _synthetic_*.onnx (original set)
|   +-- legacy_logs/
|   |   +-- run_fixed.log
|   |   +-- run_fixed_experiment.log
|   |   +-- experiment_report_fixed.txt
|   |   +-- experiment_results_fixed.json
|   +-- temp_files/
|       +-- corrupt_test.onnx
|       +-- dynamic_probe.onnx
|       +-- empty.onnx
|       +-- empty_test.onnx
|       +-- invalid_random.onnx
|       +-- temp_unsupported_op.onnx
|       +-- tmp_unsupported_case.onnx
|
+-- gui_app.py                       <- Root shim (NEW)
+-- main.py
+-- run_experiment.py
+-- run.bat
+-- pyproject.toml
+-- README.md
+-- SYSTEM_PERFORMANCE_REPORT.md
+-- mnist-8.onnx
+-- yolov5s.onnx
+-- resnet50-v2-7.onnx
```

---

## 3. Phase 3 — Files Required for Runtime

**FastAPI Server:** `src/api/gui_app.py`, `gui_routes.py`, `gui_analysis_api.py`, `gui_calibration_api.py`, `gui_decision_api.py`, `gui_state_api.py`, `src/app_state.py`

**Pipeline Engine:** `src/core/pipeline.py`, `model_analysis.py`, `model_hash.py`

**Runtime Profiler:** `src/core/runtime_profiler.py`

**Framework Adapter:** `src/core/framework_adapter.py`, `runtime_selector.py`, `runtime.py`

**Validation Modules:** `src/validation/runtime_validator.py`, `interpret_validation.py`, `execution_verifier.py`, `operator_compatibility.py`

**Not Required for Runtime:**
- `deployment_decision_engine/lsp/server.py` — LSP language server (dev tool)
- `src/cli/gate_cli.py` — CLI tool only
- `experiments/run_experiment.py` — Experiment harness
- All files in `quarantine/`

---

## 4. Phase 4 — Import Paths

All imports use the `src.*` prefix. No changes were required.

Root-level shim added:
```python
# gui_app.py (root)
from src.api.gui_app import app  # re-export for uvicorn
```

Verified working:
```
PYTHONPATH=. uvicorn gui_app:app --host 0.0.0.0 --port 8080
```

---

## 5. Phase 5 — Server Startup Verification

**Status: PASSED**

```
INFO:  Started server process
INFO:  Application startup complete.
INFO:  Uvicorn running on http://127.0.0.1:8080
```

Health check:
```
GET /api/health -> 200 {"status": "healthy"}
```

---

## 6. Phase 6 — Real UI Test

**Status: PASSED**

**UI Features verified:**
- Futuristic dark-mode landing page with "Get Started" button
- Dashboard: Home, Model, Analysis, Reports, Settings
- Model upload zone (file picker)
- Hardware profile form: CPU Cores, RAM, DDR Type, GPU, CUDA, TensorRT, VRAM, Target Latency, Memory Limit
- Run Analysis button (enabled after profile save)
- Results panel showing decision, confidence, runtime, latency

**yolov5s.onnx + STANDARD profile (4 cores, 8 GB RAM, no GPU):**

| Metric | Value |
|--------|-------|
| Best Runtime | `onnx_cpu` |
| Decision | SUPPORTED -> **APPROVED** |
| Confidence | 62% |
| Latency | 78.3 ms |
| Risk Level | MEDIUM |

---

## 7. Phase 7 — Experiment Results After Restructuring

**Status: SYSTEM_VALIDATED**

```
FINAL VERDICT: SYSTEM_VALIDATED
BUGS: []
WARNINGS: []
```

### Invariant Verification

| Invariant | Result |
|-----------|--------|
| Risk scales down: EDGE > STANDARD > PRODUCTION > HPC | PASS |
| Confidence varies across hardware profiles | PASS |
| Latency scales with hardware | PASS |
| Determinism (5 repeated requests) | PASS |
| Concurrency safety (5 workers) | PASS |
| Concurrency safety (10 workers) | PASS |
| Decision logic alignment with risk thresholds | PASS |

---

## 8. Summary of Changes

### Files Added
| File | Purpose |
|------|---------|
| `gui_app.py` (root) | Shim for `uvicorn gui_app:app` |
| `scripts/start_server.py` | Clean server launcher |
| `experiments/run_experiment.py` | Experiment copy under `experiments/` |

### Files Moved to quarantine/
- `_synthetic_synthetic_*.onnx` (4 files) -> `quarantine/synthetic_models/`
- `oversized_501mb.onnx` -> `quarantine/synthetic_models/`
- `tmp_pad10.onnx` -> `quarantine/synthetic_models/`
- `run_fixed.log`, `run_fixed_experiment.log` -> `quarantine/legacy_logs/`
- `experiment_report_fixed.txt`, `experiment_results_fixed.json` -> `quarantine/legacy_logs/`
- `corrupt_test.onnx`, `dynamic_probe.onnx`, `empty.onnx`, `empty_test.onnx` -> `quarantine/temp_files/`
- `invalid_random.onnx`, `temp_unsupported_op.onnx`, `tmp_unsupported_case.onnx` -> `quarantine/temp_files/`

### Engine Logic
**Zero changes** to any engine algorithm, risk computation, or decision logic.

---

```
+------------------------------------------------------+
|  SYSTEM_OPERATIONAL_AFTER_RESTRUCTURE                |
|                                                      |
|  [OK] Server starts cleanly                          |
|  [OK] UI loads and analysis pipeline works           |
|  [OK] Experiment: SYSTEM_VALIDATED (0 bugs)          |
|  [OK] All invariants: determinism, scaling, safety   |
|  [OK] Imports unchanged, engine logic untouched      |
+------------------------------------------------------+
```
