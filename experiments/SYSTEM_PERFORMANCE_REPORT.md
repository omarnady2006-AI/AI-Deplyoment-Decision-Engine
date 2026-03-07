# System Performance Report

## Scope

Real end-to-end verification was executed against the running FastAPI server using UI-equivalent HTTP actions from [`src/static/app.js`](src/static/app.js) and discovered inline UI actions in [`src/static/index.html`](src/static/index.html).

Primary raw evidence is in [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json).

---

## 1) UI Action Discovery

### Active UI calls from [`src/static/app.js`](src/static/app.js)

| UI_ACTION | HTTP_METHOD | ENDPOINT | JS_FUNCTION |
|---|---|---|---|
| Backend health ping | GET | `/api/health` | [`pingBackendFromUi()`](src/static/app.js:175) |
| Session state ping | GET | `/api/state` | [`pingBackendFromUi()`](src/static/app.js:175) |
| Upload model | POST | `/api/model/upload` | [`uploadModel()`](src/static/app.js:52) |
| Run pipeline | POST | `/api/pipeline/run` | [`runPipeline()`](src/static/app.js:115) |
| Analyze + decide | POST | `/api/analyze-and-decide` | [`runPipeline()`](src/static/app.js:115) |

### Additional inline UI calls discovered in [`src/static/index.html`](src/static/index.html)

| UI_ACTION | HTTP_METHOD | ENDPOINT | JS_FUNCTION |
|---|---|---|---|
| Legacy model index upload | POST | `/api/model/index` | [`API.index()`](src/static/index.html:4063) |
| Save profile | POST | `/api/deployment/profile` | [`API.profile()`](src/static/index.html:4073) |
| Clear profile | DELETE | `/api/deployment/profile` | inline handler ([`fetch()`](src/static/index.html:5360)) |
| Model diagnostics | POST | `/api/model/diagnostics` | [`API.diagnostics()`](src/static/index.html:4078) |
| Model analyze | POST | `/api/model/analyze` | [`API.analyze()`](src/static/index.html:4079) |
| Decision recommend | POST | `/api/decision/recommend` | [`API.decision()`](src/static/index.html:4080) |
| Analyze + decide (multipart variant) | POST | `/api/analyze-and-decide` | [`API.analyzeAndDecide()`](src/static/index.html:4087) |
| GPU calibration upload | POST | `/api/calibration/gpu/load` | [`API.gpuCalibration()`](src/static/index.html:4101) |
| State refresh | GET | `/api/state` | [`fetchAndDisplayStrictMode()`](src/static/index.html:5112) |

---

## 2) Backend Execution Trace (full call chain)

### Endpoints implemented and exercised

| endpoint | route_function | service_function | core_function |
|---|---|---|---|
| `GET /api/health` | [`api_health()`](src/api/gui_routes.py:195) | n/a | n/a |
| `GET /api/state` | [`api_get_state()`](src/api/gui_routes.py:199) | [`get_state()`](src/api/gui_state_api.py:91) | session store in-memory logic |
| `POST /api/model/upload` | [`api_upload_model()`](src/api/gui_routes.py:204) | [`handle_model_upload()`](src/api/gui_analysis_api.py:99) | [`validate_onnx_model()`](src/core/model_analysis.py:246), [`compute_model_hash()`](src/core/model_hash.py:11) |
| `POST /api/pipeline/run` | [`api_run_pipeline()`](src/api/gui_routes.py:249) | [`run_model_pipeline()`](src/api/gui_analysis_api.py:212) | [`run_pipeline()`](src/core/pipeline.py:44) → [`validate_onnx_model()`](src/core/model_analysis.py:246) → [`analyze_model()`](src/core/model_analysis.py:273) |
| `POST /api/analyze-and-decide` | [`api_analyze_and_decide()`](src/api/gui_routes.py:292) | [`run_model_pipeline()`](src/api/gui_analysis_api.py:212), [`format_decision_summary()`](src/api/gui_decision_api.py:130) | [`run_pipeline()`](src/core/pipeline.py:44) |

### Endpoints referenced by inline UI but missing in backend (404)

`/api/model/index`, `/api/deployment/profile` (POST/DELETE), `/api/model/diagnostics`, `/api/model/analyze`, `/api/decision/recommend`.

Observed in [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json) (`ui_endpoint_probe`), and route table confirms absence in [`register_routes()`](src/api/gui_routes.py:191).

---

## 3) Runtime instrumentation added (temporary/removable)

Instrumentation was inserted without business-logic changes:

- Upload stage timings:
  - [`UploadResult.upload_time_ms`](src/api/gui_analysis_api.py:39)
  - [`UploadResult.validation_time_ms`](src/api/gui_analysis_api.py:40)
  - measured in [`handle_model_upload()`](src/api/gui_analysis_api.py:99)
- Pipeline stage timings:
  - [`validation_time_ms`](src/core/pipeline.py:42)
  - [`analysis_time_ms`](src/core/pipeline.py:43)
  - [`decision_time_ms`](src/core/pipeline.py:44)
  - measured in [`run_pipeline()`](src/core/pipeline.py:44)
- Request total timing exposed in route responses:
  - upload response in [`api_upload_model()`](src/api/gui_routes.py:204)
  - pipeline response in [`api_run_pipeline()`](src/api/gui_routes.py:249)
  - analyze+decide response in [`api_analyze_and_decide()`](src/api/gui_routes.py:292)

---

## 4) Real execution tests (actual server, real models)

Server was started via uvicorn and exercised by [`tmp_e2e_execution_verification.py`](tmp_e2e_execution_verification.py).

Models used:
- `mnist-8.onnx`
- `resnet50-v2-7.onnx`
- `yolov5s.onnx`

Results source: [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json).

---

## END_TO_END_FLOW

UI  
↓  
API  
↓  
PIPELINE  
↓  
ANALYSIS  
↓  
DECISION

---

## 5) Latency table (real observed)

### External request latency (ms)

| stage | avg | p95 | max |
|---|---:|---:|---:|
| upload (`/api/model/upload`) | 2497.96 | 5953.64 | 5953.64 |
| pipeline (`/api/pipeline/run`) | 807.33 | 1877.65 | 1877.65 |
| analyze+decide (`/api/analyze-and-decide`) | 853.05 | 1985.59 | 1985.59 |

### Internal stage breakdown from pipeline instrumentation (ms)

| stage | avg | p95 | max |
|---|---:|---:|---:|
| validation_time | 283.75 | 730.59 | 730.59 |
| analysis_time | 265.38 | 581.49 | 581.49 |
| decision_time | 0.025 | 0.034 | 0.034 |
| total_request_time (`/api/pipeline/run`) | 804.28 | 1874.72 | 1874.72 |

---

## 6) Resource table

Captured from request probe in [`tmp_e2e_execution_verification.py`](tmp_e2e_execution_verification.py:28).

| resource | observation |
|---|---|
| memory (RSS delta) | ~0.0 MB across sampled requests (very likely underreported due process-sampling granularity) |
| cpu usage | 0.0% snapshots in sampled points (instant snapshots, not sustained CPU profiling) |
| io / payload size | upload response ~277–280 bytes, pipeline response ~531–533 bytes, analyze+decide response ~600–602 bytes |

Note: resource numbers are real captures but low-fidelity; latency metrics are reliable and are the primary performance signal.

---

## 7) Concurrency test

From [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json) (`concurrency`):

| concurrent_requests | avg_latency_ms | p95_latency_ms | p99_latency_ms | error_rate | rate_limited_requests |
|---:|---:|---:|---:|---:|---:|
| 1 | 129.38 | 129.38 | 129.38 | 0.00 | 0 |
| 5 | 45.14 | 56.05 | 56.05 | 0.00 | 0 |
| 20 | 162.83 | 223.72 | 223.77 | 0.10 | 2 |
| 50 | 15.45 | 35.56 | 41.38 | 1.00 | 50 |

Interpretation: at 50 concurrent requests, rate limiting (`429`) dominates all requests.

---

## 8) UI ↔ Backend verification

### Active UI (`app.js`) alignment

- Upload payload in [`uploadModel()`](src/static/app.js:52): multipart field `file`.
- Backend receives `UploadFile` in [`api_upload_model()`](src/api/gui_routes.py:204).
- Pipeline payload in [`runPipeline()`](src/static/app.js:115): JSON `{ model_id }`.
- Backend echo confirms payload match via `request_echo` in [`api_run_pipeline()`](src/api/gui_routes.py:249) and [`api_analyze_and_decide()`](src/api/gui_routes.py:292).
- UI rendered values (`decision`, `confidence`, `diagnostics`, `latency`) are consistent with backend response fields in all tested models (see `ui_display_projection` in [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json)).

### Mismatches detected

1. Inline UI uses `API.index()` pointing to `/api/model/index` ([`src/static/index.html`](src/static/index.html:4063)), but backend implements `/api/model/upload` ([`src/api/gui_routes.py`](src/api/gui_routes.py:204)).
2. Inline UI references profile/diagnostics/analyze/recommend endpoints not implemented in route table, returning `404` in probe.
3. Inline `analyzeAndDecide` sends multipart `formData` ([`src/static/index.html`](src/static/index.html:4087)), while backend expects JSON body model reference ([`ModelRefRequest`](src/api/gui_routes.py:76)).

---

## 9) Detected inefficiencies (with file+line)

1. **Duplicate pipeline execution from UI click path**
   - Parallel calls in [`Promise.all([...])`](src/static/app.js:128): `/api/pipeline/run` + `/api/analyze-and-decide`.
   - Both routes invoke [`run_model_pipeline()`](src/api/gui_routes.py:250) and [`run_model_pipeline()`](src/api/gui_routes.py:277).
   - Core pipeline therefore runs twice for same model request.

2. **Repeated full-file hashing (extra disk reads)**
   - Hashing in [`compute_model_hash()`](src/core/model_hash.py:11) reads file in chunks.
   - Called in upload path ([`handle_model_upload()`](src/api/gui_analysis_api.py:177)), again in pipeline ([`run_pipeline()`](src/core/pipeline.py:88)), and again inside analysis ([`analyze_model()`](src/core/model_analysis.py:334)).

3. **Double validation in common flow**
   - Upload validates in [`handle_model_upload()`](src/api/gui_analysis_api.py:173).
   - Pipeline validates again in [`run_pipeline()`](src/core/pipeline.py:75).
   - If user analyzes immediately after upload, same model is validated again.

4. **Extra upload cleanup pass**
   - `cleanup_upload_storage()` called at upload start ([`handle_model_upload()`](src/api/gui_analysis_api.py:112)) and immediately after successful upload in route ([`api_upload_model()`](src/api/gui_routes.py:216)).

5. **Rate-limit asymmetry**
   - `/api/pipeline/run` has limiter ([`api_run_pipeline()`](src/api/gui_routes.py:249)), but `/api/analyze-and-decide` path has no limiter ([`api_analyze_and_decide()`](src/api/gui_routes.py:292)); this makes duplicate UI calls uneven under load.

---

## 10) Top bottlenecks

1. `resnet50` upload and parse/validation path (largest absolute latency).
2. Repeated validation/hash across upload + pipeline + analysis.
3. Duplicate pipeline execution induced by UI (`runPipeline` calling two backend endpoints for same action).

---

## 11) Optimization targets (highest ROI)

1. [`src/static/app.js`](src/static/app.js:128) — duplicate backend calls in one UI action.  
   **Cost impact:** roughly ~2x compute for analysis path per click.

2. [`src/core/model_analysis.py`](src/core/model_analysis.py:334) and [`src/core/pipeline.py`](src/core/pipeline.py:88) — repeated hashing/validation in hot path.  
   **Cost impact:** substantial on large models (hundreds of ms to seconds in measured runs).

3. [`src/api/gui_analysis_api.py`](src/api/gui_analysis_api.py:112) and [`src/api/gui_routes.py`](src/api/gui_routes.py:216) — duplicate cleanup checks on upload path.  
   **Cost impact:** additional filesystem scanning overhead on every upload.

4. [`src/static/index.html`](src/static/index.html:4061) API map mismatch with current backend routes.  
   **Cost impact:** UI failures / wasted retries (404) and inability to execute intended backend actions.

---

## Artifacts produced

- Raw run data: [`SYSTEM_PERFORMANCE_RAW.json`](SYSTEM_PERFORMANCE_RAW.json)
- Test harness: [`tmp_e2e_execution_verification.py`](tmp_e2e_execution_verification.py)
- This report: [`SYSTEM_PERFORMANCE_REPORT.md`](SYSTEM_PERFORMANCE_REPORT.md)

