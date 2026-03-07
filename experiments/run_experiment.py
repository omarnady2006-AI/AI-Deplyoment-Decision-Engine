"""
run_experiment.py
=================
End-to-end experiment for the Deployment Decision Engine.

Tests:
  - Multiple ONNX model sizes (real binary stubs + real ONNX models)
  - Multiple hardware profiles (EDGE / STANDARD / PRODUCTION / HPC)
  - Concurrency runs (5 & 10 concurrent requests)
  - Repeated identical requests (determinism)

Produces:
  - experiment_results.json
  - experiment_report.txt
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8080"
API_URL = f"{BASE_URL}/api/analyze-and-decide"
TIMEOUT = 300  # seconds – large models take longer

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root (parent of experiments/)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry – map (label, size_mb) → path
# ──────────────────────────────────────────────────────────────────────────────

def _find_or_create_model(label: str, size_mb: int, fmt: str) -> Path:
    """Return an existing model file or create a minimal-valid ONNX stub of the requested size."""
    real_models = {
        # real ONNX models already present in the project root
        "mnist_26kb":    PROJECT_ROOT / "models" / "mnist-8.onnx",      # ~26 KB
        "yolov5s_28mb":  PROJECT_ROOT / "models" / "yolov5s.onnx",      # ~29 MB
        "resnet50_100mb": PROJECT_ROOT / "models" / "resnet50-v2-7.onnx", # ~98 MB
        "ovs_500mb":     PROJECT_ROOT / "models" / "oversized_501mb.onnx", # 500 MB stub
    }

    if label in real_models and real_models[label].exists():
        return real_models[label]

    # Build a synthetic size-padded ONNX binary (minimal valid protobuf)
    dest = PROJECT_ROOT / "models" / f"_synthetic_{label}_{size_mb}mb.onnx"
    if dest.exists():
        return dest

    # Minimal ONNX ModelProto that onnx.load / ONNX Runtime can parse as valid
    # (field 1 = ir_version int64, field 4 = opset_import[].version int64,
    #  field 2 = graph.node empty, field 8 = model_version int64)
    # Then pad to target size with a large repeated string field (field 16).
    target_bytes = size_mb * 1024 * 1024

    # Minimal header: ir_version=8, opset "ai.onnx" version=18
    # Protobuf encoding (manual, no dependency on protobuf library)
    def varint(n: int) -> bytes:
        out = []
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
        return bytes(out)

    def field_varint(tag: int, val: int) -> bytes:
        return varint((tag << 3) | 0) + varint(val)

    def field_len(tag: int, data: bytes) -> bytes:
        return varint((tag << 3) | 2) + varint(len(data)) + data

    # ir_version = 8  (field 1)
    header = field_varint(1, 8)
    # opset_import = [{domain="", version=17}] (field 8 in ModelProto is opset_import)
    opset_entry = field_len(1, b"") + field_varint(2, 17)
    header += field_len(8, opset_entry)
    # graph (field 7) = minimal GraphProto with name
    graph_proto = field_len(1, b"synthetic_graph")
    header += field_len(7, graph_proto)

    # Padding via model_doc_string (field 6, a string)
    pad_needed = max(0, target_bytes - len(header) - 16)
    if pad_needed > 0:
        # Generate pad in 1 MB chunks to avoid huge allocations
        chunk = 1024 * 1024
        remaining = pad_needed
        with open(dest, "wb") as f:
            f.write(header)
            tag_bytes = varint((6 << 3) | 2) + varint(pad_needed)
            f.write(tag_bytes)
            written = 0
            while written < pad_needed:
                to_write = min(chunk, pad_needed - written)
                f.write(b"\x00" * to_write)
                written += to_write
    else:
        dest.write_bytes(header)

    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Hardware profiles
# ──────────────────────────────────────────────────────────────────────────────

HARDWARE_PROFILES = {
    "EDGE": {
        "cpu_cores": 1,
        "ram_gb": 2.0,
        "gpu_available": False,
        "cuda_available": False,
    },
    "STANDARD": {
        "cpu_cores": 4,
        "ram_gb": 8.0,
        "gpu_available": False,
        "cuda_available": False,
    },
    "PRODUCTION": {
        "cpu_cores": 8,
        "ram_gb": 16.0,
        "gpu_available": True,
        "cuda_available": True,
    },
    "HPC": {
        "cpu_cores": 32,
        "ram_gb": 128.0,
        "gpu_available": True,
        "cuda_available": True,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Model test matrix
# ──────────────────────────────────────────────────────────────────────────────

MODEL_MATRIX = [
    # (label, size_mb, fmt)
    ("mnist_26kb",        0,   "ONNX"),   # real ONNX ~26 KB – treated as <1 MB
    ("yolov5s_28mb",      28,  "ONNX"),   # real ONNX ~28 MB
    ("resnet50_100mb",    100, "ONNX"),   # real ONNX ~100 MB
    ("synthetic_10mb",    10,  "ONNX"),   # synthetic stub 10 MB
    ("synthetic_50mb",    50,  "ONNX"),   # synthetic stub 50 MB
    ("synthetic_200mb",   200, "ONNX"),   # synthetic stub 200 MB
    ("synthetic_500mb",   500, "ONNX"),   # synthetic stub 500 MB (uses existing file if present)
    # PyTorch / TF stubs – these will intentionally FAIL analysis (non-ONNX),
    # which is expected and should be reported, not treated as system failure.
    # We include them only as format-diversity markers in the report.
]

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _verify_server() -> bool:
    """Confirm the server is reachable."""
    try:
        r = requests.get(f"{BASE_URL}/api/state", timeout=10)
        return r.status_code == 200
    except Exception as exc:
        print(f"[ERROR] Server not reachable at {BASE_URL}: {exc}")
        return False


def _post_analyze(model_path: Path, hw: dict) -> dict[str, Any]:
    """POST to /api/analyze-and-decide and return structured result."""
    t0 = time.perf_counter()
    try:
        with open(model_path, "rb") as fh:
            files = {"file": (model_path.name, fh, "application/octet-stream")}
            data = {k: str(v) for k, v in hw.items()}
            resp = requests.post(API_URL, files=files, data=data, timeout=TIMEOUT)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}

        return {
            "http_status": resp.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
            "body": body,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "http_status": -1,
            "elapsed_ms": round(elapsed_ms, 2),
            "error": str(exc),
            "body": {},
        }


def _extract_key_fields(result: dict) -> dict:
    """Pull the key fields we care about from a raw API result."""
    body = result.get("body", {})
    rd = body.get("runtime_details", {})
    summary = body.get("summary", {})
    return {
        "http_status":   result.get("http_status"),
        "elapsed_ms":    result.get("elapsed_ms"),
        "success":       body.get("success"),
        "error":         result.get("error") or body.get("detail"),
        "risk_score":    body.get("risk_score"),
        "risk_level":    body.get("risk_level"),
        "decision":      rd.get("status"),
        "confidence":    rd.get("confidence"),
        "reasons":       rd.get("reasons", []),
        "best_runtime":  rd.get("runtime"),
        # From evaluations / best eval
        "latency_ms":    _best_latency(body),
        "memory_mb":     _best_memory(body),
    }


def _best_latency(body: dict) -> float | None:
    evals = body.get("evaluations", [])
    for ev in evals:
        if ev.get("execution_success") and ev.get("predicted_latency_ms") not in (None, 0.0):
            return ev["predicted_latency_ms"]
    return None


def _best_memory(body: dict) -> float | None:
    evals = body.get("evaluations", [])
    for ev in evals:
        if ev.get("execution_success") and ev.get("memory_usage_mb") not in (None, 0.0):
            return ev["memory_usage_mb"]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment() -> dict:
    print("=" * 70)
    print("DEPLOYMENT DECISION ENGINE — FULL EXPERIMENT")
    print(f"API endpoint: {API_URL}")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print("=" * 70)

    # ── Step 1: verify server ──────────────────────────────────────────────
    print("\n[STEP 1] Verifying server…")
    if not _verify_server():
        sys.exit("FATAL: Server not reachable. Aborting.")
    print("  ✓ Server is reachable")

    # ── Step 2: build model paths ──────────────────────────────────────────
    print("\n[STEP 2] Preparing model files…")
    model_paths: dict[str, Path] = {}
    for label, size_mb, fmt in MODEL_MATRIX:
        p = _find_or_create_model(label, size_mb, fmt)
        model_paths[label] = p
        print(f"  {label:30s}  {p.stat().st_size / (1024**2):7.2f} MB  {p.name}")

    # ── Step 3+4: model × hardware matrix ─────────────────────────────────
    print("\n[STEP 3+4] Running model × hardware matrix…")
    matrix_results: list[dict] = []

    for label, size_mb, fmt in MODEL_MATRIX:
        model_path = model_paths[label]
        print(f"\n  Model: {label}")
        for hw_name, hw_params in HARDWARE_PROFILES.items():
            print(f"    ↳ {hw_name:12s} … ", end="", flush=True)
            raw = _post_analyze(model_path, hw_params)
            kf = _extract_key_fields(raw)
            record = {
                "model":           label,
                "model_size_mb":   size_mb,
                "model_format":    fmt,
                "hardware":        hw_name,
                "hw_params":       hw_params,
                **kf,
            }
            matrix_results.append(record)
            status_str = kf.get("decision") or f"HTTP-{kf.get('http_status')}"
            print(f"{status_str}  risk={kf.get('risk_score')} conf={kf.get('confidence')}")

    # ── Step 5: concurrency & determinism ─────────────────────────────────
    print("\n[STEP 5] Concurrency & determinism tests…")
    concurrency_results: list[dict] = []

    # Use mnist (smallest, fastest) for concurrency
    _conc_model = model_paths["mnist_26kb"]
    _conc_hw    = HARDWARE_PROFILES["STANDARD"]

    for n_workers in (5, 10):
        print(f"  → {n_workers} concurrent requests … ", end="", flush=True)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_post_analyze, _conc_model, _conc_hw) for _ in range(n_workers)]
            raw_results = [f.result() for f in futures]
        wall_ms = (time.perf_counter() - t0) * 1000
        kf_list = [_extract_key_fields(r) for r in raw_results]
        success_count = sum(1 for kf in kf_list if kf.get("success") or kf.get("http_status") == 200)
        risk_values = [kf["risk_score"] for kf in kf_list if kf.get("risk_score") is not None]
        decisions = [kf["decision"] for kf in kf_list if kf.get("decision")]
        concurrency_results.append({
            "concurrency": n_workers,
            "wall_ms": round(wall_ms, 2),
            "success_count": success_count,
            "risk_values": risk_values,
            "decisions": decisions,
            "all_same_decision": len(set(decisions)) <= 1,
            "risk_variance": _variance(risk_values),
        })
        print(f"{success_count}/{n_workers} ok  risk_variance={_variance(risk_values):.6f}")

    # Repeat identical request 5× (determinism)
    print("  → 5 repeated identical requests … ", end="", flush=True)
    repeat_results = []
    for _ in range(5):
        raw = _post_analyze(_conc_model, _conc_hw)
        repeat_results.append(_extract_key_fields(raw))
    risk_vals = [r["risk_score"] for r in repeat_results if r.get("risk_score") is not None]
    dec_vals  = [r["decision"] for r in repeat_results if r.get("decision")]
    repeat_summary = {
        "concurrency": 1,
        "repetitions": 5,
        "risk_values": risk_vals,
        "decisions":   dec_vals,
        "all_same_decision": len(set(dec_vals)) <= 1,
        "risk_variance": _variance(risk_vals),
    }
    concurrency_results.append(repeat_summary)
    print(f"decision_stable={repeat_summary['all_same_decision']}  risk_var={repeat_summary['risk_variance']:.6f}")

    results = {
        "experiment_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "api_endpoint": API_URL,
        "matrix_results": matrix_results,
        "concurrency_results": concurrency_results,
    }
    return results


def _variance(vals: list) -> float:
    if not vals:
        return 0.0
    floats = [float(v) for v in vals if v is not None]
    if not floats:
        return 0.0
    mean = sum(floats) / len(floats)
    return sum((x - mean) ** 2 for x in floats) / len(floats)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis & bug detection
# ──────────────────────────────────────────────────────────────────────────────

def analyze_results(results: dict) -> dict:
    """Detect logical bugs in the result matrix."""
    matrix = results["matrix_results"]
    bugs: list[str] = []
    warnings: list[str] = []

    HW_ORDER = ["EDGE", "STANDARD", "PRODUCTION", "HPC"]

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for rec in matrix:
        by_model.setdefault(rec["model"], []).append(rec)

    # ── CONFIDENCE_CONSTANT_BUG ───────────────────────────────────────────
    all_confidences = [r["confidence"] for r in matrix if r.get("confidence") is not None]
    if all_confidences:
        conf_var = _variance(all_confidences)
        if conf_var < 1e-6:
            bugs.append(
                f"CONFIDENCE_CONSTANT_BUG: All confidence values are identical "
                f"({all_confidences[0]:.4f}). Confidence must vary by hardware profile."
            )
        else:
            # Also check per-model — confidence should differ across hardware profiles
            for model_label, recs in by_model.items():
                conf_vals = [r["confidence"] for r in recs if r.get("confidence") is not None]
                if len(conf_vals) >= 2 and _variance(conf_vals) < 1e-6:
                    bugs.append(
                        f"CONFIDENCE_CONSTANT_BUG [{model_label}]: confidence identical "
                        f"across all hardware profiles ({conf_vals[0]:.4f})."
                    )

    # ── RISK_SCALING_BUG ─────────────────────────────────────────────────
    # EDGE should generally have higher (or equal) risk than STANDARD, etc.
    for model_label, recs in by_model.items():
        risk_by_hw = {r["hardware"]: r["risk_score"] for r in recs if r.get("risk_score") is not None}
        # Only check pairs where both are present
        ordered_risk = [(hw, risk_by_hw[hw]) for hw in HW_ORDER if hw in risk_by_hw]
        for i in range(len(ordered_risk) - 1):
            hw_lo, r_lo = ordered_risk[i]
            hw_hi, r_hi = ordered_risk[i + 1]
            if r_lo < r_hi:
                # More constrained hw should not have LOWER risk than a more capable one
                bugs.append(
                    f"RISK_SCALING_BUG [{model_label}]: "
                    f"{hw_lo} risk={r_lo:.4f} < {hw_hi} risk={r_hi:.4f} — "
                    f"risk should decrease (or stay flat) as hardware improves."
                )

    # ── MODEL_SCALING_BUG ────────────────────────────────────────────────
    # Larger models should have higher (or equal) latency on same hardware
    real_onnx = [r for r in matrix if r["model_format"] == "ONNX"
                 and r.get("latency_ms") is not None
                 and r.get("latency_ms", 0) > 0]
    for hw_name in HW_ORDER:
        hw_recs = [(r["model_size_mb"], r["latency_ms"]) for r in real_onnx
                   if r["hardware"] == hw_name and r.get("latency_ms")]
        hw_recs.sort(key=lambda x: x[0])
        if len(hw_recs) >= 2:
            # Check large model is not faster than small model
            # Allow some tolerance (2×) for noise
            small_size, small_lat = hw_recs[0]
            large_size, large_lat = hw_recs[-1]
            if large_size > small_size * 2 and large_lat < small_lat * 0.5:
                bugs.append(
                    f"MODEL_SCALING_BUG [{hw_name}]: "
                    f"model {large_size}MB latency={large_lat:.2f}ms "
                    f"is much faster than model {small_size}MB latency={small_lat:.2f}ms — "
                    f"expected larger models to have higher latency."
                )

    # ── DECISION_LOGIC_BUG ───────────────────────────────────────────────
    # risk < 3.8   → ALLOW  → decision should not be REJECTED (unless SLA violation)
    # risk >= 8.22 → BLOCK  → decision should be REJECTED
    DEC_MAP = {
        "ALLOW": "APPROVED",
        "BLOCK": "REJECTED",
        "ALLOW_WITH_CONDITIONS": "CONDITIONAL_APPROVAL",
    }
    for rec in matrix:
        risk = rec.get("risk_score")
        decision = rec.get("decision")
        if risk is None or decision is None:
            continue
        if risk < 3.8 and decision == "REJECTED":
            # Only a bug if there's no SLA reason
            if not rec.get("reasons"):
                bugs.append(
                    f"DECISION_LOGIC_BUG [{rec['model']} / {rec['hardware']}]: "
                    f"risk={risk:.4f} < 3.8 but decision=REJECTED with no SLA reasons."
                )
        if risk >= 8.22 and decision != "REJECTED":
            bugs.append(
                f"DECISION_LOGIC_BUG [{rec['model']} / {rec['hardware']}]: "
                f"risk={risk:.4f} >= 8.22 but decision={decision} (expected REJECTED)."
            )

    # ── API_UI_MISMATCH ──────────────────────────────────────────────────
    # We only have the API (no live UI scraping); verify internal consistency:
    # body["risk_score"] must equal runtime_details["risk_score"] if both present
    for rec in matrix:
        body = rec  # already extracted; no nested body here
        # (extraction already unified; just sanity-check risk_score vs risk_level)
        risk = rec.get("risk_score")
        rl   = rec.get("risk_level")
        if risk is not None and rl is not None:
            expected_rl = (
                "HIGH_RISK"   if risk >= 8.2 else
                "MEDIUM_RISK" if risk >= 3.8 else
                "LOW_RISK"
            )
            if rl != expected_rl:
                bugs.append(
                    f"API_UI_MISMATCH [{rec['model']} / {rec['hardware']}]: "
                    f"risk_score={risk:.4f} → expected risk_level={expected_rl} "
                    f"but got {rl}."
                )

    # ── Concurrency determinism ───────────────────────────────────────────
    for cr in results["concurrency_results"]:
        if cr.get("risk_variance", 0) > 0.1:
            warnings.append(
                f"NON_DETERMINISM [{cr.get('concurrency', cr.get('repetitions'))} workers]: "
                f"risk variance={cr['risk_variance']:.6f} — results may be non-deterministic."
            )
        if not cr.get("all_same_decision", True):
            bugs.append(
                f"NON_DETERMINISM_BUG [{cr.get('concurrency', cr.get('repetitions'))} workers]: "
                f"Different decisions returned for identical inputs: {set(cr['decisions'])}."
            )

    # ── Final verdict ─────────────────────────────────────────────────────
    verdict = "BUGS_DETECTED" if bugs else "SYSTEM_VALIDATED"
    return {
        "verdict": verdict,
        "bugs": bugs,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results: dict, analysis: dict) -> str:
    lines: list[str] = []
    sep = "=" * 70
    thin = "-" * 70

    lines.append(sep)
    lines.append("DEPLOYMENT DECISION ENGINE — EXPERIMENT REPORT")
    lines.append(f"Generated: {results['experiment_timestamp']}")
    lines.append(f"API:       {results['api_endpoint']}")
    lines.append(sep)

    # ── Model × Hardware matrix ───────────────────────────────────────────
    lines.append("\n### MODEL × HARDWARE RESULT MATRIX\n")
    hdr = f"{'Model':<28} {'HW':<12} {'Risk':>7} {'RiskLvl':<12} {'Decision':<22} {'Conf':>6} {'Lat(ms)':>10} {'Mem(MB)':>9}"
    lines.append(hdr)
    lines.append(thin)
    for rec in results["matrix_results"]:
        risk_str = f"{rec['risk_score']:.4f}" if rec.get("risk_score") is not None else "N/A"
        conf_str = f"{rec['confidence']:.4f}" if rec.get("confidence") is not None else "N/A"
        lat_str  = f"{rec['latency_ms']:.2f}" if rec.get("latency_ms") is not None else "N/A"
        mem_str  = f"{rec['memory_mb']:.2f}"  if rec.get("memory_mb")  is not None else "N/A"
        lines.append(
            f"{rec['model']:<28} {rec['hardware']:<12} {risk_str:>7} "
            f"{(rec.get('risk_level') or 'N/A'):<12} {(rec.get('decision') or 'N/A'):<22} "
            f"{conf_str:>6} {lat_str:>10} {mem_str:>9}"
        )

    # ── Concurrency ───────────────────────────────────────────────────────
    lines.append(f"\n\n### CONCURRENCY & DETERMINISM RESULTS\n")
    for cr in results["concurrency_results"]:
        n = cr.get("concurrency", cr.get("repetitions", "?"))
        label = f"{n} workers" if cr.get("concurrency") else f"{cr.get('repetitions')} repeats"
        decisions = cr.get("decisions", [])
        risk_vals = cr.get("risk_values", [])
        lines.append(f"  {label}:")
        lines.append(f"    success_count : {cr.get('success_count', 'N/A')}")
        lines.append(f"    decisions     : {decisions}")
        lines.append(f"    all_same_dec  : {cr.get('all_same_decision')}")
        lines.append(f"    risk_values   : {[round(r, 4) for r in risk_vals if r]}")
        lines.append(f"    risk_variance : {cr.get('risk_variance', 0):.8f}")
        lines.append("")

    # ── Analysis & bugs ───────────────────────────────────────────────────
    lines.append(f"\n### BUG ANALYSIS\n")
    if analysis["bugs"]:
        lines.append(f"  ⚠  {len(analysis['bugs'])} BUG(S) DETECTED:\n")
        for b in analysis["bugs"]:
            lines.append(f"  [BUG]  {b}")
    else:
        lines.append("  ✓  No bugs detected.")

    if analysis["warnings"]:
        lines.append(f"\n  ⚠  {len(analysis['warnings'])} WARNING(S):\n")
        for w in analysis["warnings"]:
            lines.append(f"  [WARN] {w}")

    # ── Verdict ───────────────────────────────────────────────────────────
    lines.append(f"\n{sep}")
    verdict = analysis["verdict"]
    lines.append(f"FINAL VERDICT: {verdict}")
    if verdict == "BUGS_DETECTED":
        lines.append("\nDetailed bug list:")
        for b in analysis["bugs"]:
            lines.append(f"  • {b}")
    lines.append(sep)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_results = run_experiment()

    print("\n[STEP 6] Analyzing results…")
    analysis = analyze_results(raw_results)

    full_output = {**raw_results, "analysis": analysis}

    # Save JSON
    json_path = Path(__file__).resolve().parent / "experiment_results.json"
    json_path.write_text(json.dumps(full_output, indent=2, default=str))
    print(f"  ✓ experiment_results.json written ({json_path.stat().st_size:,} bytes)")

    # Save report
    report_text = generate_report(raw_results, analysis)
    report_path = Path(__file__).resolve().parent / "experiment_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  ✓ experiment_report.txt written ({report_path.stat().st_size:,} bytes)")

    print("\n" + report_text)
