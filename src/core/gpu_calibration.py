"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

import datetime
from typing import Dict, FrozenSet, List, Optional, Tuple

_VALID_MODEL_SIZES: FrozenSet[str] = frozenset({"small", "medium", "large"})
_VALID_BATCH_SIZES: FrozenSet[int] = frozenset({1, 8, 32})
_REQUIRED_BENCHMARK_FIELDS: Tuple[str, ...] = (
    "latency_avg_ms",
    "latency_p95_ms",
    "throughput_samples_per_sec",
    "peak_memory_rss_delta_mb",
)

_CALIBRATION_KEY = Tuple[str, int, str]


class GPUCalibrationError(Exception):
    pass


class GPUCalibrationEntry:
    __slots__ = (
        "model_size",
        "batch_size",
        "provider",
        "latency_avg_ms",
        "latency_p95_ms",
        "throughput_samples_per_sec",
        "peak_memory_rss_delta_mb",
        "gpu_peak_memory_mb",
        "gpu_memory_growth_mb",
        "rank",
    )

    def __init__(
        self,
        model_size: str,
        batch_size: int,
        provider: str,
        latency_avg_ms: float,
        latency_p95_ms: float,
        throughput_samples_per_sec: float,
        peak_memory_rss_delta_mb: float,
        gpu_peak_memory_mb: Optional[float],
        gpu_memory_growth_mb: Optional[float],
        rank: Optional[int],
    ) -> None:
        self.model_size = model_size
        self.batch_size = batch_size
        self.provider = provider
        self.latency_avg_ms = latency_avg_ms
        self.latency_p95_ms = latency_p95_ms
        self.throughput_samples_per_sec = throughput_samples_per_sec
        self.peak_memory_rss_delta_mb = peak_memory_rss_delta_mb
        self.gpu_peak_memory_mb = gpu_peak_memory_mb
        self.gpu_memory_growth_mb = gpu_memory_growth_mb
        self.rank = rank

    def to_dict(self) -> dict:
        return {
            "model_size": self.model_size,
            "batch_size": self.batch_size,
            "provider": self.provider,
            "latency_avg_ms": self.latency_avg_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "peak_memory_rss_delta_mb": self.peak_memory_rss_delta_mb,
            "gpu_peak_memory_mb": self.gpu_peak_memory_mb,
            "gpu_memory_growth_mb": self.gpu_memory_growth_mb,
            "rank": self.rank,
        }


class GPUCalibrationProfile:
    def __init__(self, raw_json: dict) -> None:
        self._loaded_at: str = datetime.datetime.utcnow().isoformat() + "Z"
        self._environment: dict = {}
        self._entries: Dict[_CALIBRATION_KEY, GPUCalibrationEntry] = {}
        self._providers_detected: List[str] = []
        self._models_detected: List[str] = []
        self._parse(raw_json)

    def _parse(self, raw_json: dict) -> None:
        if not isinstance(raw_json, dict):
            raise GPUCalibrationError("Root JSON must be a dict.")

        # ------------------------------------------------------------------ #
        # Schema validation guardrail — fail fast before any parsing begins   #
        # ------------------------------------------------------------------ #
        try:
            from src.core.calibration_schema_validator import validate_calibration_schema, CalibrationSchemaError
            validate_calibration_schema(raw_json)
        except CalibrationSchemaError as schema_exc:
            raise GPUCalibrationError(
                f"Calibration JSON failed schema validation: {schema_exc}"
            ) from schema_exc
        # ------------------------------------------------------------------ #

        if "environment" not in raw_json:
            raise GPUCalibrationError("Missing required key: 'environment'.")
        if "benchmarks" not in raw_json:
            raise GPUCalibrationError("Missing required key: 'benchmarks'.")
        if "comparative_ranking" not in raw_json:
            raise GPUCalibrationError("Missing required key: 'comparative_ranking'.")

        self._environment = raw_json["environment"]

        benchmarks = raw_json["benchmarks"]
        if not isinstance(benchmarks, dict):
            raise GPUCalibrationError("'benchmarks' must be a dict.")

        providers_seen: List[str] = []
        models_seen: List[str] = []

        for model_size, batch_dict in benchmarks.items():
            if model_size not in _VALID_MODEL_SIZES:
                raise GPUCalibrationError(
                    f"Invalid model_size '{model_size}'. Must be one of {sorted(_VALID_MODEL_SIZES)}."
                )
            if not isinstance(batch_dict, dict):
                raise GPUCalibrationError(
                    f"benchmarks['{model_size}'] must be a dict of batch keys."
                )
            if model_size not in models_seen:
                models_seen.append(model_size)

            for batch_key, provider_dict in batch_dict.items():
                batch_size = self._parse_batch_key(batch_key)
                if not isinstance(provider_dict, dict):
                    raise GPUCalibrationError(
                        f"benchmarks['{model_size}']['{batch_key}'] must be a dict keyed by provider."
                    )

                for provider_name, result in provider_dict.items():
                    if not isinstance(result, dict):
                        raise GPUCalibrationError(
                            f"Benchmark result for provider '{provider_name}' must be a dict."
                        )

                    if "error" in result:
                        continue

                    for field in _REQUIRED_BENCHMARK_FIELDS:
                        if field not in result:
                            raise GPUCalibrationError(
                                f"Missing field '{field}' in benchmarks"
                                f"['{model_size}']['{batch_key}']['{provider_name}']."
                            )

                    entry = GPUCalibrationEntry(
                        model_size=model_size,
                        batch_size=batch_size,
                        provider=provider_name,
                        latency_avg_ms=float(result["latency_avg_ms"]),
                        latency_p95_ms=float(result["latency_p95_ms"]),
                        throughput_samples_per_sec=float(result["throughput_samples_per_sec"]),
                        peak_memory_rss_delta_mb=float(result["peak_memory_rss_delta_mb"]),
                        gpu_peak_memory_mb=(
                            float(result["gpu_peak_memory_mb"])
                            if result.get("gpu_peak_memory_mb") is not None
                            else None
                        ),
                        gpu_memory_growth_mb=(
                            float(result["gpu_memory_growth_mb"])
                            if result.get("gpu_memory_growth_mb") is not None
                            else None
                        ),
                        rank=result.get("rank"),
                    )

                    key: _CALIBRATION_KEY = (model_size, batch_size, provider_name)
                    self._entries[key] = entry

                    if provider_name not in providers_seen:
                        providers_seen.append(provider_name)

        if not self._entries:
            raise GPUCalibrationError(
                "Calibration JSON contained no valid (non-error) benchmark entries."
            )

        self._providers_detected = sorted(providers_seen)
        self._models_detected = sorted(models_seen)

    @staticmethod
    def _parse_batch_key(batch_key: str) -> int:
        if not batch_key.startswith("batch_"):
            raise GPUCalibrationError(
                f"Batch key '{batch_key}' does not match expected format 'batch_<N>'."
            )
        raw = batch_key[len("batch_"):]
        if not raw.isdigit():
            raise GPUCalibrationError(
                f"Batch key '{batch_key}' does not contain a valid integer batch size."
            )
        bs = int(raw)
        if bs not in _VALID_BATCH_SIZES:
            raise GPUCalibrationError(
                f"Batch size {bs} is not a supported calibration batch size. "
                f"Supported: {sorted(_VALID_BATCH_SIZES)}."
            )
        return bs

    def get_reference(
        self, model_size: str, batch_size: int, provider: str
    ) -> dict:
        if model_size not in _VALID_MODEL_SIZES:
            raise GPUCalibrationError(
                f"Invalid model_size '{model_size}'. Must be one of {sorted(_VALID_MODEL_SIZES)}."
            )
        if batch_size not in _VALID_BATCH_SIZES:
            raise GPUCalibrationError(
                f"Batch size {batch_size} not in supported set {sorted(_VALID_BATCH_SIZES)}."
            )
        key: _CALIBRATION_KEY = (model_size, batch_size, provider)
        entry = self._entries.get(key)
        if entry is None:
            raise GPUCalibrationError(
                f"No calibration reference for model_size='{model_size}', "
                f"batch_size={batch_size}, provider='{provider}'."
            )
        return entry.to_dict()

    def get_best_provider(
        self,
        model_size: str,
        batch_size: int,
        max_gpu_memory_mb: Optional[float] = None,
        target_latency_ms: Optional[float] = None,
    ) -> Optional[str]:
        candidates: List[GPUCalibrationEntry] = []

        for provider in self._providers_detected:
            key: _CALIBRATION_KEY = (model_size, batch_size, provider)
            entry = self._entries.get(key)
            if entry is None:
                continue

            if (
                max_gpu_memory_mb is not None
                and entry.gpu_peak_memory_mb is not None
                and entry.gpu_peak_memory_mb > max_gpu_memory_mb
            ):
                continue

            if (
                target_latency_ms is not None
                and entry.latency_avg_ms > target_latency_ms
            ):
                continue

            candidates.append(entry)

        if not candidates:
            return None

        candidates.sort(
            key=lambda e: (
                e.latency_avg_ms,
                e.latency_p95_ms,
                e.peak_memory_rss_delta_mb,
                e.gpu_peak_memory_mb if e.gpu_peak_memory_mb is not None else float("inf"),
            )
        )
        return candidates[0].provider

    def all_providers_for(self, model_size: str, batch_size: int) -> List[dict]:
        results: List[dict] = []
        for provider in self._providers_detected:
            key: _CALIBRATION_KEY = (model_size, batch_size, provider)
            entry = self._entries.get(key)
            if entry is not None:
                results.append(entry.to_dict())
        results.sort(key=lambda r: r["latency_avg_ms"])
        return results

    @property
    def environment(self) -> dict:
        return dict(self._environment)

    @property
    def providers_detected(self) -> List[str]:
        return list(self._providers_detected)

    @property
    def models_detected(self) -> List[str]:
        return list(self._models_detected)

    @property
    def loaded_at(self) -> str:
        return self._loaded_at

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        return {
            "loaded_at": self._loaded_at,
            "environment": self._environment,
            "providers_detected": self._providers_detected,
            "models_detected": self._models_detected,
            "entry_count": self.entry_count,
        }
