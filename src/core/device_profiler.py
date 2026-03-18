"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Device profiling for performance-based deployment decisions."""
import logging
import platform
import subprocess
import tempfile
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import sqlite3
import threading

from src.core.optional_dependencies import (
    PSUTIL, ONNXRUNTIME, TENSORRT, OPENVINO, TENSORFLOW, NUMPY, ONNX
)

logger = logging.getLogger(__name__)


@dataclass
class DeviceProfile:
    """Profile of a device's compute capabilities."""
    device_fingerprint: str
    device_class: str  # cpu, cuda, rocm, npu, etc.
    
    # CPU info
    cpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_freq_mhz: Optional[float] = None
    
    # GPU info
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    gpu_compute_capability: Optional[str] = None
    
    # Memory
    system_memory_mb: Optional[int] = None
    
    # Benchmark results
    matmul_throughput: Optional[float] = None  # GFLOPS
    memory_bandwidth: Optional[float] = None  # GB/s
    fp16_vs_fp32_ratio: Optional[float] = None
    kernel_launch_overhead_us: Optional[float] = None
    
    # Platform info
    platform: Optional[str] = None
    platform_version: Optional[str] = None
    python_version: Optional[str] = None
    
    # Runtime availability
    onnxruntime_available: bool = False
    tensorrt_available: bool = False
    openvino_available: bool = False
    tflite_available: bool = False
    
    # Timestamp
    profile_timestamp: Optional[float] = None


class DeviceProfiler:
    """Profile device capabilities for deployment decisions."""
    
    def __init__(self, cache_path: str = "device_profiles.db"):
        self.cache_path = Path(cache_path)
        self._local = threading.local()
        self._init_cache()
        self._current_profile: Optional[DeviceProfile] = None
    
    def _init_cache(self):
        """Initialize profile cache database."""
        if not self.cache_path.exists():
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE device_profiles (
                    device_fingerprint TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
    
    def get_device_fingerprint(self) -> str:
        """Generate unique device fingerprint."""
        data = []
        
        try:
            processor = platform.processor()
            if processor:
                data.append(processor)
        except (OSError, ValueError) as e:
            logger.warning("exception_occurred", exc_info=True)
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data.append(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning("exception_occurred", exc_info=True)
        
        if PSUTIL.available:
            try:
                data.append(str(PSUTIL.module.virtual_memory().total))
            except (OSError, RuntimeError) as e:
                logger.warning("exception_occurred", exc_info=True)
        
        data.append(platform.platform())
        
        fingerprint = hashlib.sha256("|".join(data).encode()).hexdigest()[:16]
        return fingerprint
    
    def profile_current_device(self, force_refresh: bool = False) -> DeviceProfile:
        """Profile the current device."""
        fingerprint = self.get_device_fingerprint()
        
        # Check cache
        if not force_refresh:
            cached = self._load_cached_profile(fingerprint)
            if cached:
                self._current_profile = cached
                return cached
        
        # Run benchmarks
        profile = DeviceProfile(
            device_fingerprint=fingerprint,
            device_class=self._detect_device_class(),
            profile_timestamp=time.time()
        )
        
        # System info
        profile.platform = platform.system()
        profile.platform_version = platform.version()
        profile.python_version = platform.python_version()
        
        # CPU info
        profile.cpu_name = platform.processor()
        if PSUTIL.available:
            try:
                profile.cpu_cores = PSUTIL.module.cpu_count(logical=False)
                profile.cpu_freq_mhz = PSUTIL.module.cpu_freq().max if PSUTIL.module.cpu_freq() else None
                profile.system_memory_mb = PSUTIL.module.virtual_memory().total // (1024 * 1024)
            except (OSError, RuntimeError, AttributeError) as e:
                logger.warning("exception_occurred", exc_info=True)
        
        # GPU info
        gpu_info = self._get_gpu_info()
        profile.gpu_name = gpu_info.get("name")
        profile.gpu_memory_mb = gpu_info.get("memory_mb")
        profile.gpu_compute_capability = gpu_info.get("compute_capability")
        
        # Runtime availability
        profile.onnxruntime_available = self._check_onnxruntime()
        profile.tensorrt_available = self._check_tensorrt()
        profile.openvino_available = self._check_openvino()
        profile.tflite_available = self._check_tflite()
        
        # Run benchmarks
        if profile.onnxruntime_available:
            benchmark_results = self._run_benchmarks()
            profile.matmul_throughput = benchmark_results.get("matmul_gflops")
            profile.memory_bandwidth = benchmark_results.get("memory_bandwidth_gbps")
            profile.fp16_vs_fp32_ratio = benchmark_results.get("fp16_fp32_ratio")
            profile.kernel_launch_overhead_us = benchmark_results.get("kernel_overhead_us")
        
        # Cache profile
        self._save_profile(profile)
        self._current_profile = profile
        
        return profile
    
    def _detect_device_class(self) -> str:
        """Detect the primary compute device class."""
        from src.core.optional_dependencies import TORCH, TENSORFLOW
        
        if TORCH.available and hasattr(TORCH.module, 'cuda') and TORCH.module.cuda.is_available():
            return "cuda"
        
        if TENSORFLOW.available:
            gpus = TENSORFLOW.module.config.list_physical_devices('GPU')
            if gpus:
                return "gpu"
        
        return "cpu"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        info = {}
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    info["name"] = parts[0]
                    info["memory_mb"] = int(parts[1])
                if len(parts) >= 3:
                    info["compute_capability"] = parts[2]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError) as e:
            logger.warning("exception_occurred", exc_info=True)
        
        return info
    
    def _check_onnxruntime(self) -> bool:
        """Check if ONNX Runtime is available."""
        return ONNXRUNTIME.available
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        return TENSORRT.available
    
    def _check_openvino(self) -> bool:
        """Check if OpenVINO is available."""
        return OPENVINO.available
    
    def _check_tflite(self) -> bool:
        """Check if TFLite is available."""
        return TENSORFLOW.available
    
    def _run_benchmarks(self) -> Dict[str, float]:
        """Run microbenchmarks to measure performance."""
        results = {}
        
        if not NUMPY.available or not ONNXRUNTIME.available:
            logger.warning("Benchmark dependencies not available")
            return results
        
        try:
            results["matmul_gflops"] = self._benchmark_matmul()
            results["memory_bandwidth_gbps"] = self._benchmark_memory_bandwidth()
            results["fp16_fp32_ratio"] = self._benchmark_fp16_fp32_ratio()
            results["kernel_overhead_us"] = self._benchmark_kernel_overhead()
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Benchmark execution failed: {e}")
        
        return results
    
    def _benchmark_matmul(self, size: int = 2048, runs: int = 10) -> Optional[float]:
        """Benchmark matrix multiplication throughput."""
        if not NUMPY.available:
            return None
        
        try:
            np = NUMPY.module
            
            # Create test data
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                _ = np.dot(A, B)
            
            # Time runs
            timings = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = np.dot(A, B)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
            
            avg_time = np.mean(timings)
            
            # Compute GFLOPS: 2 * n^3 for matmul
            flops = 2 * (size ** 3)
            gflops = flops / (avg_time * 1e9)
            
            return float(gflops)
            
        except (RuntimeError, ValueError):
            return None
    
    def _benchmark_memory_bandwidth(self, size_mb: int = 100, runs: int = 10) -> Optional[float]:
        """Benchmark memory bandwidth."""
        if not NUMPY.available:
            return None
        
        try:
            np = NUMPY.module
            
            # Create large array
            arr = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                _ = arr.copy()
            
            # Time runs
            timings = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = arr.copy()
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
            
            avg_time = np.mean(timings)
            bandwidth_gbps = (size_mb / 1024) / avg_time
            
            return float(bandwidth_gbps)
            
        except (RuntimeError, ValueError):
            return None
    
    def _benchmark_fp16_fp32_ratio(self, size: int = 1024, runs: int = 10) -> Optional[float]:
        """Benchmark FP16 vs FP32 performance ratio."""
        if not NUMPY.available:
            return None
        
        try:
            np = NUMPY.module
            
            A_fp32 = np.random.randn(size, size).astype(np.float32)
            B_fp32 = np.random.randn(size, size).astype(np.float32)
            
            # FP32 timing
            timings_fp32 = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = np.dot(A_fp32, B_fp32)
                timings_fp32.append(time.perf_counter() - start)
            
            avg_fp32 = np.mean(timings_fp32)
            
            # FP16 timing (if supported)
            try:
                A_fp16 = A_fp32.astype(np.float16)
                B_fp16 = B_fp32.astype(np.float16)
                
                timings_fp16 = []
                for _ in range(runs):
                    start = time.perf_counter()
                    _ = np.dot(A_fp16, B_fp16)
                    timings_fp16.append(time.perf_counter() - start)
                
                avg_fp16 = np.mean(timings_fp16)
                ratio = avg_fp32 / avg_fp16
                
                return float(ratio)
            except (RuntimeError, ValueError):
                return 1.0
                
        except (RuntimeError, ValueError):
            return None
    
    def _benchmark_kernel_overhead(self, runs: int = 100) -> Optional[float]:
        """Benchmark kernel launch overhead."""
        if not NUMPY.available or not ONNXRUNTIME.available or not ONNX.available:
            return None
        
        try:
            np = NUMPY.module
            ort = ONNXRUNTIME.module
            onnx = ONNX.module
            from onnx import helper, TensorProto
            
            input_a = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1])
            input_b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1])
            output = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1])
            
            add_node = helper.make_node('Add', ['A', 'B'], ['C'])
            
            graph = helper.make_graph(
                [add_node],
                'simple_add',
                [input_a, input_b],
                [output]
            )
            
            model = helper.make_model(graph)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                temp_path = f.name
            
            onnx.save(model, temp_path)
            
            sess = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])
            
            timings = []
            for _ in range(runs):
                start = time.perf_counter()
                sess.run(None, {'A': [1.0], 'B': [2.0]})
                timings.append(time.perf_counter() - start)
            
            avg_us = float(np.mean(timings)) * 1e6
            
            Path(temp_path).unlink(missing_ok=True)
            
            return avg_us
            
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug(f"Kernel overhead benchmark unavailable: {e}")
            return None
    
    def _load_cached_profile(self, fingerprint: str) -> Optional[DeviceProfile]:
        """Load profile from cache."""
        try:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT profile_data FROM device_profiles WHERE device_fingerprint = ?",
                (fingerprint,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                data = json.loads(row[0])
                return DeviceProfile(**data)
        except (sqlite3.Error, json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to load cached profile: {e}")
        
        return None
    
    def _save_profile(self, profile: DeviceProfile):
        """Save profile to cache."""
        try:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO device_profiles (device_fingerprint, profile_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (profile.device_fingerprint, json.dumps(asdict(profile))))
            
            conn.commit()
            conn.close()
        except (sqlite3.Error, OSError, TypeError) as e:
            logger.warning(f"Failed to save device profile: {e}")
    
    def get_current_profile(self) -> Optional[DeviceProfile]:
        """Get current device profile."""
        if self._current_profile is None:
            self._current_profile = self.profile_current_device()
        return self._current_profile
    
    def get_profile_for_fingerprint(self, fingerprint: str) -> Optional[DeviceProfile]:
        """Get profile for specific device fingerprint."""
        return self._load_cached_profile(fingerprint)
    
    def get_device_class(self) -> str:
        """Get device class for current device."""
        profile = self.get_current_profile()
        return profile.device_class if profile else "cpu"
    
    def estimate_latency(
        self,
        model_flops: float,
        model_memory_mb: float,
        runtime: str
    ) -> Optional[float]:
        """Estimate latency based on device profile."""
        profile = self.get_current_profile()
        
        if profile is None or profile.matmul_throughput is None:
            return None
        
        # Estimate based on FLOPS and throughput
        if runtime == "onnx" or runtime == "onnxruntime":
            if profile.matmul_throughput > 0:
                estimated_seconds = model_flops / (profile.matmul_throughput * 1e9)
                return estimated_seconds * 1000  # Convert to ms
        
        return None
    
    def get_supported_runtimes(self) -> List[str]:
        """Get list of supported runtimes on current device."""
        profile = self.get_current_profile()
        
        if profile is None:
            return []
        
        runtimes = []
        if profile.onnxruntime_available:
            runtimes.append("onnx")
        if profile.tensorrt_available:
            runtimes.append("tensorrt")
        if profile.openvino_available:
            runtimes.append("openvino")
        if profile.tflite_available:
            runtimes.append("tflite")
        
        return runtimes


# Global profiler instance
_profiler_instance: Optional[DeviceProfiler] = None


def get_profiler() -> DeviceProfiler:
    """Get global device profiler instance."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = DeviceProfiler()
    return _profiler_instance


def profile_device() -> DeviceProfile:
    """Profile current device and return profile."""
    return get_profiler().profile_current_device()
