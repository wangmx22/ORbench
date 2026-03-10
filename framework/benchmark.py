"""
benchmark.py - Performance measurement using CUDA Events
Two-level timing: end-to-end (from program stdout) + nsys trace (kernel-only)
"""

import os
import re
import subprocess
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .task import load_task, get_task_dir
from .profile import run_nsys_profile, analyze_nsys_trace, analyze_all_nsys_csvs, write_nsys_full_summary


@dataclass
class TimingStats:
    mean: float = -1.0
    std: float = -1.0
    min: float = -1.0
    max: float = -1.0
    num_trials: int = 0


@dataclass
class BenchmarkResult:
    # End-to-end timing (CUDA Event, from program stdout)
    e2e_time_ms: TimingStats = field(default_factory=TimingStats)

    # Kernel-only timing (from nsys trace)
    kernel_time_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None     # kernel_time / e2e_time
    num_kernel_launches: Optional[int] = None
    memcpy_overhead_ms: Optional[float] = None
    nsys_csv_path: Optional[str] = None         # path to saved CSV

    # CPU baseline
    cpu_baseline_ms: float = -1.0

    # Speedups
    speedup_e2e: float = -1.0
    speedup_kernel: Optional[float] = None

    # Metadata
    hardware: str = ""
    device_id: int = 0
    error: str = ""


def parse_timing_output(stdout: str) -> Optional[float]:
    """
    Parse GPU timing from program stdout.
    
    Expected format (programs should print this):
        GPU_TIME_MS: 0.634
    
    Also supports:
        Time: 0.634 ms
        Elapsed: 0.634ms
    """
    patterns = [
        r'GPU_TIME_MS:\s*([\d.]+)',
        r'Time:\s*([\d.]+)\s*ms',
        r'Elapsed:\s*([\d.]+)\s*ms',
        r'gpu_time_ms=([\d.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def get_gpu_name(device_id: int = 0) -> str:
    """Get GPU device name"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             f"--id={device_id}"],
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_cpu_baseline(task_id: str, data_dir: str = None) -> float:
    """
    Run CPU reference and parse its timing.
    Returns time in milliseconds.
    """
    task_dir = get_task_dir(task_id)
    cpu_exe = os.path.join(task_dir, "cpu_reference")

    if not os.path.exists(cpu_exe):
        # Try to compile
        cpu_src = os.path.join(task_dir, "cpu_reference.cu")
        subprocess.run(
            ["nvcc", "-O2", "-o", cpu_exe, cpu_src],
            capture_output=True,
        )

    if not os.path.exists(cpu_exe):
        return -1.0

    # Find data_dir if not provided
    if data_dir is None:
        for size_name in ["large", "medium", "small"]:
            candidate = os.path.join(task_dir, "data", size_name)
            if os.path.exists(os.path.join(candidate, "input.txt")):
                data_dir = candidate
                break

    if data_dir is None:
        return -1.0

    ok, stdout, stderr = _run_exe(cpu_exe, args=[data_dir], timeout=300)
    if ok:
        t = parse_timing_output(stdout)
        if t is not None:
            return t

    return -1.0


def _run_exe(exe_path: str, args: list[str] = None, timeout: int = 180,
             device_id: int = 0) -> tuple[bool, str, str]:
    """Run executable with CUDA device selection"""
    cmd = [exe_path] + (args or [])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout, text=True, env=env,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timed out"
    except Exception as e:
        return False, "", str(e)


def benchmark_solution(
    task_id: str,
    exe_path: str,
    device_id: int = 0,
    run_nsys: bool = True,
    save_nsys_csv: bool = False,
    save_nsys_csv_dir: str = None,
) -> BenchmarkResult:
    """
    Benchmark a compiled GPU solution.
    
    Level A (always): Parse CUDA Event timing from program stdout.
    Level B (optional): Run nsys for kernel-level timing breakdown.
    
    Uses the largest available data size for benchmarking.
    """
    task = load_task(task_id)
    task_dir = get_task_dir(task_id)
    result = BenchmarkResult()
    result.device_id = device_id
    result.hardware = get_gpu_name(device_id)

    # Pick the largest available data size for benchmarking
    data_dir = None
    for size_name in ["large", "medium", "small"]:
        candidate = os.path.join(task_dir, "data", size_name)
        if os.path.exists(os.path.join(candidate, "input.txt")):
            data_dir = candidate
            break

    if data_dir is None:
        result.error = "No pre-generated data found for benchmarking"
        return result

    exe_args = [data_dir]

    # === Level A: End-to-end timing from CUDA Events ===
    # Harness internally does: 3 warmup + 10 timed trials, prints mean
    ok, stdout, stderr = _run_exe(exe_path, args=exe_args, device_id=device_id, timeout=task.timeout)
    if not ok:
        result.error = f"Execution failed: {stderr[:200]}"
        return result

    t = parse_timing_output(stdout)
    if t is not None:
        result.e2e_time_ms = TimingStats(
            mean=t,
            std=0.0,
            min=t,
            max=t,
            num_trials=1,  # harness internally ran 10 trials, we get the mean
        )

    # === CPU baseline ===
    result.cpu_baseline_ms = run_cpu_baseline(task_id, data_dir=data_dir)

    if result.cpu_baseline_ms > 0 and result.e2e_time_ms.mean > 0:
        result.speedup_e2e = result.cpu_baseline_ms / result.e2e_time_ms.mean

    # === Level B: nsys trace analysis (optional) ===
    if run_nsys:
        try:
            # Save nsys output next to the executable
            nsys_output_dir = os.path.dirname(exe_path)
            nsys_result = run_nsys_profile(
                exe_path, exe_args=exe_args, device_id=device_id,
                timeout=task.timeout, output_dir=nsys_output_dir
            )

            if nsys_result is not None:
                # Unpack: (primary_csv_path, {report_name: csv_path, ...})
                if isinstance(nsys_result, tuple):
                    primary_csv, exported_csvs = nsys_result
                else:
                    # Backward compat: old version returned just a string
                    primary_csv = nsys_result
                    exported_csvs = {"cuda_gpu_trace": primary_csv} if primary_csv else {}

                # Basic analysis from gpu_trace
                if primary_csv and isinstance(primary_csv, str):
                    nsys_data = analyze_nsys_trace(primary_csv)
                    result.kernel_time_ms = nsys_data.get("total_kernel_time_ms")
                    result.num_kernel_launches = nsys_data.get("num_kernel_launches")
                    result.memcpy_overhead_ms = nsys_data.get("total_memcpy_time_ms")

                    if result.kernel_time_ms and result.e2e_time_ms.mean > 0:
                        result.gpu_utilization = result.kernel_time_ms / result.e2e_time_ms.mean

                    if result.cpu_baseline_ms > 0 and result.kernel_time_ms:
                        result.speedup_kernel = result.cpu_baseline_ms / result.kernel_time_ms

                # Full analysis from all CSVs
                if save_nsys_csv and save_nsys_csv_dir and exported_csvs:
                    import shutil
                    os.makedirs(save_nsys_csv_dir, exist_ok=True)

                    # Copy all CSV files to run directory
                    for report_name, csv_path in exported_csvs.items():
                        dst = os.path.join(save_nsys_csv_dir, f"nsys_{report_name}.csv")
                        shutil.copy2(csv_path, dst)
                    print(f"  [nsys] {len(exported_csvs)} CSV files saved to {save_nsys_csv_dir}")

                    # Generate comprehensive summary
                    full_analysis = analyze_all_nsys_csvs(exported_csvs)
                    summary_path = os.path.join(save_nsys_csv_dir, "nsys_summary.txt")
                    write_nsys_full_summary(full_analysis, summary_path)
                    print(f"  [nsys] Full summary saved to {summary_path}")

        except Exception as e:
            # nsys is optional; don't fail the benchmark
            result.error += f"nsys profiling failed: {e}. "

    return result