"""
profile.py - Automated nsys profiling and CSV analysis
Works in K8s/Docker containers (no SYS_ADMIN required).
"""

import os
import subprocess
import shutil
import tempfile
from typing import Optional

import pandas as pd


def check_nsys_available() -> bool:
    """Check if nsys is available in the environment"""
    return shutil.which("nsys") is not None


def check_ncu_available() -> bool:
    """
    Check if NCU is available AND has profiling permission.
    NCU requires SYS_ADMIN in containers.
    """
    if not shutil.which("ncu"):
        return False

    # Check RmProfilingAdminOnly
    param_file = "/proc/driver/nvidia/params"
    if os.path.exists(param_file):
        try:
            with open(param_file, "r") as f:
                content = f.read()
            if "RmProfilingAdminOnly: 1" in content:
                # In a container, this means NCU won't work
                # (root in container != root on host)
                if os.path.exists("/.dockerenv") or _is_k8s_pod():
                    return False
        except Exception:
            pass

    return True


def _is_k8s_pod() -> bool:
    """Check if running inside a Kubernetes pod"""
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "kubepods" in f.read()
    except Exception:
        return False


def run_nsys_profile(
    exe_path: str,
    exe_args: list[str] = None,
    device_id: int = 0,
    timeout: int = 60,
    output_dir: str = None,
) -> Optional[str]:
    """
    Run nsys profile on an executable and export all useful CSV reports.
    
    Args:
        exe_path: Path to compiled CUDA executable
        exe_args: Arguments to pass to the executable
        device_id: GPU device ID
        timeout: Max profiling duration in seconds
        output_dir: Where to save outputs (default: temp dir)
    
    Returns:
        Tuple of (primary_csv_path, dict of report_name -> csv_path),
        or None if profiling failed.
    """
    if not check_nsys_available():
        return None

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="orbench_nsys_")

    report_base = os.path.join(output_dir, "profile")

    # Clean up any existing files
    for ext in [".nsys-rep", ".sqlite", ".qdstrm"]:
        f = report_base + ext
        if os.path.exists(f):
            os.remove(f)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # Step 1: Profile
    nsys_cmd = [
        "nsys", "profile",
        "-t", "cuda",
        "-s", "none",           # no CPU sampling (avoids permission issues)
        f"--duration={timeout}",
        "-o", report_base,
        exe_path,
    ] + (exe_args or [])

    try:
        result = subprocess.run(
            nsys_cmd, capture_output=True, timeout=timeout + 30,
            text=True, env=env,
        )
    except subprocess.TimeoutExpired:
        print("[nsys] Profiling timed out")
        return None
    except Exception as e:
        print(f"[nsys] Error: {e}")
        return None

    nsys_rep = report_base + ".nsys-rep"
    if not os.path.exists(nsys_rep):
        print("[nsys] No .nsys-rep generated")
        return None

    # Step 2: Export ALL useful CSV reports
    ALL_REPORTS = [
        "cuda_gpu_trace",          # every kernel/memcpy with timestamps
        "cuda_gpu_kern_sum",       # kernel summary (aggregated by name)
        "cuda_gpu_mem_time_trace", # every memcpy/memset with direction, size, throughput
        "cuda_gpu_mem_time_sum",   # memops summary by type (H2D/D2H/memset)
        "cuda_gpu_mem_size_sum",   # memops summary by size
        "cuda_api_trace",          # every CPU-side CUDA API call
        "cuda_api_sum",            # CUDA API summary by function name
    ]

    exported_csvs = {}
    for report_name in ALL_REPORTS:
        csv_path = f"{report_base}_{report_name}.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

        export_cmd = [
            "nsys", "stats",
            "--force-export=true",
            "--report", report_name,
            "--format", "csv",
            "-o", report_base,
            nsys_rep,
        ]

        try:
            subprocess.run(export_cmd, capture_output=True, timeout=60, text=True)
            if os.path.exists(csv_path):
                exported_csvs[report_name] = csv_path
        except Exception:
            pass  # some reports may not have data, skip silently

    # Return the gpu_trace csv as primary (backward compatible),
    # but all CSVs are saved on disk
    primary_csv = exported_csvs.get("cuda_gpu_trace")
    return primary_csv, exported_csvs


def analyze_nsys_trace(csv_path: str) -> dict:
    """
    Analyze nsys GPU trace CSV and extract key metrics.
    (Backward compatible: only needs the gpu_trace CSV)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[nsys] Failed to read CSV: {e}")
        return {}

    if df.empty or "Duration (ns)" not in df.columns:
        return {}

    is_kernel = ~df["Name"].str.startswith("[", na=False)
    is_memop = df["Name"].str.startswith("[", na=False)

    kernels = df[is_kernel]
    memops = df[is_memop]

    start_min = df["Start (ns)"].min()
    end_max = (df["Start (ns)"] + df["Duration (ns)"]).max()
    total_span_ns = end_max - start_min

    total_kernel_ns = kernels["Duration (ns)"].sum() if not kernels.empty else 0
    total_memop_ns = memops["Duration (ns)"].sum() if not memops.empty else 0

    kernel_breakdown = {}
    if not kernels.empty:
        for name, group in kernels.groupby("Name"):
            short_name = name.split("(")[0]
            kernel_breakdown[short_name] = {
                "count": len(group),
                "total_ms": group["Duration (ns)"].sum() / 1e6,
                "avg_us": group["Duration (ns)"].mean() / 1e3,
                "min_us": group["Duration (ns)"].min() / 1e3,
                "max_us": group["Duration (ns)"].max() / 1e3,
            }

    # Compute inter-kernel gaps (GPU idle time between consecutive operations)
    if len(df) > 1:
        sorted_df = df.sort_values("Start (ns)")
        ends = (sorted_df["Start (ns)"] + sorted_df["Duration (ns)"]).values[:-1]
        starts = sorted_df["Start (ns)"].values[1:]
        gaps = starts - ends
        gaps = gaps[gaps > 0]
        total_gap_ns = gaps.sum() if len(gaps) > 0 else 0
    else:
        total_gap_ns = 0

    return {
        "total_kernel_time_ms": total_kernel_ns / 1e6,
        "num_kernel_launches": len(kernels),
        "kernel_breakdown": kernel_breakdown,
        "total_memcpy_time_ms": total_memop_ns / 1e6,
        "total_gpu_span_ms": total_span_ns / 1e6,
        "gpu_active_ratio": (total_kernel_ns + total_memop_ns) / total_span_ns if total_span_ns > 0 else 0,
        "gpu_idle_ms": total_gap_ns / 1e6,
    }


def analyze_all_nsys_csvs(exported_csvs: dict) -> dict:
    """
    Analyze ALL exported nsys CSV reports and return comprehensive metrics.
    
    Args:
        exported_csvs: dict mapping report_name -> csv_path
    
    Returns:
        Dict with all extracted metrics organized by category.
    """
    result = {}

    # 1. GPU Kernel Trace (per-launch detail)
    if "cuda_gpu_trace" in exported_csvs:
        result["gpu_trace"] = analyze_nsys_trace(exported_csvs["cuda_gpu_trace"])

    # 2. GPU Kernel Summary (aggregated by kernel name)
    if "cuda_gpu_kern_sum" in exported_csvs:
        try:
            df = pd.read_csv(exported_csvs["cuda_gpu_kern_sum"])
            if not df.empty:
                kern_sum = {}
                for _, row in df.iterrows():
                    name = str(row.get("Name", "unknown")).split("(")[0]
                    kern_sum[name] = {
                        "count": int(row.get("Instances", 0)),
                        "total_ns": int(row.get("Total Time (ns)", 0)),
                        "total_ms": int(row.get("Total Time (ns)", 0)) / 1e6,
                        "avg_ns": float(row.get("Avg (ns)", 0)),
                        "avg_us": float(row.get("Avg (ns)", 0)) / 1e3,
                        "min_us": float(row.get("Min (ns)", 0)) / 1e3,
                        "max_us": float(row.get("Max (ns)", 0)) / 1e3,
                        "time_pct": float(row.get("Time (%)", 0)),
                    }
                result["kernel_summary"] = kern_sum
        except Exception:
            pass

    # 3. Memory Operations by Time (H2D/D2H/memset timing)
    if "cuda_gpu_mem_time_sum" in exported_csvs:
        try:
            df = pd.read_csv(exported_csvs["cuda_gpu_mem_time_sum"])
            if not df.empty:
                mem_time = {}
                for _, row in df.iterrows():
                    op = str(row.get("Operation", "unknown"))
                    mem_time[op] = {
                        "count": int(row.get("Count", 0)),
                        "total_ns": int(row.get("Total Time (ns)", 0)),
                        "total_ms": int(row.get("Total Time (ns)", 0)) / 1e6,
                        "avg_us": float(row.get("Avg (ns)", 0)) / 1e3,
                        "time_pct": float(row.get("Time (%)", 0)),
                    }
                result["mem_time_summary"] = mem_time
        except Exception:
            pass

    # 4. Memory Operations by Size (how much data transferred)
    if "cuda_gpu_mem_size_sum" in exported_csvs:
        try:
            df = pd.read_csv(exported_csvs["cuda_gpu_mem_size_sum"])
            if not df.empty:
                mem_size = {}
                for _, row in df.iterrows():
                    op = str(row.get("Operation", "unknown"))
                    mem_size[op] = {
                        "count": int(row.get("Count", 0)),
                        "total_mb": float(row.get("Total (MB)", 0)),
                        "avg_mb": float(row.get("Avg (MB)", 0)),
                    }
                result["mem_size_summary"] = mem_size
        except Exception:
            pass

    # 5. CUDA API Summary (CPU-side overhead)
    if "cuda_api_sum" in exported_csvs:
        try:
            df = pd.read_csv(exported_csvs["cuda_api_sum"])
            if not df.empty:
                api_sum = {}
                for _, row in df.iterrows():
                    name = str(row.get("Name", "unknown"))
                    api_sum[name] = {
                        "count": int(row.get("Num Calls", 0)),
                        "total_ns": int(row.get("Total Time (ns)", 0)),
                        "total_ms": int(row.get("Total Time (ns)", 0)) / 1e6,
                        "avg_us": float(row.get("Avg (ns)", 0)) / 1e3,
                        "time_pct": float(row.get("Time (%)", 0)),
                    }
                result["cuda_api_summary"] = api_sum
        except Exception:
            pass

    # 6. Memory Trace Detail (per-operation: direction, size, throughput)
    if "cuda_gpu_mem_time_trace" in exported_csvs:
        try:
            df = pd.read_csv(exported_csvs["cuda_gpu_mem_time_trace"])
            if not df.empty:
                h2d = df[df["Name"].str.contains("Host-to-Device", na=False)]
                d2h = df[df["Name"].str.contains("Device-to-Host", na=False)]
                memset = df[df["Name"].str.contains("memset", na=False)]

                result["mem_detail"] = {
                    "h2d_count": len(h2d),
                    "h2d_total_ms": h2d["Duration (ns)"].sum() / 1e6 if not h2d.empty else 0,
                    "h2d_total_mb": h2d["Bytes (MB)"].sum() if not h2d.empty and "Bytes (MB)" in h2d.columns else 0,
                    "d2h_count": len(d2h),
                    "d2h_total_ms": d2h["Duration (ns)"].sum() / 1e6 if not d2h.empty else 0,
                    "d2h_total_mb": d2h["Bytes (MB)"].sum() if not d2h.empty and "Bytes (MB)" in d2h.columns else 0,
                    "memset_count": len(memset),
                    "memset_total_ms": memset["Duration (ns)"].sum() / 1e6 if not memset.empty else 0,
                }
        except Exception:
            pass

    return result


def write_nsys_full_summary(analysis: dict, output_path: str):
    """
    Write a comprehensive human-readable nsys summary file.
    """
    with open(output_path, "w") as f:
        gpu = analysis.get("gpu_trace", {})

        f.write("=" * 70 + "\n")
        f.write("  ORBench nsys Profiling Summary\n")
        f.write("=" * 70 + "\n\n")

        # --- Overall ---
        f.write("[ Overall GPU Activity ]\n")
        f.write(f"  Total GPU span:      {gpu.get('total_gpu_span_ms', 0):.3f} ms\n")
        f.write(f"  Kernel time:         {gpu.get('total_kernel_time_ms', 0):.3f} ms\n")
        f.write(f"  Memcpy/memset time:  {gpu.get('total_memcpy_time_ms', 0):.3f} ms\n")
        f.write(f"  GPU idle time:       {gpu.get('gpu_idle_ms', 0):.3f} ms\n")
        f.write(f"  GPU utilization:     {gpu.get('gpu_active_ratio', 0):.1%}\n")
        f.write(f"  Kernel launches:     {gpu.get('num_kernel_launches', 0)}\n")
        f.write("\n")

        # --- Kernel Breakdown ---
        kern = analysis.get("kernel_summary", gpu.get("kernel_breakdown", {}))
        if kern:
            f.write("[ Kernel Breakdown ]\n")
            f.write(f"  {'Kernel':<45s} {'Count':>6s} {'Total(ms)':>10s} {'Avg(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s}\n")
            f.write("  " + "-" * 95 + "\n")
            for name, stats in kern.items():
                f.write(f"  {name:<45s} {stats.get('count',0):>6d} "
                        f"{stats.get('total_ms',0):>10.3f} "
                        f"{stats.get('avg_us',0):>10.1f} "
                        f"{stats.get('min_us',0):>10.1f} "
                        f"{stats.get('max_us',0):>10.1f}\n")
            f.write("\n")

        # --- Memory Operations ---
        mem_detail = analysis.get("mem_detail", {})
        if mem_detail:
            f.write("[ Memory Operations ]\n")
            f.write(f"  Host-to-Device:  {mem_detail.get('h2d_count', 0):>6d} calls, "
                    f"{mem_detail.get('h2d_total_ms', 0):.3f} ms, "
                    f"{mem_detail.get('h2d_total_mb', 0):.2f} MB\n")
            f.write(f"  Device-to-Host:  {mem_detail.get('d2h_count', 0):>6d} calls, "
                    f"{mem_detail.get('d2h_total_ms', 0):.3f} ms, "
                    f"{mem_detail.get('d2h_total_mb', 0):.2f} MB\n")
            f.write(f"  cudaMemset:      {mem_detail.get('memset_count', 0):>6d} calls, "
                    f"{mem_detail.get('memset_total_ms', 0):.3f} ms\n")
            f.write("\n")

        # --- CUDA API ---
        api = analysis.get("cuda_api_summary", {})
        if api:
            f.write("[ CUDA API Summary (CPU-side) ]\n")
            f.write(f"  {'Function':<40s} {'Calls':>8s} {'Total(ms)':>10s} {'Avg(us)':>10s} {'Time%':>7s}\n")
            f.write("  " + "-" * 80 + "\n")
            # Sort by total time descending
            for name, stats in sorted(api.items(), key=lambda x: x[1].get('total_ms', 0), reverse=True):
                f.write(f"  {name:<40s} {stats.get('count',0):>8d} "
                        f"{stats.get('total_ms',0):>10.3f} "
                        f"{stats.get('avg_us',0):>10.1f} "
                        f"{stats.get('time_pct',0):>6.1f}%\n")
            f.write("\n")

        f.write("=" * 70 + "\n")