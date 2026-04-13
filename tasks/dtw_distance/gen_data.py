#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) — Generate DTW distance data.

Subjects are generated as cylinder-bell-funnel (CBF) synthetic time series
matching the cuDTW++ benchmark setup, then z-normalized. The query is the
first generated CBF series (also z-normalized).

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re as re_mod
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"num_entries": 256,  "num_features": 1023, "seed": 42},
    "medium": {"num_entries": 1024, "num_features": 1023, "seed": 42},
    "large":  {"num_entries": 4096, "num_features": 1023, "seed": 42},
}


def generate_cbf(num_entries, num_features, seed):
    """
    Cylinder-Bell-Funnel synthetic time series generator (Saito 1994), as used
    by cuDTW++ for benchmarking. Each entry is randomly assigned one of three
    pattern types and parameters, then z-normalized.
    """
    rng = np.random.RandomState(seed)
    series = np.zeros((num_entries, num_features), dtype=np.float32)

    for i in range(num_entries):
        cls = i % 3  # 0=cylinder, 1=bell, 2=funnel
        a = rng.randint(0, num_features // 4 + 1)
        b = a + rng.randint(num_features // 8, num_features // 2 + 1)
        b = min(b, num_features - 1)
        eta = rng.randn(num_features).astype(np.float32)
        amp = 6.0 + rng.randn() * 2.0
        x = np.zeros(num_features, dtype=np.float32)
        idx = np.arange(num_features)
        mask = (idx >= a) & (idx <= b)
        if cls == 0:  # cylinder: flat plateau
            x[mask] = amp
        elif cls == 1:  # bell: rising ramp
            denom = max(b - a, 1)
            x[mask] = amp * (idx[mask] - a) / denom
        else:  # funnel: falling ramp
            denom = max(b - a, 1)
            x[mask] = amp * (b - idx[mask]) / denom
        x = x + eta
        # z-normalize
        mu = float(x.mean())
        sd = float(x.std())
        if sd < 1e-6:
            sd = 1.0
        series[i] = (x - mu) / sd

    return series


def generate_data(num_entries, num_features, seed):
    series = generate_cbf(num_entries, num_features, seed)
    query = series[0].copy()
    return series, query


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "dtw_distance"
    exe = task_dir / "solution_cpu"
    src = task_dir / "cpu_reference.c"
    task_io_cpu = task_dir / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass

    cmd = [
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re_mod.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    num_entries  = cfg["num_entries"]
    num_features = cfg["num_features"]

    print(f"[gen_data] Generating {size_name}: "
          f"num_entries={num_entries}, num_features={num_features}")

    subjects, query = generate_data(num_entries, num_features, cfg["seed"])

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("subjects", "float32", subjects.reshape(-1)),
            ("query",    "float32", query.reshape(-1)),
        ],
        params={
            "num_entries":  num_entries,
            "num_features": num_features,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{num_entries}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
