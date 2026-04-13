#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Black-Scholes option pricing data.

Uses the 37 hardcoded option configurations from FinanceBench, cycled to fill N options.

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
    "small":  {"N": 100000,    "seed": 42},
    "medium": {"N": 1000000,   "seed": 42},
    "large":  {"N": 10000000,  "seed": 42},
}

# 37 option configurations from FinanceBench blackScholesAnalyticEngine.c
# Format: (type, strike, spot, q, r, t, vol)
# type: 0=CALL, 1=PUT
OPTION_CONFIGS = [
    (0,  40.0,  42.0, 0.08, 0.04, 0.75, 0.35),
    (0, 100.0,  90.0, 0.10, 0.10, 0.10, 0.15),
    (0, 100.0, 100.0, 0.10, 0.10, 0.10, 0.15),
    (0, 100.0, 110.0, 0.10, 0.10, 0.10, 0.15),
    (0, 100.0,  90.0, 0.10, 0.10, 0.10, 0.25),
    (0, 100.0, 100.0, 0.10, 0.10, 0.10, 0.25),
    (0, 100.0, 110.0, 0.10, 0.10, 0.10, 0.25),
    (0, 100.0,  90.0, 0.10, 0.10, 0.10, 0.35),
    (0, 100.0, 100.0, 0.10, 0.10, 0.10, 0.35),
    (0, 100.0, 110.0, 0.10, 0.10, 0.10, 0.35),
    (0, 100.0,  90.0, 0.10, 0.10, 0.50, 0.15),
    (0, 100.0, 100.0, 0.10, 0.10, 0.50, 0.15),
    (0, 100.0, 110.0, 0.10, 0.10, 0.50, 0.15),
    (0, 100.0,  90.0, 0.10, 0.10, 0.50, 0.25),
    (0, 100.0, 100.0, 0.10, 0.10, 0.50, 0.25),
    (0, 100.0, 110.0, 0.10, 0.10, 0.50, 0.25),
    (0, 100.0,  90.0, 0.10, 0.10, 0.50, 0.35),
    (0, 100.0, 100.0, 0.10, 0.10, 0.50, 0.35),
    (0, 100.0, 110.0, 0.10, 0.10, 0.50, 0.35),
    (1, 100.0,  90.0, 0.10, 0.10, 0.10, 0.15),
    (1, 100.0, 100.0, 0.10, 0.10, 0.10, 0.15),
    (1, 100.0, 110.0, 0.10, 0.10, 0.10, 0.15),
    (1, 100.0,  90.0, 0.10, 0.10, 0.10, 0.25),
    (1, 100.0, 100.0, 0.10, 0.10, 0.10, 0.25),
    (1, 100.0, 110.0, 0.10, 0.10, 0.10, 0.25),
    (1, 100.0,  90.0, 0.10, 0.10, 0.10, 0.35),
    (1, 100.0, 100.0, 0.10, 0.10, 0.10, 0.35),
    (1, 100.0, 110.0, 0.10, 0.10, 0.10, 0.35),
    (1, 100.0,  90.0, 0.10, 0.10, 0.50, 0.15),
    (1, 100.0, 100.0, 0.10, 0.10, 0.50, 0.15),
    (1, 100.0, 110.0, 0.10, 0.10, 0.50, 0.15),
    (1, 100.0,  90.0, 0.10, 0.10, 0.50, 0.25),
    (1, 100.0, 100.0, 0.10, 0.10, 0.50, 0.25),
    (1, 100.0, 110.0, 0.10, 0.10, 0.50, 0.25),
    (1, 100.0,  90.0, 0.10, 0.10, 0.50, 0.35),
    (1, 100.0, 100.0, 0.10, 0.10, 0.50, 0.35),
    (1, 100.0, 110.0, 0.10, 0.10, 0.50, 0.35),
]


# ---------------------------------------------------------------------------
# Option data generation
# ---------------------------------------------------------------------------

def generate_options(N):
    """Generate N options by cycling through the 37 configurations."""
    n_configs = len(OPTION_CONFIGS)
    types   = np.zeros(N, dtype=np.int32)
    strikes = np.zeros(N, dtype=np.float32)
    spots   = np.zeros(N, dtype=np.float32)
    qs      = np.zeros(N, dtype=np.float32)
    rs      = np.zeros(N, dtype=np.float32)
    ts      = np.zeros(N, dtype=np.float32)
    vols    = np.zeros(N, dtype=np.float32)

    for i in range(N):
        cfg = OPTION_CONFIGS[i % n_configs]
        types[i]   = cfg[0]
        strikes[i] = cfg[1]
        spots[i]   = cfg[2]
        qs[i]      = cfg[3]
        rs[i]      = cfg[4]
        ts[i]      = cfg[5]
        vols[i]    = cfg[6]

    return types, strikes, spots, qs, rs, ts, vols


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "black_scholes" / "solution_cpu"
    src = orbench_root / "tasks" / "black_scholes" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "black_scholes" / "task_io_cpu.c"
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
    N = cfg["N"]

    print(f"[gen_data] Generating {size_name}: {N} options...")

    types, strikes, spots, qs, rs, ts, vols = generate_options(N)

    print(f"  {N} options, {len(OPTION_CONFIGS)} distinct configs (cycled)")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("types",   "int32",   types),
            ("strikes", "float32", strikes),
            ("spots",   "float32", spots),
            ("qs",      "float32", qs),
            ("rs",      "float32", rs),
            ("ts",      "float32", ts),
            ("vols",    "float32", vols),
        ],
        params={
            "N": N,
        },
    )

    # Write requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{N}\n")

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
