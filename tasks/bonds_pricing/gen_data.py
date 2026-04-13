#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate random bond portfolios for pricing.

Generates random issue/maturity dates and coupon rates, then optionally
computes expected outputs via the CPU baseline.

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
    "small":  {"N": 100000,   "seed": 42},
    "medium": {"N": 1000000,  "seed": 42},
    "large":  {"N": 10000000, "seed": 42},
}


def generate_bonds(N, seed):
    """Generate random bond parameters matching FinanceBench conventions."""
    rng = np.random.default_rng(seed)

    # Issue dates: random day/month in 1998-1999
    issue_day   = rng.integers(1, 29, size=N).astype(np.int32)
    issue_month = rng.integers(1, 13, size=N).astype(np.int32)
    issue_year  = (1999 - rng.integers(0, 2, size=N)).astype(np.int32)

    # Maturity dates: random day/month in 2000-2001
    maturity_day   = rng.integers(1, 29, size=N).astype(np.int32)
    maturity_month = rng.integers(1, 13, size=N).astype(np.int32)
    maturity_year  = (2000 + rng.integers(0, 2, size=N)).astype(np.int32)

    # Bond rates: ~0.08 +/- 0.05
    rates = (0.08 + (rng.random(N).astype(np.float32) - 0.5) * 0.1).astype(np.float32)

    return issue_year, issue_month, issue_day, maturity_year, maturity_month, maturity_day, rates


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "bonds_pricing"
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
    N = cfg["N"]
    seed = cfg["seed"]
    coupon_freq = 2.0  # semiannual

    print(f"[gen_data] Generating {size_name}: {N} bonds...")

    issue_year, issue_month, issue_day, \
        maturity_year, maturity_month, maturity_day, rates = generate_bonds(N, seed)

    coupon_freq_x100 = int(round(coupon_freq * 100))

    print(f"  {N} bonds, coupon_freq={coupon_freq}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("issue_year",     "int32",   issue_year),
            ("issue_month",    "int32",   issue_month),
            ("issue_day",      "int32",   issue_day),
            ("maturity_year",  "int32",   maturity_year),
            ("maturity_month", "int32",   maturity_month),
            ("maturity_day",   "int32",   maturity_day),
            ("rates",          "float32", rates),
        ],
        params={
            "N": N,
            "coupon_freq_x100": coupon_freq_x100,
        },
    )

    # Dummy requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

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
