#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate SPH position update data.

Generates random particle positions and displacement vectors simulating
a dam-break scenario, ported from DualSPHysics KerUpdatePos.

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
    "small":  {"N": 50000,    "seed": 42},
    "medium": {"N": 500000,   "seed": 42},
    "large":  {"N": 5000000,  "seed": 42},
}


def generate_data(N, seed):
    """Generate particle positions (dam-break like) and displacement vectors."""
    rng = np.random.RandomState(seed)

    # Dam-break: particles in a rectangular block [0, 2] x [0, 1] x [0, 1]
    pos_x = rng.uniform(0.0, 2.0, N).astype(np.float32)
    pos_y = rng.uniform(0.0, 1.0, N).astype(np.float32)
    pos_z = rng.uniform(0.0, 1.0, N).astype(np.float32)

    # Small displacement vectors (dt * velocity, typical SPH timestep)
    dt = 1e-4
    mov_x = (rng.randn(N) * 0.5 * dt).astype(np.float32)
    mov_y = (rng.randn(N) * 0.5 * dt).astype(np.float32)
    mov_z = ((rng.randn(N) * 0.5 - 9.81) * dt).astype(np.float32)  # gravity

    cell_size = 0.02  # typical SPH cell size = 2*h

    return {
        "pos_x": ("float32", pos_x),
        "pos_y": ("float32", pos_y),
        "pos_z": ("float32", pos_z),
        "mov_x": ("float32", mov_x),
        "mov_y": ("float32", mov_y),
        "mov_z": ("float32", mov_z),
    }, cell_size


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "sph_position"
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

    print(f"[gen_data] Generating {size_name}: {N} particles...")

    data, cell_size = generate_data(N, cfg["seed"])

    print(f"  {N} particles, cell_size={cell_size}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[(name, dtype, arr) for name, (dtype, arr) in data.items()],
        params={
            "N": N,
            "cell_size_x1000000": int(cell_size * 1000000),
        },
    )

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
