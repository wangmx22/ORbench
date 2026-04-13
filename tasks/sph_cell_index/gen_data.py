#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate SPH cell-linked list data.

Generates random particle positions in a dam-break domain and grid parameters,
ported from DualSPHysics KerCalcBeginEndCell.

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
    """Generate particle positions and grid parameters for cell indexing."""
    rng = np.random.RandomState(seed)

    # Domain: [0, 4] x [0, 2] x [0, 2], dam-break like
    domain_x, domain_y, domain_z = 4.0, 2.0, 2.0
    cell_size = 0.04  # typical = 2*h for SPH

    xs = rng.uniform(0.0, domain_x, N).astype(np.float32)
    ys = rng.uniform(0.0, domain_y, N).astype(np.float32)
    zs = rng.uniform(0.0, domain_z, N).astype(np.float32)

    grid_nx = int(np.ceil(domain_x / cell_size))
    grid_ny = int(np.ceil(domain_y / cell_size))
    grid_nz = int(np.ceil(domain_z / cell_size))

    tensors = {
        "xs": ("float32", xs),
        "ys": ("float32", ys),
        "zs": ("float32", zs),
    }

    params = {
        "N": N,
        "cell_size_x1000000": int(cell_size * 1000000),
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "grid_nz": grid_nz,
    }

    return tensors, params


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "sph_cell_index"
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

    tensors, params = generate_data(N, cfg["seed"])

    print(f"  {N} particles, grid={params['grid_nx']}x{params['grid_ny']}x{params['grid_nz']}, "
          f"cell_size={params['cell_size_x1000000'] / 1000000.0}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[(name, dtype, arr) for name, (dtype, arr) in tensors.items()],
        params=params,
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
