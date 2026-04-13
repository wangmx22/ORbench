#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate SPH force computation data.

Generates random particle positions, velocities, densities and masses in a
dam-break domain. Precomputes cell-linked list as input for force computation.
Ported from DualSPHysics KerInteractionForcesFluid.

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
    "small":  {"N": 10000,    "seed": 42},
    "medium": {"N": 100000,   "seed": 42},
    "large":  {"N": 500000,   "seed": 42},
}

# SPH parameters (typical dam-break)
H = 0.01            # smoothing length
CS0 = 40.0           # speed of sound
RHOP0 = 1000.0       # reference density (water)
ALPHA_VISC = 0.01    # artificial viscosity coefficient
CELL_SIZE = 2.0 * H  # cell size = kernel support diameter


def build_cell_linked_list(xs, ys, zs, cell_size, grid_nx, grid_ny, grid_nz):
    """Build cell-linked list: sort particles by cell, compute begin/end arrays."""
    N = len(xs)
    num_cells = grid_nx * grid_ny * grid_nz

    # Compute cell ids
    cx = np.clip(np.floor(xs / cell_size).astype(np.int32), 0, grid_nx - 1)
    cy = np.clip(np.floor(ys / cell_size).astype(np.int32), 0, grid_ny - 1)
    cz = np.clip(np.floor(zs / cell_size).astype(np.int32), 0, grid_nz - 1)
    cell_ids = cx + cy * grid_nx + cz * grid_nx * grid_ny

    # Sort by cell_id (stable sort preserves order within cell)
    sorted_idx = np.argsort(cell_ids, kind='stable').astype(np.int32)
    sorted_cell_ids = cell_ids[sorted_idx]

    # Build begin/end
    cell_begin = np.full(num_cells, -1, dtype=np.int32)
    cell_end = np.full(num_cells, -1, dtype=np.int32)

    if N > 0:
        cell_begin[sorted_cell_ids[0]] = 0
        for i in range(1, N):
            if sorted_cell_ids[i] != sorted_cell_ids[i - 1]:
                cell_end[sorted_cell_ids[i - 1]] = i
                cell_begin[sorted_cell_ids[i]] = i
        cell_end[sorted_cell_ids[N - 1]] = N

    return sorted_idx, cell_begin, cell_end


def generate_data(N, seed):
    """Generate SPH particle data with precomputed cell-linked list."""
    rng = np.random.RandomState(seed)

    # Domain: [0, 0.5] x [0, 0.5] x [0, 0.5] (small dam-break)
    domain_x, domain_y, domain_z = 0.5, 0.5, 0.5

    # Particle positions: roughly uniform with some clustering
    xs = rng.uniform(0.001, domain_x - 0.001, N).astype(np.float32)
    ys = rng.uniform(0.001, domain_y - 0.001, N).astype(np.float32)
    zs = rng.uniform(0.001, domain_z - 0.001, N).astype(np.float32)

    # Velocities: small random + downward gravity component
    vxs = (rng.randn(N) * 0.1).astype(np.float32)
    vys = (rng.randn(N) * 0.1).astype(np.float32)
    vzs = (rng.randn(N) * 0.1 - 0.5).astype(np.float32)

    # Densities: around reference density with 5% variation
    rhos = (RHOP0 + rng.randn(N) * RHOP0 * 0.05).astype(np.float32)
    rhos = np.clip(rhos, RHOP0 * 0.8, RHOP0 * 1.2)

    # Masses: uniform (dp^3 * rhop0, but we just use a constant)
    dp = 0.005  # particle spacing
    mass_val = dp * dp * dp * RHOP0
    masses = np.full(N, mass_val, dtype=np.float32)

    # Grid parameters
    grid_nx = int(np.ceil(domain_x / CELL_SIZE))
    grid_ny = int(np.ceil(domain_y / CELL_SIZE))
    grid_nz = int(np.ceil(domain_z / CELL_SIZE))

    # Build cell-linked list
    sorted_idx, cell_begin, cell_end = build_cell_linked_list(
        xs, ys, zs, CELL_SIZE, grid_nx, grid_ny, grid_nz)

    tensors = {
        "xs":         ("float32", xs),
        "ys":         ("float32", ys),
        "zs":         ("float32", zs),
        "vxs":        ("float32", vxs),
        "vys":        ("float32", vys),
        "vzs":        ("float32", vzs),
        "rhos":       ("float32", rhos),
        "masses":     ("float32", masses),
        "cell_begin": ("int32",   cell_begin),
        "cell_end":   ("int32",   cell_end),
        "sorted_idx": ("int32",   sorted_idx),
    }

    params = {
        "N": N,
        "h_x1000000": int(H * 1000000),
        "cs0_x10000": int(CS0 * 10000),
        "rhop0_x100": int(RHOP0 * 100),
        "alpha_visc_x10000": int(ALPHA_VISC * 10000),
        "cell_size_x1000000": int(CELL_SIZE * 1000000),
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "grid_nz": grid_nz,
    }

    return tensors, params


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "sph_forces"
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
          f"h={params['h_x1000000'] / 1000000.0}, cell_size={params['cell_size_x1000000'] / 1000000.0}")

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
