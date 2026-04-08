#!/usr/bin/env python3
"""
Generate ORBench v2 inputs for 2D heat diffusion (Jacobi stencil).

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import sys
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"H": 128,  "W": 128,  "T": 100, "seed": 42},
    "medium": {"H": 512,  "W": 512,  "T": 120, "seed": 42},
    "large":  {"H": 1024, "W": 1024, "T": 160, "seed": 42},
}

ALPHA = 0.20


def make_initial_grid(H, W, seed):
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    base = 0.15 + 0.25 * xx + 0.15 * yy
    grid = base.astype(np.float32)

    # Add several smooth hot spots.
    num_spots = 6
    for _ in range(num_spots):
        cy = rng.uniform(0.1, 0.9)
        cx = rng.uniform(0.1, 0.9)
        amp = rng.uniform(0.4, 1.2)
        sigma = rng.uniform(0.03, 0.10)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        grid += (amp * np.exp(-dist2 / (2.0 * sigma * sigma))).astype(np.float32)

    # Fix boundaries to a colder frame with mild variation.
    top = 0.05 + 0.02 * x
    bottom = 0.08 + 0.01 * x
    left = 0.04 + 0.01 * y
    right = 0.06 + 0.015 * y
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[:, 0] = left
    grid[:, -1] = right
    return grid.astype(np.float32)


def jacobi_reference(u0, T, alpha):
    cur = u0.astype(np.float32).copy()
    nxt = cur.copy()
    for _ in range(T):
        nxt[0, :] = cur[0, :]
        nxt[-1, :] = cur[-1, :]
        nxt[:, 0] = cur[:, 0]
        nxt[:, -1] = cur[:, -1]
        nxt[1:-1, 1:-1] = cur[1:-1, 1:-1] + alpha * (
            cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:] - 4.0 * cur[1:-1, 1:-1]
        )
        cur, nxt = nxt, cur
    return cur


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "heat_diffusion_2d" / "solution_cpu"
    src = orbench_root / "tasks" / "heat_diffusion_2d" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "heat_diffusion_2d" / "task_io_cpu.c"
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
        "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu(exe: Path, data_dir: Path, validate=False, warmup=None, trials=None):
    cmd = [str(exe), str(data_dir)]
    if validate:
        cmd.append("--validate")
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if trials is not None:
        cmd += ["--trials", str(trials)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


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
    H, W, T, seed = cfg["H"], cfg["W"], cfg["T"], cfg["seed"]
    alpha_x1e6 = int(round(ALPHA * 1_000_000))

    print(f"[gen_data] Generating {size_name}: H={H}, W={W}, T={T}")
    u0 = make_initial_grid(H, W, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("u0", "float32", u0.reshape(-1))],
        params={"H": H, "W": W, "T": T, "alpha_x1e6": alpha_x1e6},
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print("  Computing expected output via NumPy reference...")
        out = jacobi_reference(u0, T, ALPHA).reshape(-1)
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in out:
                f.write(f"{float(v):.8e}\n")

        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_ms = run_cpu(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{cpu_ms:.3f}\n")

        # also emit output.txt + timing.json through the CPU harness for local validation
        run_cpu(exe, out_dir, validate=True)
        print(f"  CPU baseline mean time: {cpu_ms:.3f} ms")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
