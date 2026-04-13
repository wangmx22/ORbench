#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate DBSCAN clustering data.

Generates clustered 2D point data with known cluster structure.
Points are arranged in Gaussian clusters so that DBSCAN with the right
epsilon/minPts will discover meaningful clusters.

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
    "small":  {"N": 10000,  "eps_x10000": 300, "minPts": 4, "seed": 42,
               "num_clusters": 15, "cluster_std": 0.02, "noise_frac": 0.05},
    "medium": {"N": 100000, "eps_x10000": 100, "minPts": 8, "seed": 42,
               "num_clusters": 30, "cluster_std": 0.008, "noise_frac": 0.05},
    "large":  {"N": 500000, "eps_x10000": 50,  "minPts": 8, "seed": 42,
               "num_clusters": 50, "cluster_std": 0.004, "noise_frac": 0.05},
}


def generate_clustered_points(N, num_clusters, cluster_std, noise_frac, seed):
    """Generate 2D points in Gaussian clusters + uniform noise."""
    rng = np.random.RandomState(seed)

    n_noise = int(N * noise_frac)
    n_clustered = N - n_noise
    pts_per_cluster = n_clustered // num_clusters
    remainder = n_clustered - pts_per_cluster * num_clusters

    xs = np.zeros(N, dtype=np.float32)
    ys = np.zeros(N, dtype=np.float32)

    # Generate cluster centers uniformly in [0, 1]
    centers_x = rng.uniform(0.05, 0.95, num_clusters)
    centers_y = rng.uniform(0.05, 0.95, num_clusters)

    idx = 0
    for c in range(num_clusters):
        n = pts_per_cluster + (1 if c < remainder else 0)
        xs[idx:idx+n] = rng.normal(centers_x[c], cluster_std, n).astype(np.float32)
        ys[idx:idx+n] = rng.normal(centers_y[c], cluster_std, n).astype(np.float32)
        idx += n

    # Uniform noise
    xs[idx:idx+n_noise] = rng.uniform(0, 1, n_noise).astype(np.float32)
    ys[idx:idx+n_noise] = rng.uniform(0, 1, n_noise).astype(np.float32)
    idx += n_noise

    # Shuffle
    perm = rng.permutation(N)
    xs = xs[perm]
    ys = ys[perm]

    return xs, ys


def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "dbscan"
    exe = task_dir / "solution_cpu"
    src = task_dir / "cpu_reference.c"
    task_io = task_dir / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

    sources = [src, task_io, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass

    cmd = ["gcc", "-O2", "-I", str(orbench_root / "framework"),
           str(harness), str(task_io), str(src),
           "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=600)
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
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


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

    print(f"[gen_data] Generating {size_name}: {N} points, "
          f"eps={cfg['eps_x10000']/10000.0}, minPts={cfg['minPts']}")

    xs, ys = generate_clustered_points(
        N, cfg["num_clusters"], cfg["cluster_std"], cfg["noise_frac"], cfg["seed"])

    print(f"  {N} points in [{xs.min():.4f},{xs.max():.4f}] x [{ys.min():.4f},{ys.max():.4f}]")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("xs", "float32", xs),
            ("ys", "float32", ys),
        ],
        params={
            "N": N,
            "eps_x10000": cfg["eps_x10000"],
            "minPts": cfg["minPts"],
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

        # Print cluster stats
        labels = [int(l.strip()) for l in open(out_dir / "expected_output.txt")]
        n_noise = sum(1 for l in labels if l == -2)
        n_clusters = len(set(l for l in labels if l > 0))
        print(f"  Clusters: {n_clusters}, Noise: {n_noise}/{N} ({100*n_noise/N:.1f}%)")

    print(f"[gen_data] {size_name}: wrote all files in {out_dir}")


if __name__ == "__main__":
    main()
