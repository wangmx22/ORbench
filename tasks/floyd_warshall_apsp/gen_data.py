#!/usr/bin/env python3
"""
Generate ORBench v2 inputs for batched Floyd-Warshall APSP.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"B": 8,  "N": 96,  "seed": 42},
    "medium": {"B": 12, "N": 128, "seed": 42},
    "large":  {"B": 16, "N": 160, "seed": 42},
}

INF = 1_000_000_000


def make_graph_batch(B: int, N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = np.full((B, N, N), INF, dtype=np.int32)
    for g in range(B):
        mats[g, np.arange(N), np.arange(N)] = 0

        # Strongly connected backbone: directed ring in both directions.
        forward = rng.integers(1, 20, size=N, dtype=np.int32)
        backward = rng.integers(1, 20, size=N, dtype=np.int32)
        for i in range(N):
            mats[g, i, (i + 1) % N] = int(forward[i])
            mats[g, i, (i - 1 + N) % N] = int(backward[i])

        # Additional random directed edges.
        p = 0.35
        mask = rng.random((N, N)) < p
        np.fill_diagonal(mask, False)
        weights = rng.integers(2, 100, size=(N, N), dtype=np.int32)
        mats[g][mask] = np.minimum(mats[g][mask], weights[mask])
    return mats


def floyd_reference(batch: np.ndarray, inf: int) -> np.ndarray:
    dist = batch.astype(np.int64, copy=True)
    B, N, _ = dist.shape
    for g in range(B):
        d = dist[g]
        for k in range(N):
            dik = d[:, [k]]
            dkj = d[[k], :]
            valid = (dik < inf) & (dkj < inf)
            via = dik + dkj
            d = np.where(valid & (via < d), via, d)
        dist[g] = d
    return dist.astype(np.int32)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "floyd_warshall_apsp" / "solution_cpu"
    src = orbench_root / "tasks" / "floyd_warshall_apsp" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "floyd_warshall_apsp" / "task_io_cpu.c"
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


def run_cpu(exe: Path, data_dir: Path, validate=False, warmup=None, trials=None) -> float:
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
    B, N, seed = cfg["B"], cfg["N"], cfg["seed"]

    print(f"[gen_data] Generating {size_name}: B={B}, N={N}")
    adj = make_graph_batch(B, N, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("adj", "int32", adj.reshape(-1))],
        params={"B": B, "N": N, "INF": INF},
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print("  Computing expected output via NumPy reference...")
        out = floyd_reference(adj, INF).reshape(-1)
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in out:
                f.write(f"{int(v)}\n")

        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_ms = run_cpu(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{cpu_ms:.3f}\n")

        run_cpu(exe, out_dir, validate=True)
        print(f"  CPU baseline mean time: {cpu_ms:.3f} ms")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
