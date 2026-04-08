#!/usr/bin/env python3
"""
Generate ORBench inputs for PageRank power iteration.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import sys
import re
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 4096,   "K": 8,  "iters": 30, "seed": 42},
    "medium": {"N": 32768,  "K": 16, "iters": 40, "seed": 42},
    "large":  {"N": 131072, "K": 16, "iters": 50, "seed": 42},
}

DAMPING = 0.85


def build_random_graph(N: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    # Fixed out-degree K for every node. Multi-edges are allowed; this keeps generation simple,
    # deterministic, and still valid for a PageRank benchmark.
    dst = rng.integers(0, N, size=(N, K), dtype=np.int32)
    src = np.repeat(np.arange(N, dtype=np.int32), K)
    dst_flat = dst.reshape(-1)

    indeg = np.bincount(dst_flat, minlength=N).astype(np.int32)
    row_ptr_in = np.empty(N + 1, dtype=np.int32)
    row_ptr_in[0] = 0
    np.cumsum(indeg, out=row_ptr_in[1:])

    order = np.argsort(dst_flat, kind="stable")
    col_ind_in = src[order].astype(np.int32, copy=False)
    inv_out_deg = np.full(N, 1.0 / float(K), dtype=np.float32)
    return src, dst_flat, row_ptr_in, col_ind_in, inv_out_deg


def pagerank_reference(N: int, src_flat, dst_flat, inv_out_deg, iters: int, damping: float):
    rank = np.full(N, 1.0 / float(N), dtype=np.float64)
    base = (1.0 - damping) / float(N)
    for _ in range(iters):
        contrib = rank[src_flat] * inv_out_deg[src_flat]
        summed = np.bincount(dst_flat, weights=contrib, minlength=N)
        rank = base + damping * summed
    return rank.astype(np.float32)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "pagerank_power_iteration" / "solution_cpu"
    src = orbench_root / "tasks" / "pagerank_power_iteration" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "pagerank_power_iteration" / "task_io_cpu.c"
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
    N, K, iters, seed = cfg["N"], cfg["K"], cfg["iters"], cfg["seed"]
    M = N * K
    damping_x1e6 = int(round(DAMPING * 1_000_000))

    print(f"[gen_data] Generating {size_name}: N={N}, K={K}, M={M}, iters={iters}")
    src_flat, dst_flat, row_ptr_in, col_ind_in, inv_out_deg = build_random_graph(N, K, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("row_ptr_in", "int32", row_ptr_in),
            ("col_ind_in", "int32", col_ind_in),
            ("inv_out_deg", "float32", inv_out_deg),
        ],
        params={"N": N, "M": M, "iters": iters, "damping_x1e6": damping_x1e6},
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print("  Computing expected output via NumPy reference...")
        out = pagerank_reference(N, src_flat, dst_flat, inv_out_deg.astype(np.float64), iters, DAMPING)
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in out:
                f.write(f"{float(v):.8e}\n")

        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_ms = run_cpu(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{cpu_ms:.3f}\n")

        run_cpu(exe, out_dir, validate=True)
        print(f"  CPU baseline mean time: {cpu_ms:.3f} ms")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
