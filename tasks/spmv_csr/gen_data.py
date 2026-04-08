#!/usr/bin/env python3
"""
Generate ORBench v2 inputs for CSR sparse matrix-vector multiply.

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
    "small":  {"N": 4096,   "nnz_per_row": 16, "seed": 42},
    "medium": {"N": 32768,  "nnz_per_row": 24, "seed": 42},
    "large":  {"N": 131072, "nnz_per_row": 32, "seed": 42},
}


def make_spmv_instance(N: int, nnz_per_row: int, seed: int):
    rng = np.random.default_rng(seed)
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    col_idx = np.empty(N * nnz_per_row, dtype=np.int32)
    vals = np.empty(N * nnz_per_row, dtype=np.float32)

    x = rng.normal(loc=0.0, scale=1.0, size=N).astype(np.float32)
    # Add low-frequency structure to x so outputs are not purely random noise.
    x += np.linspace(-0.5, 0.5, N, dtype=np.float32)

    write_pos = 0
    all_cols = np.arange(N, dtype=np.int32)
    for i in range(N):
        cols = rng.choice(all_cols, size=nnz_per_row, replace=False)
        cols[0] = i  # force diagonal for stability / realism
        cols = np.unique(cols)
        # ensure exact row length after unique
        while cols.shape[0] < nnz_per_row:
            extra = int(rng.integers(0, N))
            if extra not in cols:
                cols = np.append(cols, np.int32(extra))
        cols.sort()
        row_nnz = cols.shape[0]
        row_ptr[i] = write_pos
        col_idx[write_pos:write_pos + row_nnz] = cols

        # Positive diagonal bias plus mixed-sign off-diagonal weights.
        v = rng.uniform(-0.2, 0.2, size=row_nnz).astype(np.float32)
        diag_loc = np.where(cols == i)[0]
        if diag_loc.size > 0:
            v[diag_loc[0]] += np.float32(1.5 + 0.2 * rng.random())
        vals[write_pos:write_pos + row_nnz] = v
        write_pos += row_nnz

    row_ptr[N] = write_pos
    return row_ptr, col_idx[:write_pos].copy(), vals[:write_pos].copy(), x


def spmv_reference(row_ptr, col_idx, vals, x):
    N = row_ptr.shape[0] - 1
    y = np.empty(N, dtype=np.float32)
    for i in range(N):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        y[i] = np.dot(vals[start:end], x[col_idx[start:end]])
    return y


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "spmv_csr" / "solution_cpu"
    src = orbench_root / "tasks" / "spmv_csr" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "spmv_csr" / "task_io_cpu.c"
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
    N = cfg["N"]
    nnz_per_row = cfg["nnz_per_row"]
    seed = cfg["seed"]

    print(f"[gen_data] Generating {size_name}: N={N}, nnz_per_row={nnz_per_row}")
    row_ptr, col_idx, vals, x = make_spmv_instance(N, nnz_per_row, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("row_ptr", "int32", row_ptr),
            ("col_idx", "int32", col_idx),
            ("vals", "float32", vals),
            ("x", "float32", x),
        ],
        params={"N": N},
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print("  Computing expected output via NumPy reference...")
        y = spmv_reference(row_ptr, col_idx, vals, x)
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in y:
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
