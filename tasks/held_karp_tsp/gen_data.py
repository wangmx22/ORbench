#!/usr/bin/env python3
"""
Generate batched Euclidean TSP instances for Held-Karp exact DP.
"""
import sys
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"B": 8, "n": 15, "seed": 42},
    "medium": {"B": 8, "n": 18, "seed": 42},
    "large":  {"B": 4, "n": 21, "seed": 42},
}

def generate_cost_matrices(B, n, seed):
    rng = np.random.default_rng(seed)
    costs = np.zeros((B, n, n), dtype=np.int32)
    for b in range(B):
        pts = rng.uniform(0.0, 1000.0, size=(n, 2))
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        mat = np.rint(dist).astype(np.int32)
        np.fill_diagonal(mat, 0)
        mat = np.maximum(mat, 1)
        np.fill_diagonal(mat, 0)
        costs[b] = mat
    return costs

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "held_karp_tsp" / "solution_cpu"
    src = orbench_root / "tasks" / "held_karp_tsp" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "held_karp_tsp" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    cmd = [
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe

def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 1800) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))

def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int = 1800) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SIZES[size_name]
    B, n, seed = cfg["B"], cfg["n"], cfg["seed"]
    costs = generate_cost_matrices(B, n, seed)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("costs", "int32", costs.reshape(-1).astype(np.int32))],
        params={"B": int(B), "n": int(n)}
    )
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")
    print(f"[gen_data] {size_name}: B={B}, n={n}, seed={seed}")
    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        timeout = 1800
        time_ms = run_cpu_time(exe, out_dir, timeout=timeout)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir, timeout=timeout)
        print(f"[gen_data] {size_name}: CPU time={time_ms:.3f}ms")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin only")

if __name__ == "__main__":
    main()
