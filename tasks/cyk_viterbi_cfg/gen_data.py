#!/usr/bin/env python3
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
    "small":  {"B": 10, "N": 12, "V": 32, "L": 28, "seed": 42},
    "medium": {"B": 8,  "N": 15, "V": 48, "L": 40, "seed": 42},
    "large":  {"B": 8,  "N": 18, "V": 64, "L": 48, "seed": 42},
}

def generate_instance(B, N, V, L, seed):
    rng = np.random.default_rng(seed)
    # Dense weighted grammar, kept in a moderate range to avoid overflow.
    binary_scores = rng.integers(-7, 8, size=(N, N, N), dtype=np.int32)
    unary_scores = rng.integers(-9, 10, size=(N, V), dtype=np.int32)
    # Bias start symbol slightly so answers are nontrivial but stable.
    binary_scores[0] += 1
    unary_scores[0] += 1
    tokens = rng.integers(0, V, size=(B, L), dtype=np.int32)
    return binary_scores, unary_scores, tokens

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "cyk_viterbi_cfg" / "solution_cpu"
    src = orbench_root / "tasks" / "cyk_viterbi_cfg" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "cyk_viterbi_cfg" / "task_io_cpu.c"
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
    B, N, V, L, seed = cfg["B"], cfg["N"], cfg["V"], cfg["L"], cfg["seed"]
    binary_scores, unary_scores, tokens = generate_instance(B, N, V, L, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("binary_scores", "int32", binary_scores.reshape(-1).astype(np.int32)),
            ("unary_scores", "int32", unary_scores.reshape(-1).astype(np.int32)),
            ("tokens", "int32", tokens.reshape(-1).astype(np.int32)),
        ],
        params={"B": int(B), "N": int(N), "V": int(V), "L": int(L)}
    )
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")
    print(f"[gen_data] {size_name}: B={B}, N={N}, V={V}, L={L}, seed={seed}")
    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir, timeout=1800)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir, timeout=1800)
        print(f"[gen_data] {size_name}: CPU time={time_ms:.3f}ms")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin only")

if __name__ == "__main__":
    main()
