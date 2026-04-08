#!/usr/bin/env python3
"""
Generate ORBench inputs for batched Viterbi HMM decoding.

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
    "small":  {"B": 256,  "T": 64,  "H": 32, "V": 16, "seed": 42},
    "medium": {"B": 768,  "T": 96,  "H": 32, "V": 20, "seed": 42},
    "large":  {"B": 1536, "T": 128, "H": 32, "V": 24, "seed": 42},
}


def random_log_probs(rng: np.random.Generator, shape):
    x = rng.random(shape, dtype=np.float64) + 1e-3
    x /= x.sum(axis=-1, keepdims=True)
    return np.log(x).astype(np.float32)


def make_instance(B: int, T: int, H: int, V: int, seed: int):
    rng = np.random.default_rng(seed)
    log_init = random_log_probs(rng, (H,)).reshape(H)
    log_trans = random_log_probs(rng, (H, H)).reshape(H * H)
    log_emit = random_log_probs(rng, (H, V)).reshape(H * V)
    observations = rng.integers(0, V, size=(B, T), dtype=np.int32).reshape(B * T)
    return log_init, log_trans, log_emit, observations


def viterbi_reference(B: int, T: int, H: int, V: int,
                      log_init: np.ndarray,
                      log_trans: np.ndarray,
                      log_emit: np.ndarray,
                      observations: np.ndarray) -> np.ndarray:
    log_trans = log_trans.reshape(H, H).astype(np.float32, copy=False)
    log_emit = log_emit.reshape(H, V).astype(np.float32, copy=False)
    obs = observations.reshape(B, T)

    dp = log_init[None, :] + log_emit[:, obs[:, 0]].T  # [B, H]
    back = np.empty((B, T, H), dtype=np.int16 if H < 32768 else np.int32)
    back[:, 0, :] = -1

    for t in range(1, T):
        # scores[b, i, j] = dp[b, i] + log_trans[i, j]
        scores = dp[:, :, None] + log_trans[None, :, :]
        best_prev = np.argmax(scores, axis=1)
        dp = np.take_along_axis(scores, best_prev[:, None, :], axis=1)[:, 0, :]
        dp = dp + log_emit[:, obs[:, t]].T
        back[:, t, :] = best_prev.astype(back.dtype, copy=False)

    best_last = np.argmax(dp, axis=1).astype(np.int32)
    out = np.empty((B, T), dtype=np.int32)
    out[:, T - 1] = best_last
    cur = best_last
    for t in range(T - 1, 0, -1):
        cur = back[np.arange(B), t, cur].astype(np.int32, copy=False)
        out[:, t - 1] = cur
    return out.reshape(B * T)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "batched_viterbi_hmm" / "solution_cpu"
    src = orbench_root / "tasks" / "batched_viterbi_hmm" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "batched_viterbi_hmm" / "task_io_cpu.c"
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
    B, T, H, V, seed = cfg["B"], cfg["T"], cfg["H"], cfg["V"], cfg["seed"]
    print(f"[gen_data] Generating {size_name}: B={B}, T={T}, H={H}, V={V}")

    log_init, log_trans, log_emit, observations = make_instance(B, T, H, V, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("log_init", "float32", log_init),
            ("log_trans", "float32", log_trans),
            ("log_emit", "float32", log_emit),
            ("observations", "int32", observations),
        ],
        params={"B": B, "T": T, "H": H, "V": V},
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print("  Computing expected output via NumPy Viterbi reference...")
        out = viterbi_reference(B, T, H, V, log_init, log_trans, log_emit, observations)
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
