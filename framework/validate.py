"""
validate.py - Correctness validation against CPU reference
Multi-scale testing to prevent overfitting to specific input sizes.
"""

import os
import subprocess
import json
from dataclasses import dataclass, field

from .task import load_task, get_task_dir


@dataclass
class ValidationResult:
    correct: bool = False
    results_by_size: dict = field(default_factory=dict)  # size_name -> bool
    error: str = ""


def run_program(exe_path: str, args: list[str] = None, timeout: int = 180,
                env: dict = None, cwd: str = None) -> tuple[bool, str, str]:
    """
    Run a compiled executable, return (success, stdout, stderr).
    """
    cmd = [exe_path] + (args or [])
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
            env=run_env,
            cwd=cwd,
        )
        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)


def generate_test_data(task_id: str, size_name: str, output_dir: str) -> bool:
    """
    Run gen_data.py to produce test inputs for a given size.
    
    gen_data.py should accept: python gen_data.py <size_name> <output_dir>
    and write input files (e.g., input.bin) and expected output (e.g., expected_output.bin)
    """
    task_dir = get_task_dir(task_id)
    gen_script = os.path.join(task_dir, "gen_data.py")

    if not os.path.exists(gen_script):
        return False

    os.makedirs(output_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ["python", gen_script, size_name, output_dir],
            capture_output=True,
            timeout=120,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def compare_bin_files(expected_bin: str, output_bin: str, atol: float = 0.01, rtol: float = 0.01) -> tuple[bool, str]:
    """
    Compare two binary float32 files with numerical tolerance.
    Returns (passed, message).
    """
    import array

    INF_VAL = 1e30

    # Read expected
    with open(expected_bin, "rb") as f:
        cpu_arr = array.array("f")
        cpu_arr.fromfile(f, os.path.getsize(expected_bin) // 4)

    # Read output
    with open(output_bin, "rb") as f:
        gpu_arr = array.array("f")
        gpu_arr.fromfile(f, os.path.getsize(output_bin) // 4)

    if len(cpu_arr) != len(gpu_arr):
        return False, f"Length mismatch: expected {len(cpu_arr)}, got {len(gpu_arr)}"

    mismatches = 0
    max_err = 0.0

    for i in range(len(cpu_arr)):
        c, g = cpu_arr[i], gpu_arr[i]
        c_inf = c >= INF_VAL
        g_inf = g >= INF_VAL

        if c_inf != g_inf:
            mismatches += 1
            continue
        if c_inf:
            continue

        err = abs(c - g)
        rel_err = err / max(abs(c), 1e-10)

        if err > atol and rel_err > rtol:
            mismatches += 1

        max_err = max(max_err, err)

    if mismatches > 0:
        return False, f"FAIL: {mismatches}/{len(cpu_arr)} mismatches (max_err={max_err:.6f})"
    else:
        return True, f"PASS: {len(cpu_arr)} values (max_err={max_err:.6f})"


def data_exists(data_dir: str) -> bool:
    """Check if pre-generated data already exists in data_dir"""
    required = ["expected_dist.bin", "input.txt"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


def validate_solution(
    task_id: str,
    gpu_exe_path: str,
    device_id: int = 0,
) -> ValidationResult:
    """
    Validate a GPU solution against pre-generated expected output.
    
    Flow:
      1. Run GPU executable with --validate → writes output.bin
      2. Compare output.bin vs expected_dist.bin (binary float32 comparison)
    
    Data must be pre-generated via gen_data.py before running eval.
    """
    task = load_task(task_id)
    task_dir = get_task_dir(task_id)
    result = ValidationResult()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    sizes_to_test = task.input_sizes if task.input_sizes else {"default": {}}

    for size_name, size_params in sizes_to_test.items():
        # 1. Check pre-generated data exists
        data_dir = os.path.join(task_dir, "data", size_name)
        if not data_exists(data_dir):
            if not generate_test_data(task_id, size_name, data_dir):
                result.results_by_size[size_name] = False
                result.error += f"No data for size '{size_name}'. Run gen_data.py first. "
                continue

        # 2. Run GPU solution with --validate (writes output.bin)
        output_bin = os.path.join(data_dir, "output.bin")
        if os.path.exists(output_bin):
            os.remove(output_bin)  # clean previous run

        gpu_ok, gpu_stdout, gpu_stderr = run_program(
            gpu_exe_path, args=[data_dir, "--validate"], timeout=task.timeout, env=env
        )
        if not gpu_ok:
            result.results_by_size[size_name] = False
            result.error += f"GPU solution failed on size '{size_name}': {gpu_stderr[:200]}. "
            continue

        # 3. Check output.bin was created
        if not os.path.exists(output_bin):
            result.results_by_size[size_name] = False
            result.error += f"GPU solution did not produce output.bin for size '{size_name}'. "
            continue

        # 4. Compare binary files
        expected_bin = os.path.join(data_dir, "expected_dist.bin")
        passed, msg = compare_bin_files(expected_bin, output_bin, atol=task.atol, rtol=task.rtol)
        result.results_by_size[size_name] = passed
        print(f"  [{size_name}] {msg}")

        if not passed:
            result.error += f"Output mismatch on size '{size_name}': {msg}. "

    result.correct = bool(result.results_by_size) and all(result.results_by_size.values())
    return result
