"""
batch_eval.py - Batch evaluation across tasks, samples, and GPUs
Orchestrates the full pipeline: compile → validate → benchmark → save results
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from .task import load_task, load_all_tasks, ORBENCH_ROOT
from .compile import compile_solution
from .validate import validate_solution, ValidationResult
from .benchmark import benchmark_solution, BenchmarkResult


@dataclass
class EvalResult:
    task_id: str
    sample_id: int
    compiled: bool = False
    compile_error: str = ""
    correct: bool = False
    correctness_detail: dict = None
    benchmark: dict = None
    error: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


def eval_single_sample(
    task_id: str,
    sample_path: str,
    sample_id: int,
    device_id: int = 0,
    arch: str = "sm_89",
    run_nsys: bool = True,
    save_nsys_csv: bool = False,
) -> EvalResult:
    """
    Evaluate a single solution sample: compile → validate → benchmark.
    Runs in a subprocess to isolate CUDA context.
    """
    result = EvalResult(task_id=task_id, sample_id=sample_id)

    # Step 1: Compile
    compile_result = compile_solution(task_id, sample_path, arch=arch)
    result.compiled = compile_result.success

    if not compile_result.success:
        result.compile_error = compile_result.stderr[:500]
        return result

    exe_path = compile_result.executable_path

    # Step 2: Validate correctness
    try:
        val_result = validate_solution(task_id, exe_path, device_id=device_id)
        result.correct = val_result.correct
        result.correctness_detail = val_result.results_by_size
        if val_result.error:
            result.error += val_result.error
    except Exception as e:
        result.error += f"Validation error: {str(e)[:200]}. "
        return result

    # Step 3: Benchmark (only if correct)
    if result.correct:
        try:
            # Save nsys outputs in a dedicated subfolder: runs/.../bellman_ford/nsys_sample_0/
            nsys_csv_dir = None
            if save_nsys_csv:
                nsys_csv_dir = os.path.join(os.path.dirname(sample_path), f"nsys_sample_{sample_id}")

            bench_result = benchmark_solution(
                task_id, exe_path, device_id=device_id,
                run_nsys=run_nsys,
                save_nsys_csv=save_nsys_csv,
                save_nsys_csv_dir=nsys_csv_dir,
            )
            result.benchmark = asdict(bench_result)
        except Exception as e:
            result.error += f"Benchmark error: {str(e)[:200]}. "

    return result


def _eval_worker(args):
    """Worker function for multiprocessing - already in a spawned process"""
    task_id, sample_path, sample_id, device_id, arch, run_nsys, save_nsys_csv, timeout = args

    try:
        result = eval_single_sample(
            task_id, sample_path, sample_id, device_id, arch, run_nsys, save_nsys_csv
        )
        return result
    except Exception as e:
        return EvalResult(
            task_id=task_id, sample_id=sample_id,
            error=f"Worker error: {str(e)[:200]}",
        )


def save_eval_result(result: EvalResult, eval_file_path: str):
    """Append a single eval result to the results JSON file"""
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"{result.task_id}_sample_{result.sample_id}"
    all_results[key] = result.to_dict()

    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def batch_eval(
    run_name: str,
    task_ids: list[str] = None,
    arch: str = "sm_89",
    num_gpu_devices: int = 1,
    timeout: int = 180,
    run_nsys: bool = True,
    save_nsys_csv: bool = False,
):
    """
    Batch evaluation for all samples in a run.
    
    Args:
        run_name: Name of the run directory (under runs/)
        task_ids: List of task IDs to evaluate (None = all)
        arch: GPU architecture
        num_gpu_devices: Number of GPUs to use in parallel
        timeout: Per-sample timeout
        run_nsys: Whether to run nsys profiling
        save_nsys_csv: Whether to save nsys CSV and summary to run directory
    """
    # Ensure spawn for CUDA context isolation
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name)
    eval_file = os.path.join(run_dir, "eval_results.json")

    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return

    # Discover tasks and samples
    if task_ids is None:
        task_ids = [d for d in sorted(os.listdir(run_dir))
                    if os.path.isdir(os.path.join(run_dir, d))]

    # Build work list
    work_list = []
    for task_id in task_ids:
        task_dir = os.path.join(run_dir, task_id)
        if not os.path.isdir(task_dir):
            continue

        for filename in sorted(os.listdir(task_dir)):
            if filename.startswith("sample_") and filename.endswith(".cu"):
                sample_id = int(filename.split("_")[1].split(".")[0])
                sample_path = os.path.join(task_dir, filename)

                # Check if already evaluated
                key = f"{task_id}_sample_{sample_id}"
                if os.path.exists(eval_file):
                    with open(eval_file, "r") as f:
                        existing = json.load(f)
                    if key in existing:
                        print(f"  [SKIP] {key} already evaluated")
                        continue

                work_list.append((task_id, sample_path, sample_id))

    print(f"{'='*60}")
    print(f"  ORBench Batch Evaluation")
    print(f"  Run: {run_name}")
    print(f"  Tasks: {len(task_ids)}")
    print(f"  Samples to evaluate: {len(work_list)}")
    print(f"  GPUs: {num_gpu_devices}")
    print(f"{'='*60}\n")

    if not work_list:
        print("Nothing to evaluate.")
        return

    # Phase 1: Compile all (CPU parallel, no GPU needed)
    print("[Phase 1] Compiling solutions...")
    # (compilation happens inside eval_single_sample, but could be separated)

    # Phase 2: Evaluate in GPU-parallel batches
    print(f"[Phase 2] Evaluating on {num_gpu_devices} GPU(s)...\n")

    batch_size = num_gpu_devices
    total_done = 0

    for batch_start in range(0, len(work_list), batch_size):
        batch = work_list[batch_start:batch_start + batch_size]

        # Prepare work args with device assignment
        batch_args = [
            (task_id, sample_path, sample_id,
             i % batch_size,  # device_id
             arch, run_nsys, save_nsys_csv, timeout)
            for i, (task_id, sample_path, sample_id) in enumerate(batch)
        ]

        start_time = time.time()

        with mp.Pool(batch_size) as pool:
            async_results = [pool.apply_async(_eval_worker, (args,)) for args in batch_args]

            for i, ar in enumerate(async_results):
                task_id, _, sample_id = batch[i]
                try:
                    eval_result = ar.get(timeout=timeout + 30)
                except Exception as e:
                    eval_result = EvalResult(
                        task_id=task_id, sample_id=sample_id,
                        error=f"Batch error: {str(e)[:200]}",
                    )

                # Print result
                status_parts = []
                if eval_result.compiled:
                    status_parts.append("compiled")
                else:
                    status_parts.append("COMPILE_FAIL")
                if eval_result.correct:
                    status_parts.append("correct")
                if eval_result.benchmark and eval_result.benchmark.get("speedup_e2e", -1) > 0:
                    status_parts.append(f"speedup={eval_result.benchmark['speedup_e2e']:.1f}x")

                status = " | ".join(status_parts)
                print(f"  [{task_id}/sample_{sample_id}] {status}")

                # Save incrementally
                save_eval_result(eval_result, eval_file)
                total_done += 1

        elapsed = time.time() - start_time
        print(f"  Batch done in {elapsed:.1f}s ({total_done}/{len(work_list)} total)\n")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete: {total_done} samples")
    print(f"  Results saved to: {eval_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="ORBench Batch Evaluation")
    parser.add_argument("--run", required=True, help="Run name")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task IDs to evaluate")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--save-nsys", action="store_true", help="Save nsys CSV to run dir")
    args = parser.parse_args()

    batch_eval(
        run_name=args.run,
        task_ids=args.tasks,
        arch=args.arch,
        num_gpu_devices=args.gpus,
        timeout=args.timeout,
        run_nsys=not args.no_nsys,
        save_nsys_csv=args.save_nsys,
    )


if __name__ == "__main__":
    main()