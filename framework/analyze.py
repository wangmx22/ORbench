"""
analyze.py - Aggregate evaluation results and generate analysis
"""

import os
import json
import argparse
from collections import defaultdict

from .task import load_task, load_all_tasks, ORBENCH_ROOT


def load_eval_results(run_name: str) -> dict:
    """Load eval_results.json for a run"""
    path = os.path.join(ORBENCH_ROOT, "runs", run_name, "eval_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def compute_summary(run_name: str) -> dict:
    """
    Compute aggregate metrics for a run.
    
    Returns:
        Dict with per-task and overall statistics
    """
    results = load_eval_results(run_name)

    # Group by task
    by_task = defaultdict(list)
    for key, result in results.items():
        task_id = result["task_id"]
        by_task[task_id].append(result)

    summary = {"tasks": {}, "overall": {}}

    total_compiled = 0
    total_correct = 0
    total_samples = 0
    all_speedups = []

    for task_id, samples in by_task.items():
        try:
            task = load_task(task_id)
        except Exception:
            task = None

        n = len(samples)
        compiled = sum(1 for s in samples if s.get("compiled", False))
        correct = sum(1 for s in samples if s.get("correct", False))

        speedups = []
        kernel_speedups = []
        gpu_utils = []

        for s in samples:
            bench = s.get("benchmark")
            if bench:
                sp = bench.get("speedup_e2e", -1)
                if sp > 0:
                    speedups.append(sp)

                ksp = bench.get("speedup_kernel")
                if ksp and ksp > 0:
                    kernel_speedups.append(ksp)

                gu = bench.get("gpu_utilization")
                if gu and gu > 0:
                    gpu_utils.append(gu)

        task_summary = {
            "num_samples": n,
            "compiled": compiled,
            "correct": correct,
            "compile_rate": compiled / n if n > 0 else 0,
            "pass_rate": correct / n if n > 0 else 0,
            "category": task.category if task else "unknown",
            "difficulty": task.difficulty if task else 0,
        }

        if speedups:
            task_summary["best_speedup_e2e"] = max(speedups)
            task_summary["avg_speedup_e2e"] = sum(speedups) / len(speedups)
            all_speedups.extend(speedups)

        if kernel_speedups:
            task_summary["best_speedup_kernel"] = max(kernel_speedups)

        if gpu_utils:
            task_summary["avg_gpu_utilization"] = sum(gpu_utils) / len(gpu_utils)

        summary["tasks"][task_id] = task_summary

        total_compiled += compiled
        total_correct += correct
        total_samples += n

    # Overall
    summary["overall"] = {
        "total_tasks": len(by_task),
        "total_samples": total_samples,
        "compile_rate": total_compiled / total_samples if total_samples > 0 else 0,
        "pass_rate": total_correct / total_samples if total_samples > 0 else 0,
    }

    if all_speedups:
        summary["overall"]["avg_speedup"] = sum(all_speedups) / len(all_speedups)
        summary["overall"]["median_speedup"] = sorted(all_speedups)[len(all_speedups) // 2]

    # fast_p: fraction of correct samples with speedup > p
    for p in [1.0, 1.5, 2.0, 5.0, 10.0]:
        fast = sum(1 for sp in all_speedups if sp >= p)
        summary["overall"][f"fast_{p}"] = fast / total_samples if total_samples > 0 else 0

    # By category
    by_category = defaultdict(list)
    for task_id, ts in summary["tasks"].items():
        cat = ts.get("category", "unknown")
        by_category[cat].append(ts)

    summary["by_category"] = {}
    for cat, task_summaries in by_category.items():
        n_tasks = len(task_summaries)
        avg_pass = sum(t["pass_rate"] for t in task_summaries) / n_tasks
        speedups_in_cat = [t["best_speedup_e2e"] for t in task_summaries if "best_speedup_e2e" in t]
        summary["by_category"][cat] = {
            "num_tasks": n_tasks,
            "avg_pass_rate": avg_pass,
            "avg_best_speedup": sum(speedups_in_cat) / len(speedups_in_cat) if speedups_in_cat else 0,
        }

    return summary


def print_summary(summary: dict):
    """Pretty-print evaluation summary"""
    overall = summary["overall"]

    print(f"\n{'='*70}")
    print(f"  ORBench Evaluation Summary")
    print(f"{'='*70}")
    print(f"  Tasks: {overall['total_tasks']}  |  Samples: {overall['total_samples']}")
    print(f"  Compile rate: {overall['compile_rate']:.1%}")
    print(f"  Pass rate:    {overall['pass_rate']:.1%}")
    if "avg_speedup" in overall:
        print(f"  Avg speedup:  {overall['avg_speedup']:.1f}x")
    print()

    # fast_p table
    print("  fast_p metrics:")
    for key in sorted(k for k in overall if k.startswith("fast_")):
        p = key.replace("fast_", "")
        print(f"    fast_{p}: {overall[key]:.1%}")
    print()

    # By category
    print(f"  {'Category':<25s} {'Tasks':>5s} {'Pass%':>8s} {'Avg Speedup':>12s}")
    print(f"  {'-'*50}")
    for cat, data in sorted(summary["by_category"].items()):
        sp_str = f"{data['avg_best_speedup']:.1f}x" if data["avg_best_speedup"] > 0 else "N/A"
        print(f"  {cat:<25s} {data['num_tasks']:>5d} {data['avg_pass_rate']:>7.1%} {sp_str:>12s}")
    print()

    # Per-task detail
    print(f"  {'Task':<25s} {'Diff':>4s} {'Comp':>5s} {'Pass':>5s} {'Speedup(e2e)':>13s} {'GPU Util':>9s}")
    print(f"  {'-'*65}")
    for task_id, ts in sorted(summary["tasks"].items()):
        comp = f"{ts['compiled']}/{ts['num_samples']}"
        corr = f"{ts['correct']}/{ts['num_samples']}"
        sp = f"{ts['best_speedup_e2e']:.1f}x" if "best_speedup_e2e" in ts else "N/A"
        gu = f"{ts['avg_gpu_utilization']:.0%}" if "avg_gpu_utilization" in ts else "N/A"
        print(f"  {task_id:<25s} {'*'*ts['difficulty']:>4s} {comp:>5s} {corr:>5s} {sp:>13s} {gu:>9s}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ORBench results")
    parser.add_argument("--run", required=True, help="Run name")
    parser.add_argument("--output", default=None, help="Save summary JSON to file")
    args = parser.parse_args()

    summary = compute_summary(args.run)
    print_summary(summary)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
