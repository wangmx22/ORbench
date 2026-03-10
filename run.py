#!/usr/bin/env python3
"""
ORBench - General-Purpose CPU-to-CUDA Acceleration Benchmark for LLMs

Usage:
    # Generate solutions
    python run.py generate --task bellman_ford --model claude-sonnet-4-20250514 --level 2

    # Evaluate a run
    python run.py eval --run claude_sonnet_4_l2 --gpus 1

    # Analyze results
    python run.py analyze --run claude_sonnet_4_l2

    # List available tasks
    python run.py list
"""

import sys
import argparse


def cmd_list(args):
    from framework.task import load_all_tasks
    tasks = load_all_tasks()
    print(f"\nAvailable tasks ({len(tasks)}):\n")
    print(f"  {'ID':<25s} {'Category':<20s} {'Diff':>4s}  Tags")
    print(f"  {'-'*70}")
    for t in tasks:
        print(f"  {t.task_id:<25s} {t.category:<20s} {'*'*t.difficulty:>4s}  {', '.join(t.tags)}")
    print()


def cmd_generate(args):
    import os
    from framework.generate import generate_solutions
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: Set --api-key or LLM_API_KEY environment variable")
        sys.exit(1)

    generate_solutions(
        task_id=args.task,
        model=args.model,
        level=args.level,
        num_samples=args.samples,
        api_key=api_key,
        api_base=args.api_base,
        run_name=args.run_name,
    )


def cmd_eval(args):
    from framework.batch_eval import batch_eval
    batch_eval(
        run_name=args.run,
        task_ids=args.tasks,
        arch=args.arch,
        num_gpu_devices=args.gpus,
        timeout=args.timeout,
        run_nsys=not args.no_nsys,
        save_nsys_csv=args.save_nsys,
    )


def cmd_analyze(args):
    from framework.analyze import compute_summary, print_summary
    import json
    summary = compute_summary(args.run)
    print_summary(summary)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ORBench CLI")
    subparsers = parser.add_subparsers(dest="command")

    # list
    subparsers.add_parser("list", help="List available tasks")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate CUDA solutions using LLMs")
    p_gen.add_argument("--task", required=True)
    p_gen.add_argument("--model", required=True)
    p_gen.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    p_gen.add_argument("--samples", type=int, default=3)
    p_gen.add_argument("--api-key", default=None)
    p_gen.add_argument("--api-base", default=None)
    p_gen.add_argument("--run-name", default=None)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate generated solutions")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--tasks", nargs="*", default=None)
    p_eval.add_argument("--arch", default="sm_89")
    p_eval.add_argument("--gpus", type=int, default=1)
    p_eval.add_argument("--timeout", type=int, default=180)
    p_eval.add_argument("--no-nsys", action="store_true", help="Skip nsys profiling entirely")
    p_eval.add_argument("--save-nsys", action="store_true", help="Save nsys CSV and summary to run directory")

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Analyze evaluation results")
    p_ana.add_argument("--run", required=True)
    p_ana.add_argument("--output", default=None, help="Save summary JSON")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()