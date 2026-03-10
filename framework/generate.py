"""
generate.py - Call LLM APIs to generate CUDA solutions
"""

import os
import re
import json
import argparse
from typing import Optional

from .task import load_task, load_prompt, ORBENCH_ROOT


def extract_cuda_code(response_text: str) -> str:
    """
    Extract CUDA code from LLM response.
    Handles markdown code blocks (```cuda, ```cpp, ```c, or bare ```)
    """
    # Try to find fenced code blocks
    patterns = [
        r'```cuda\s*\n(.*?)```',
        r'```cpp\s*\n(.*?)```',
        r'```c\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            # Return the longest match (most likely the full solution)
            return max(matches, key=len).strip()

    # If no code block found, return the entire response
    # (LLM might have returned raw code without fences)
    return response_text.strip()


def call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int = 8192) -> str:
    """Call Anthropic API"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_openai(model: str, prompt: str, api_key: str, api_base: str = None, max_tokens: int = 8192) -> str:
    """Call OpenAI-compatible API (OpenAI, DeepSeek, etc.)"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_llm(model: str, prompt: str, api_key: str, api_base: str = None, max_tokens: int = 8192) -> str:
    """Dispatch to the appropriate LLM API based on model name"""
    if "claude" in model.lower():
        return call_anthropic(model, prompt, api_key, max_tokens)
    else:
        return call_openai(model, prompt, api_key, api_base, max_tokens)


def generate_solutions(
    task_id: str,
    model: str,
    level: int,
    num_samples: int = 3,
    api_key: str = None,
    api_base: str = None,
    run_name: str = None,
) -> list[str]:
    """
    Generate CUDA solutions for a task.
    
    Returns list of file paths to saved solutions.
    """
    task = load_task(task_id)
    prompt = load_prompt(task_id, level)

    if run_name is None:
        run_name = f"{model.replace('/', '_')}_l{level}"

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name, task_id)
    os.makedirs(run_dir, exist_ok=True)

    saved_paths = []
    for i in range(num_samples):
        output_path = os.path.join(run_dir, f"sample_{i}.cu")

        # Skip if already generated
        if os.path.exists(output_path):
            print(f"  [SKIP] {task_id} sample_{i} already exists")
            saved_paths.append(output_path)
            continue

        print(f"  [GEN] {task_id} sample_{i} with {model}...")
        try:
            response = call_llm(model, prompt, api_key, api_base)
            code = extract_cuda_code(response)

            with open(output_path, "w") as f:
                f.write(code)

            # Also save raw response for debugging
            raw_path = os.path.join(run_dir, f"sample_{i}_raw.txt")
            with open(raw_path, "w") as f:
                f.write(response)

            saved_paths.append(output_path)
            print(f"  [OK] Saved to {output_path}")

        except Exception as e:
            print(f"  [ERROR] {task_id} sample_{i}: {e}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate CUDA solutions using LLMs")
    parser.add_argument("--task", required=True, help="Task ID (e.g., bellman_ford)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    generate_solutions(
        task_id=args.task,
        model=args.model,
        level=args.level,
        num_samples=args.samples,
        api_key=args.api_key,
        api_base=args.api_base,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
