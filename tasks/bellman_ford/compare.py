#!/usr/bin/env python3
"""
compare.py - Compare GPU output against CPU reference for Bellman-Ford
Usage: python compare.py <cpu_output_file> <gpu_output_file>

Exits 0 if correct, 1 if mismatch.
"""

import sys

ATOL = 0.01
RTOL = 0.01
INF_VAL = 1e30


def parse_result_line(filepath):
    """Parse a RESULT: line from output file"""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("RESULT:"):
                values_str = line[len("RESULT:"):].strip()
                values = []
                for v in values_str.split():
                    if v.upper() == "INF":
                        values.append(INF_VAL)
                    else:
                        values.append(float(v))
                return values
    return None


def compare(cpu_vals, gpu_vals):
    """Compare two distance arrays with tolerance"""
    if len(cpu_vals) != len(gpu_vals):
        print(f"MISMATCH: length differs ({len(cpu_vals)} vs {len(gpu_vals)})")
        return False

    mismatches = 0
    max_err = 0.0

    for i, (c, g) in enumerate(zip(cpu_vals, gpu_vals)):
        c_inf = c >= INF_VAL
        g_inf = g >= INF_VAL

        if c_inf != g_inf:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at [{i}]: cpu={'INF' if c_inf else c:.4f}, gpu={'INF' if g_inf else g:.4f}")
            continue

        if c_inf:
            continue

        err = abs(c - g)
        rel_err = err / max(abs(c), 1e-10)

        if err > ATOL and rel_err > RTOL:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at [{i}]: cpu={c:.4f}, gpu={g:.4f}, err={err:.6f}")

        max_err = max(max_err, err)

    if mismatches > 0:
        print(f"FAIL: {mismatches} mismatches out of {len(cpu_vals)} values (max_err={max_err:.6f})")
        return False
    else:
        print(f"PASS: {len(cpu_vals)} values match (max_err={max_err:.6f})")
        return True


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cpu_output> <gpu_output>")
        sys.exit(1)

    cpu_file, gpu_file = sys.argv[1], sys.argv[2]

    cpu_vals = parse_result_line(cpu_file)
    gpu_vals = parse_result_line(gpu_file)

    if cpu_vals is None:
        print(f"ERROR: No RESULT line found in {cpu_file}")
        sys.exit(1)

    if gpu_vals is None:
        print(f"ERROR: No RESULT line found in {gpu_file}")
        sys.exit(1)

    if compare(cpu_vals, gpu_vals):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
