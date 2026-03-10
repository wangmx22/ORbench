#!/usr/bin/env python3
"""
gen_data.py - Generate random graph and CPU reference output for Bellman-Ford
Usage: python gen_data.py <size_name> <output_dir>
"""

import sys
import os
import json
import struct
import array
import random

SIZES = {
    "small":  {"V": 1000,   "E": 5000,    "seed": 42},
    "medium": {"V": 100000, "E": 500000,  "seed": 42},
    "large":  {"V": 500000, "E": 2500000, "seed": 42},
}

INF_VAL = 1e30


def generate_graph(V, E, seed):
    """Generate random directed graph in CSR format"""
    random.seed(seed)

    edges = []
    while len(edges) < E:
        u = random.randint(0, V - 1)
        v = random.randint(0, V - 1)
        if u == v:
            continue
        w = 1.0 + random.random() * 99.0
        edges.append((u, v, w))

    # Sort by source for CSR
    edges.sort(key=lambda e: e[0])

    row_offsets = [0] * (V + 1)
    col_indices = []
    weights = []

    for u, v, w in edges:
        row_offsets[u + 1] += 1
        col_indices.append(v)
        weights.append(w)

    for i in range(1, V + 1):
        row_offsets[i] += row_offsets[i - 1]

    return row_offsets, col_indices, weights


def bellman_ford_cpu(V, row_offsets, col_indices, weights, source=0):
    """CPU reference implementation"""
    dist = [INF_VAL] * V
    dist[source] = 0.0

    for r in range(V - 1):
        updated = False
        for u in range(V):
            if dist[u] >= INF_VAL:
                continue
            for idx in range(row_offsets[u], row_offsets[u + 1]):
                v = col_indices[idx]
                nd = dist[u] + weights[idx]
                if nd < dist[v]:
                    dist[v] = nd
                    updated = True
        if not updated:
            break

    return dist


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <size_name> <output_dir>")
        print(f"  size_name: {list(SIZES.keys())}")
        sys.exit(1)

    size_name = sys.argv[1]
    output_dir = sys.argv[2]

    if size_name not in SIZES:
        print(f"Unknown size: {size_name}. Options: {list(SIZES.keys())}")
        sys.exit(1)

    params = SIZES[size_name]
    V, E, seed = params["V"], params["E"], params["seed"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating graph: V={V}, E={E}, seed={seed}")
    row_offsets, col_indices, weights = generate_graph(V, E, seed)

    print(f"Running CPU Bellman-Ford...")
    dist = bellman_ford_cpu(V, row_offsets, col_indices, weights)

    reachable = sum(1 for d in dist if d < INF_VAL)
    print(f"  Reachable: {reachable}/{V}")

    # =============================================
    # Save INPUT: graph data in binary format
    # =============================================
    # Binary files for fast loading by GPU programs
    # Format: raw arrays of int32 / float32

    # row_offsets: (V+1) int32 values
    input_path = os.path.join(output_dir, "row_offsets.bin")
    with open(input_path, "wb") as f:
        array.array("i", row_offsets).tofile(f)

    # col_indices: E int32 values
    input_path = os.path.join(output_dir, "col_indices.bin")
    with open(input_path, "wb") as f:
        array.array("i", col_indices).tofile(f)

    # weights: E float32 values
    input_path = os.path.join(output_dir, "weights.bin")
    with open(input_path, "wb") as f:
        array.array("f", weights).tofile(f)

    # Also save a human-readable text version of graph params
    # (GPU programs can use either binary or generate internally from seed)
    input_txt_path = os.path.join(output_dir, "input.txt")
    with open(input_txt_path, "w") as f:
        f.write(f"{V} {E} 0 {seed}\n")  # V E source seed

    print(f"  Input saved: row_offsets.bin ({(V+1)*4} bytes), "
          f"col_indices.bin ({E*4} bytes), weights.bin ({E*4} bytes)")

    # =============================================
    # Save OUTPUT: expected distances
    # =============================================
    output_path = os.path.join(output_dir, "expected_output.txt")
    with open(output_path, "w") as f:
        parts = []
        for d in dist:
            if d >= INF_VAL:
                parts.append("INF")
            else:
                parts.append(f"{d:.6f}")
        f.write("RESULT: " + " ".join(parts) + "\n")

    # Also save distances as binary for fast comparison
    dist_bin_path = os.path.join(output_dir, "expected_dist.bin")
    with open(dist_bin_path, "wb") as f:
        array.array("f", [d if d < INF_VAL else 1e30 for d in dist]).tofile(f)

    print(f"  Output saved: expected_output.txt, expected_dist.bin ({V*4} bytes)")

    # =============================================
    # Save META: parameters for reproducibility
    # =============================================
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "V": V, "E": E, "seed": seed, "source": 0,
            "reachable": reachable,
            "files": {
                "input": {
                    "row_offsets.bin": {"dtype": "int32", "shape": [V + 1]},
                    "col_indices.bin": {"dtype": "int32", "shape": [E]},
                    "weights.bin":     {"dtype": "float32", "shape": [E]},
                    "input.txt":       {"format": "V E source seed"},
                },
                "output": {
                    "expected_output.txt": {"format": "RESULT: d0 d1 ... dV-1"},
                    "expected_dist.bin":   {"dtype": "float32", "shape": [V]},
                }
            }
        }, f, indent=2)

    print(f"  Meta saved: meta.json")
    print(f"Done. All files in {output_dir}")


if __name__ == "__main__":
    main()
