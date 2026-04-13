#!/usr/bin/env python3
"""
gen_data.py — Generate GROMACS-style cluster-based non-bonded force data.

Creates a molecular system, groups atoms into spatial clusters (size 4),
builds cluster pair lists with exclusion masks, and runs CPU baseline.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

CLUSTER_SIZE = 4

SIZES = {
    "small":  {"N": 12000,  "rcut_x100": 120, "density_x10": 333, "num_types": 2, "seed": 42},
    "medium": {"N": 100000, "rcut_x100": 100, "density_x10": 333, "num_types": 2, "seed": 42},
    "large":  {"N": 500000, "rcut_x100": 100, "density_x10": 333, "num_types": 2, "seed": 42},
}

def generate_lj_params(num_types):
    """Generate GROMACS-style interleaved LJ parameters: nbfp[type_i * ntype2 + type_j * 2] = c6, +1 = c12"""
    sigmas = np.array([0.3166, 0.04], dtype=np.float32)[:num_types]
    epsilons = np.array([0.6502, 0.0657], dtype=np.float32)[:num_types]

    ntype2 = num_types * 2
    nbfp = np.zeros(num_types * ntype2, dtype=np.float32)
    for i in range(num_types):
        for j in range(num_types):
            sig = (sigmas[i] + sigmas[j]) / 2
            eps = np.sqrt(epsilons[i] * epsilons[j])
            nbfp[i * ntype2 + j * 2]     = 4 * eps * sig**6     # c6
            nbfp[i * ntype2 + j * 2 + 1] = 4 * eps * sig**12    # c12
    return nbfp


def build_clusters_and_pairlist(coords_raw, N, L, rcut, rng):
    """
    Sort atoms into spatial clusters of CLUSTER_SIZE and build cluster pair list.
    Returns: sorted coords, cluster pair list arrays.
    """
    C = CLUSTER_SIZE

    # Pad N to multiple of C
    N_padded = ((N + C - 1) // C) * C
    num_clusters = N_padded // C

    # Sort atoms by spatial cell index for locality
    ncell = max(1, int(L / rcut))
    cell_size = L / ncell
    cell_idx = np.floor(coords_raw / cell_size).astype(np.int32)
    cell_idx = np.clip(cell_idx, 0, ncell - 1)
    # Morton-like ordering: interleave x, y, z bits for spatial locality
    cell_id = cell_idx[:, 0] * ncell * ncell + cell_idx[:, 1] * ncell + cell_idx[:, 2]
    sort_order = np.argsort(cell_id)

    # Build sorted coordinate array (pad with zeros for dummy atoms)
    sorted_coords = np.zeros((N_padded, 3), dtype=np.float32)
    sorted_coords[:N] = coords_raw[sort_order]
    # Dummy atoms placed far away so they don't interact
    sorted_coords[N:] = np.array([1e10, 1e10, 1e10], dtype=np.float32)

    # Map from new index to old index (for charge/type reordering)
    new_to_old = np.full(N_padded, -1, dtype=np.int32)
    new_to_old[:N] = sort_order

    # Build cluster pair list using cell lists on cluster centers
    print(f"  Building cluster pair list ({num_clusters} clusters)...")
    cluster_centers = np.zeros((num_clusters, 3), dtype=np.float32)
    for ci in range(num_clusters):
        start = ci * C
        end = min(start + C, N)
        if end > start:
            cluster_centers[ci] = sorted_coords[start:end].mean(axis=0)
        else:
            cluster_centers[ci] = np.array([1e10, 1e10, 1e10])

    # Cell list for clusters
    # Use slightly larger cutoff for cluster pairs (cluster radius ~ rcut/ncell)
    cluster_rcut = rcut + 2 * cell_size  # conservative: cluster diameter + cutoff
    cluster_rcut2 = cluster_rcut * cluster_rcut

    ncell_cl = max(1, int(L / cluster_rcut))
    cl_cell_size = L / ncell_cl
    cl_cell_idx = np.floor(cluster_centers / cl_cell_size).astype(np.int32)
    cl_cell_idx = np.clip(cl_cell_idx, 0, ncell_cl - 1)
    cl_cell_id = cl_cell_idx[:, 0] * ncell_cl**2 + cl_cell_idx[:, 1] * ncell_cl + cl_cell_idx[:, 2]

    cell_clusters = defaultdict(list)
    for ci in range(num_clusters):
        cell_clusters[cl_cell_id[ci]].append(ci)

    offsets = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]]

    # Build ci → [cj] mapping
    ci_entries = []  # (ci, [cj_indices])
    for ci in range(num_clusters):
        # Skip clusters with only dummy atoms
        if ci * C >= N:
            continue
        cx, cy, cz = cl_cell_idx[ci]
        cj_list = []
        for dx, dy, dz in offsets:
            nx_, ny_, nz_ = cx+dx, cy+dy, cz+dz
            if nx_ < 0 or nx_ >= ncell_cl or ny_ < 0 or ny_ >= ncell_cl or nz_ < 0 or nz_ >= ncell_cl:
                continue
            nid = nx_ * ncell_cl**2 + ny_ * ncell_cl + nz_
            for cj in cell_clusters.get(nid, []):
                if cj < ci:
                    continue  # only store ci <= cj to avoid double counting; but GROMACS stores both
                              # Actually for Newton's 3rd law, we need ci != cj pairs once, self once
                # Check cluster-center distance
                d = cluster_centers[ci] - cluster_centers[cj]
                if np.dot(d, d) < cluster_rcut2:
                    cj_list.append(cj)
        if cj_list:
            ci_entries.append((ci, cj_list))

    # Flatten into arrays
    ci_idx_list = []
    ci_cj_start_list = []
    ci_cj_end_list = []
    cj_idx_list = []
    cj_excl_list = []

    cj_offset = 0
    rcut2 = rcut * rcut

    for ci, cj_list in ci_entries:
        ci_idx_list.append(ci)
        ci_cj_start_list.append(cj_offset)

        for cj in cj_list:
            # Build exclusion mask: bit (i*C+j) = 1 if interaction allowed
            excl = 0
            for i in range(C):
                ai = ci * C + i
                if ai >= N:
                    continue
                for j in range(C):
                    aj = cj * C + j
                    if aj >= N:
                        continue
                    if ci == cj and aj <= ai:
                        continue  # self-cluster: only j > i (avoid double counting + self)
                    excl |= (1 << (i * C + j))

            if excl != 0:  # skip empty cluster pairs
                cj_idx_list.append(cj)
                cj_excl_list.append(excl)

        ci_cj_end_list.append(cj_offset + len(cj_idx_list) - sum(ci_cj_end_list) if not ci_cj_end_list else len(cj_idx_list))
        # Fix: just track current offset
        ci_cj_end_list[-1] = len(cj_idx_list)
        cj_offset = len(cj_idx_list)

    # Recompute start/end properly
    ci_cj_start_arr = []
    ci_cj_end_arr = []
    ptr = 0
    ci_idx_final = []
    cj_idx_final = []
    cj_excl_final = []

    cj_global_idx = 0
    for ci, cj_list in ci_entries:
        start = cj_global_idx
        for cj in cj_list:
            excl = 0
            for i in range(C):
                ai = ci * C + i
                if ai >= N: continue
                for j in range(C):
                    aj = cj * C + j
                    if aj >= N: continue
                    if ci == cj and aj <= ai: continue
                    excl |= (1 << (i * C + j))
            if excl != 0:
                cj_idx_final.append(cj)
                cj_excl_final.append(excl)
                cj_global_idx += 1
        end = cj_global_idx
        if end > start:
            ci_idx_final.append(ci)
            ci_cj_start_arr.append(start)
            ci_cj_end_arr.append(end)

    return (
        sorted_coords.flatten(), new_to_old,
        np.array(ci_idx_final, dtype=np.int32),
        np.array(ci_cj_start_arr, dtype=np.int32),
        np.array(ci_cj_end_arr, dtype=np.int32),
        np.array(cj_idx_final, dtype=np.int32),
        np.array(cj_excl_final, dtype=np.int32),  # stored as int32, interpreted as uint32
        N_padded
    )


def compile_cpu_baseline(orbench_root):
    exe = orbench_root / "tasks" / "nbnxm_forces" / "solution_cpu"
    src = orbench_root / "tasks" / "nbnxm_forces" / "cpu_reference.c"
    task_io = orbench_root / "tasks" / "nbnxm_forces" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    sources = [src, task_io, harness]
    if exe.exists():
        try:
            if all(exe.stat().st_mtime >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass
    cmd = [
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Compile failed:\n{r.stderr}")
    return exe


def run_cpu(exe, data_dir, validate=False, timeout=7200):
    args = [str(exe), str(data_dir)]
    if validate:
        args.append("--validate")
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU run failed:\n{r.stderr}\n{r.stdout}")
    return r.stdout


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = len(sys.argv) == 4 and sys.argv[3] == "--with-expected"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    N = cfg["N"]
    rcut = cfg["rcut_x100"] / 100.0
    density = cfg["density_x10"] / 10.0
    num_types = cfg["num_types"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    print(f"[gen_data] {size_name}: N={N}, rcut={rcut}, density={density}, types={num_types}")

    # Generate atom positions
    volume = N / density
    L = volume ** (1.0 / 3.0)
    coords_raw = rng.uniform(0, L, size=(N, 3)).astype(np.float32)

    # Atom types and charges
    types_raw = np.tile(np.arange(num_types, dtype=np.int32), N // num_types + 1)[:N]
    charge_by_type = np.array([-0.8476, 0.4238], dtype=np.float32)[:num_types]
    charges_raw = charge_by_type[types_raw]

    # Build clusters and pair list
    (x_sorted, new_to_old,
     ci_idx, ci_cj_start, ci_cj_end,
     cj_idx, cj_excl, N_padded) = build_clusters_and_pairlist(coords_raw, N, L, rcut, rng)

    # Reorder charges and types to match sorted coordinates
    q_sorted = np.zeros(N_padded, dtype=np.float32)
    type_sorted = np.zeros(N_padded, dtype=np.int32)
    for new_i in range(N):
        old_i = new_to_old[new_i]
        q_sorted[new_i] = charges_raw[old_i]
        type_sorted[new_i] = types_raw[old_i]

    # LJ parameters
    nbfp = generate_lj_params(num_types)

    num_ci = len(ci_idx)
    num_cj = len(cj_idx)
    print(f"[gen_data] N_padded={N_padded}, clusters={N_padded//CLUSTER_SIZE}, "
          f"ci_entries={num_ci}, cj_entries={num_cj}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("x",           "float32", x_sorted),
            ("q",           "float32", q_sorted),
            ("type",        "int32",   type_sorted),
            ("nbfp",        "float32", nbfp),
            ("ci_idx",      "int32",   ci_idx),
            ("ci_cj_start", "int32",   ci_cj_start),
            ("ci_cj_end",   "int32",   ci_cj_end),
            ("cj_idx",      "int32",   cj_idx),
            ("cj_excl",     "int32",   cj_excl),
        ],
        params={
            "N": N_padded,
            "num_ci": num_ci,
            "num_cj": num_cj,
            "num_types": num_types,
            "rcut_x100": cfg["rcut_x100"],
        },
    )
    sz = (out_dir / "input.bin").stat().st_size
    print(f"[gen_data] Wrote input.bin ({sz / 1e6:.1f} MB)")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        print(f"[gen_data] Running CPU baseline...")
        stdout = run_cpu(exe, out_dir, validate=False, timeout=7200)
        m = re.search(r"TIME_MS:\s*([0-9.]+)", stdout)
        if m:
            time_ms = float(m.group(1))
            with open(out_dir / "cpu_time_ms.txt", "w") as f:
                f.write(f"{time_ms:.3f}\n")
            print(f"[gen_data] CPU time: {time_ms:.1f}ms")
        run_cpu(exe, out_dir, validate=True, timeout=7200)
        if (out_dir / "output.txt").exists():
            shutil.copy2(out_dir / "output.txt", out_dir / "expected_output.txt")
            print(f"[gen_data] Wrote expected_output.txt")
    else:
        print(f"[gen_data] Wrote input.bin only")


if __name__ == "__main__":
    main()
