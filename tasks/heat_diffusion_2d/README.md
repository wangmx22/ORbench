# 2D Heat Diffusion (Jacobi Stencil)

## Problem background

This task simulates transient heat diffusion on a 2D rectangular grid using an explicit finite-difference update. Each interior cell exchanges heat with its four von Neumann neighbors (up, down, left, right), while boundary cells remain fixed throughout the simulation. This is one of the most classical stencil-computation kernels in scientific computing, PDE solvers, and GPU programming.

Given an initial temperature field `u0` on an `H x W` grid, apply `T` Jacobi iterations:

```text
u_new[i,j] = u[i,j] + alpha * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])
```

for all interior cells `1 <= i < H-1`, `1 <= j < W-1`. Boundary cells are copied unchanged at every step.

## Algorithm source

Classical 5-point stencil / Jacobi relaxation for the 2D heat equation. Standard material in numerical PDE textbooks, finite-difference methods, and GPU stencil benchmarks.

## Why it is suitable for GPU acceleration

- Each interior cell update in a time step is independent once the previous grid is fixed.
- The main bottleneck is memory bandwidth and repeated nearest-neighbor accesses.
- GPU kernels can exploit 2D tiling, shared-memory halos, and coalesced row-major loads.
- The computation is highly regular, making it a good benchmark for stencil optimization.

## Input format

Tensors:
- `u0` (`float32`, length `H*W`): initial temperature field in row-major order.

Params:
- `H` (`int`): grid height
- `W` (`int`): grid width
- `T` (`int`): number of Jacobi iterations
- `alpha_x1e6` (`int`): diffusion coefficient scaled by `1e6` (decode as `alpha = alpha_x1e6 / 1e6`)

## Output format

- Final temperature grid after `T` iterations, written as `H*W` float values in row-major order.
- One scalar per line in `output.txt` / `expected_output.txt`.
