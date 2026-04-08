# CSR Sparse Matrix-Vector Multiply

## Problem background

Sparse matrix-vector multiply (SpMV) is one of the most fundamental kernels in scientific computing, graph analytics, optimization, and iterative linear solvers. Given a sparse matrix `A` and dense vector `x`, compute the dense output vector `y = A x`.

This task stores the sparse matrix in compressed sparse row (CSR) format. For each row `i`, the nonzeros are stored in a contiguous slice `row_ptr[i] : row_ptr[i+1]` of the `col_idx` and `vals` arrays.

## Algorithm source

Classical CSR-format sparse matrix-vector multiplication. Standard material in sparse linear algebra textbooks, HPC libraries, and benchmark suites.

## Why it is suitable for GPU acceleration

- Different rows can be processed in parallel.
- The main bottleneck is memory bandwidth rather than arithmetic throughput.
- GPU implementations can exploit row-level parallelism, warp-level reduction for long rows, and read-only caching for the dense vector `x`.
- SpMV is a canonical benchmark for irregular memory access and sparse linear algebra kernels.

## Input format

Tensors:
- `row_ptr` (`int32`, length `N+1`): CSR row pointer array
- `col_idx` (`int32`, length `nnz`): column indices of each nonzero
- `vals` (`float32`, length `nnz`): nonzero values
- `x` (`float32`, length `N`): dense input vector

Params:
- `N` (`int`): matrix dimension (square `N x N` matrix)

## Output format

- Dense vector `y` (`float32`, length `N`) where
  `y[i] = sum_{k=row_ptr[i]}^{row_ptr[i+1]-1} vals[k] * x[col_idx[k]]`
- One scalar per line in `output.txt` / `expected_output.txt`
