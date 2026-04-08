# Held-Karp Dynamic Programming for TSP Tour Cost

## Problem background
The Traveling Salesman Problem (TSP) asks for the minimum-cost Hamiltonian tour that starts at a depot, visits every city exactly once, and returns to the start. This task uses the classic exact Held-Karp dynamic programming algorithm to compute the optimal tour cost for batches of small complete weighted graphs.

## Algorithm source
Held-Karp subset DP is a foundational exact algorithm for TSP and combinatorial optimization.

- Held, M. and Karp, R. M. (1962), *A Dynamic Programming Approach to Sequencing Problems*.
- Standard textbook exact TSP algorithm.

## Why it fits GPU acceleration
The full dependency graph is not fully parallel, but states at the same subset cardinality are independent once the previous cardinality layer is computed. This creates a frontier-style parallel pattern:

- parallel across independent graph instances in the batch
- parallel across DP states `(mask, j)` within the same subset-size layer
- reduction over predecessor cities for each state

The main bottleneck is the exponential DP state space and irregular subset indexing.

## Input format
The task is encoded as a batch of complete weighted graphs with a fixed number of cities `n` per batch.

- `costs` (int32): flattened row-major cost matrices of shape `[B, n, n]`
- params:
  - `B`: batch size
  - `n`: number of cities

City `0` is always the fixed start/end city.

## Output format
- `tour_costs_out[b]` (int32): exact optimal TSP tour cost for graph `b`

One integer is written per line in `expected_output.txt` / `output.txt`.
