# Batched Viterbi Decoding for Hidden Markov Models

## Background
The Viterbi algorithm finds the single most likely hidden-state sequence for an observed sequence under a Hidden Markov Model (HMM). It is a standard dynamic-programming algorithm used in speech recognition, bioinformatics, sequence labeling, and communication systems.

This task benchmarks **batched Viterbi decoding**: many independent observation sequences are decoded under the same HMM parameters. Each sequence has fixed length `T`, the HMM has `H` hidden states and an observation vocabulary of size `V`.

## Why this is suitable for GPU acceleration
The task exposes two levels of parallelism:

1. **Across sequences**: all sequences in the batch are independent.
2. **Across destination states**: for each time step and each destination state, the algorithm reduces over all predecessor states.

The key bottleneck is the repeated max-reduction over predecessor states for every `(sequence, time, destination_state)` triple. This makes the task more structured than graph algorithms such as Bellman-Ford, but less trivial than purely embarrassingly parallel Monte Carlo simulation.

## Input format
The task stores the following tensors in `input.bin`:

- `log_init` (`float32`, shape `[H]`): log initial-state probabilities.
- `log_trans` (`float32`, shape `[H * H]`, row-major): log transition probabilities from previous state `i` to next state `j`.
- `log_emit` (`float32`, shape `[H * V]`, row-major): log emission probabilities for state `s` and symbol `o`.
- `observations` (`int32`, shape `[B * T]`): observation symbols for each sequence, row-major by sequence.

The scalar parameters are:

- `B`: number of sequences in the batch.
- `T`: sequence length.
- `H`: number of hidden states.
- `V`: vocabulary size.

## Output format
The output is the decoded most-likely hidden-state path for every sequence:

- `out_path` (`int32`, shape `[B * T]`): flattened row-major matrix of decoded states.

The output text file contains one integer per line in row-major order.

## Problem source
The Viterbi algorithm was introduced by Andrew Viterbi in 1967 and is a standard textbook dynamic-programming algorithm for HMM decoding.
