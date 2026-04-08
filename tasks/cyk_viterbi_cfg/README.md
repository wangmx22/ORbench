# Batched CYK/Viterbi Parsing for CNF Grammars

This task asks for the **best parse score** of a batch of token sequences under a dense weighted context-free grammar in **Chomsky Normal Form (CNF)**. It is a classic interval dynamic-programming problem that appears in formal-language parsing, probabilistic parsing, and structured prediction.

## Problem

For each input sentence of length `L`, compute the maximum-scoring derivation rooted at the start symbol `S = 0` using:

- terminal (unary lexical) scores `unary_scores[A, token]`
- binary production scores `binary_scores[A, B, C]`

The grammar is dense and weighted. Every nonterminal can potentially expand into any pair `(B, C)` and every nonterminal can emit any token. Scores are int32 and are added along the parse tree. The result for one sentence is a single integer:

```text
best_score(sentence) = max score of any parse tree rooted at nonterminal 0
```

## Why it is interesting for GPU acceleration

CYK/Viterbi parsing has a highly structured but nontrivial dependency pattern:

- spans of the same length are independent once all shorter spans are known;
- each chart cell requires a large reduction over split points and child nonterminal pairs;
- the binary rule tensor is reused heavily across chart cells.

This makes it a good benchmark for **phase-synchronous dynamic programming**, **large reductions**, and **cache-aware rule-tensor reuse**.

## Input

- `binary_scores`: int32 array of shape `[N, N, N]` flattened in row-major order, representing scores for `A -> B C`
- `unary_scores`: int32 array of shape `[N, V]` flattened in row-major order, representing scores for `A -> token`
- `tokens`: int32 array of shape `[B, L]` flattened in row-major order

Integer parameters:

- `B`: batch size
- `N`: number of nonterminals
- `V`: vocabulary size
- `L`: sentence length

## Output

One integer per sentence: the best CYK/Viterbi parse score rooted at start symbol 0.

## Source

Classical Cocke–Younger–Kasami parsing and Viterbi-style max-product chart parsing for CNF grammars; standard textbook dynamic programming.
