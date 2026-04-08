#include <stdlib.h>
#include <stdint.h>

#define NEG_INF (-1000000000)

static inline int idx3(int a, int b, int c, int N) {
    return ((a * N) + b) * N + c;
}

static inline size_t chart_index(int i, int j, int A, int L, int N) {
    return (((size_t)i * (size_t)(L + 1) + (size_t)j) * (size_t)N + (size_t)A);
}

static int solve_one_sentence(int N, int V, int L,
                              const int* binary_scores,
                              const int* unary_scores,
                              const int* sentence) {
    (void)V;
    size_t chart_elems = (size_t)(L + 1) * (size_t)(L + 1) * (size_t)N;
    int* dp = (int*)malloc(chart_elems * sizeof(int));
    if (!dp) return NEG_INF;
    for (size_t t = 0; t < chart_elems; ++t) dp[t] = NEG_INF;

    for (int i = 0; i < L; ++i) {
        int tok = sentence[i];
        for (int A = 0; A < N; ++A) {
            dp[chart_index(i, i + 1, A, L, N)] = unary_scores[A * V + tok];
        }
    }

    for (int len = 2; len <= L; ++len) {
        for (int i = 0; i + len <= L; ++i) {
            int j = i + len;
            for (int A = 0; A < N; ++A) {
                int best = NEG_INF;
                for (int k = i + 1; k < j; ++k) {
                    const int* rulesA = binary_scores + (size_t)A * (size_t)N * (size_t)N;
                    for (int B = 0; B < N; ++B) {
                        int left = dp[chart_index(i, k, B, L, N)];
                        if (left <= NEG_INF / 2) continue;
                        const int* rowBC = rulesA + (size_t)B * (size_t)N;
                        for (int C = 0; C < N; ++C) {
                            int right = dp[chart_index(k, j, C, L, N)];
                            if (right <= NEG_INF / 2) continue;
                            int cand = rowBC[C] + left + right;
                            if (cand > best) best = cand;
                        }
                    }
                }
                dp[chart_index(i, j, A, L, N)] = best;
            }
        }
    }

    int ans = dp[chart_index(0, L, 0, L, N)];
    free(dp);
    return ans;
}

void solution_compute(int B, int N, int V, int L,
                      const int* binary_scores,
                      const int* unary_scores,
                      const int* tokens,
                      int* parse_scores_out) {
    for (int b = 0; b < B; ++b) {
        const int* sentence = tokens + (size_t)b * (size_t)L;
        parse_scores_out[b] = solve_one_sentence(N, V, L, binary_scores, unary_scores, sentence);
    }
}

void solution_free(void) {}
