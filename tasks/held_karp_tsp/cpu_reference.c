#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INF_COST 1000000000

static int solve_one_instance(int n, const int* cost) {
    int m = n - 1;
    int subset_count = 1 << m;
    int *dp = (int*)malloc((size_t)subset_count * (size_t)n * sizeof(int));
    if (!dp) return -1;

    for (int i = 0; i < subset_count * n; ++i) dp[i] = INF_COST;

    for (int j = 1; j < n; ++j) {
        int mask = 1 << (j - 1);
        dp[mask * n + j] = cost[j];  // cost[0*n + j]
    }

    for (int mask = 1; mask < subset_count; ++mask) {
        for (int j = 1; j < n; ++j) {
            int bitj = 1 << (j - 1);
            if ((mask & bitj) == 0) continue;
            int prev_mask = mask ^ bitj;
            if (prev_mask == 0) continue;

            int best = INF_COST;
            int pm = prev_mask;
            while (pm) {
                int lowbit = pm & -pm;
                int k = __builtin_ctz((unsigned)lowbit) + 1;
                int cand = dp[prev_mask * n + k] + cost[k * n + j];
                if (cand < best) best = cand;
                pm ^= lowbit;
            }
            dp[mask * n + j] = best;
        }
    }

    int full_mask = subset_count - 1;
    int answer = INF_COST;
    for (int j = 1; j < n; ++j) {
        int cand = dp[full_mask * n + j] + cost[j * n + 0];
        if (cand < answer) answer = cand;
    }

    free(dp);
    return answer;
}

void solution_compute(int B, int n, const int* costs, int* tour_costs_out) {
    for (int b = 0; b < B; ++b) {
        const int* cost = costs + (size_t)b * (size_t)n * (size_t)n;
        tour_costs_out[b] = solve_one_instance(n, cost);
    }
}

void solution_free(void) {}
