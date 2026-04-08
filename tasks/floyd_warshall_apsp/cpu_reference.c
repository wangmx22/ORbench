#include <stdlib.h>
#include <string.h>

void solution_compute(
    int B,
    int N,
    int INF,
    const int* adj,
    int* out
) {
    size_t total = (size_t)B * (size_t)N * (size_t)N;
    memcpy(out, adj, total * sizeof(int));

    int graph_stride = N * N;
    for (int g = 0; g < B; ++g) {
        int* dist = out + (size_t)g * (size_t)graph_stride;
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < N; ++i) {
                int dik = dist[i * N + k];
                if (dik >= INF) continue;
                for (int j = 0; j < N; ++j) {
                    int dkj = dist[k * N + j];
                    if (dkj >= INF) continue;
                    int via = dik + dkj;
                    int idx = i * N + j;
                    if (via < dist[idx]) dist[idx] = via;
                }
            }
        }
    }
}

void solution_free(void) {}
