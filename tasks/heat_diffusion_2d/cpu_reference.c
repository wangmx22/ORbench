#include <stdlib.h>
#include <string.h>

void solution_compute(
    int H,
    int W,
    int T,
    int alpha_x1e6,
    const float* u0,
    float* out
) {
    const int N = H * W;
    const float alpha = (float)alpha_x1e6 / 1000000.0f;

    float* cur = (float*)malloc((size_t)N * sizeof(float));
    float* nxt = (float*)malloc((size_t)N * sizeof(float));
    if (!cur || !nxt) {
        if (cur) free(cur);
        if (nxt) free(nxt);
        return;
    }

    memcpy(cur, u0, (size_t)N * sizeof(float));
    memcpy(nxt, u0, (size_t)N * sizeof(float));

    for (int t = 0; t < T; ++t) {
        // copy fixed boundaries
        for (int j = 0; j < W; ++j) {
            nxt[j] = cur[j];
            nxt[(H - 1) * W + j] = cur[(H - 1) * W + j];
        }
        for (int i = 0; i < H; ++i) {
            nxt[i * W] = cur[i * W];
            nxt[i * W + (W - 1)] = cur[i * W + (W - 1)];
        }

        for (int i = 1; i < H - 1; ++i) {
            int row = i * W;
            for (int j = 1; j < W - 1; ++j) {
                int idx = row + j;
                float c = cur[idx];
                float lap = cur[idx - W] + cur[idx + W] + cur[idx - 1] + cur[idx + 1] - 4.0f * c;
                nxt[idx] = c + alpha * lap;
            }
        }

        float* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    memcpy(out, cur, (size_t)N * sizeof(float));
    free(cur);
    free(nxt);
}

void solution_free(void) {
}
