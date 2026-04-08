#include <stddef.h>

void solution_compute(
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* vals,
    const float* x,
    float* y
) {
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; ++k) {
            sum += vals[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

void solution_free(void) {
}
