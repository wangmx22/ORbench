#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" void solution_compute(
    int N,
    const int* row_ptr,
    const int* col_idx,
    const float* vals,
    const float* x,
    float* y
);

extern "C" void solution_free(void);

typedef struct {
    int N;
    const int* row_ptr;
    const int* col_idx;
    const float* vals;
    const float* x;
    float* y;
} TaskIOContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->N = (int)get_param(data, "N");
    ctx->row_ptr = get_tensor_int(data, "row_ptr");
    ctx->col_idx = get_tensor_int(data, "col_idx");
    ctx->vals = get_tensor_float(data, "vals");
    ctx->x = get_tensor_float(data, "x");
    if (!ctx->row_ptr || !ctx->col_idx || !ctx->vals || !ctx->x) {
        fprintf(stderr, "[task_io] missing required tensors for spmv_csr\n");
        free(ctx);
        return NULL;
    }
    ctx->y = (float*)calloc((size_t)ctx->N, sizeof(float));
    if (!ctx->y) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->N, ctx->row_ptr, ctx->col_idx, ctx->vals, ctx->x, ctx->y);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; ++i) fprintf(f, "%.8e\n", ctx->y[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->y);
    free(ctx);
}
