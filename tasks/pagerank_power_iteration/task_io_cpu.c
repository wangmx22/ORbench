#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(
    int N,
    int M,
    int iters,
    int damping_x1e6,
    const int* row_ptr_in,
    const int* col_ind_in,
    const float* inv_out_deg,
    float* out
);

extern void solution_free(void);

typedef struct {
    int N;
    int M;
    int iters;
    int damping_x1e6;
    const int* row_ptr_in;
    const int* col_ind_in;
    const float* inv_out_deg;
    float* out;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;

    ctx->N = (int)get_param(data, "N");
    ctx->M = (int)get_param(data, "M");
    ctx->iters = (int)get_param(data, "iters");
    ctx->damping_x1e6 = (int)get_param(data, "damping_x1e6");

    ctx->row_ptr_in = get_tensor_int(data, "row_ptr_in");
    ctx->col_ind_in = get_tensor_int(data, "col_ind_in");
    ctx->inv_out_deg = get_tensor_float(data, "inv_out_deg");

    if (!ctx->row_ptr_in || !ctx->col_ind_in || !ctx->inv_out_deg) {
        fprintf(stderr, "[task_io] missing required tensor(s)\n");
        free(ctx);
        return NULL;
    }

    ctx->out = (float*)calloc((size_t)ctx->N, sizeof(float));
    if (!ctx->out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(
        ctx->N,
        ctx->M,
        ctx->iters,
        ctx->damping_x1e6,
        ctx->row_ptr_in,
        ctx->col_ind_in,
        ctx->inv_out_deg,
        ctx->out
    );
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; ++i) fprintf(f, "%.8e\n", ctx->out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out);
    free(ctx);
}
