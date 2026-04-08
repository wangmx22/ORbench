#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(
    int H,
    int W,
    int T,
    int alpha_x1e6,
    const float* u0,
    float* out
);

extern void solution_free(void);

typedef struct {
    int H;
    int W;
    int T;
    int alpha_x1e6;
    const float* u0;
    float* out;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int H = (int)get_param(data, "H");
    int W = (int)get_param(data, "W");
    int T = (int)get_param(data, "T");
    int alpha_x1e6 = (int)get_param(data, "alpha_x1e6");

    const float* u0 = get_tensor_float(data, "u0");
    if (!u0) {
        fprintf(stderr, "[task_io] missing tensor u0\n");
        return NULL;
    }

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->H = H;
    ctx->W = W;
    ctx->T = T;
    ctx->alpha_x1e6 = alpha_x1e6;
    ctx->u0 = u0;
    ctx->out = (float*)calloc((size_t)H * (size_t)W, sizeof(float));
    if (!ctx->out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->H, ctx->W, ctx->T, ctx->alpha_x1e6, ctx->u0, ctx->out);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    int N = ctx->H * ctx->W;
    for (int i = 0; i < N; ++i) fprintf(f, "%.8e\n", ctx->out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out);
    free(ctx);
}
