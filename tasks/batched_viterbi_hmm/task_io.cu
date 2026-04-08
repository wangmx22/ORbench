#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" void solution_compute(
    int B,
    int T,
    int H,
    int V,
    const float* log_init,
    const float* log_trans,
    const float* log_emit,
    const int* observations,
    int* out_path
);

extern "C" void solution_free(void);

typedef struct {
    int B;
    int T;
    int H;
    int V;
    const float* log_init;
    const float* log_trans;
    const float* log_emit;
    const int* observations;
    int* out_path;
} TaskIOContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;

    ctx->B = (int)get_param(data, "B");
    ctx->T = (int)get_param(data, "T");
    ctx->H = (int)get_param(data, "H");
    ctx->V = (int)get_param(data, "V");

    ctx->log_init = get_tensor_float(data, "log_init");
    ctx->log_trans = get_tensor_float(data, "log_trans");
    ctx->log_emit = get_tensor_float(data, "log_emit");
    ctx->observations = get_tensor_int(data, "observations");

    if (!ctx->log_init || !ctx->log_trans || !ctx->log_emit || !ctx->observations) {
        fprintf(stderr, "[task_io] missing required tensor(s)\n");
        free(ctx);
        return NULL;
    }

    ctx->out_path = (int*)calloc((size_t)ctx->B * (size_t)ctx->T, sizeof(int));
    if (!ctx->out_path) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(
        ctx->B,
        ctx->T,
        ctx->H,
        ctx->V,
        ctx->log_init,
        ctx->log_trans,
        ctx->log_emit,
        ctx->observations,
        ctx->out_path
    );
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->B * (size_t)ctx->T;
    for (size_t i = 0; i < total; ++i) fprintf(f, "%d\n", ctx->out_path[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out_path);
    free(ctx);
}
