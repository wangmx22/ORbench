#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(int B, int N, int INF, const int* adj, int* out);
extern void solution_free(void);

typedef struct {
    int B;
    int N;
    int INF;
    const int* adj;
    int* out;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int B = (int)get_param(data, "B");
    int N = (int)get_param(data, "N");
    int INF = (int)get_param(data, "INF");
    const int* adj = get_tensor_int(data, "adj");
    if (!adj) {
        fprintf(stderr, "[task_io] missing tensor adj\n");
        return NULL;
    }

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->B = B;
    ctx->N = N;
    ctx->INF = INF;
    ctx->adj = adj;
    ctx->out = (int*)calloc((size_t)B * (size_t)N * (size_t)N, sizeof(int));
    if (!ctx->out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->B, ctx->N, ctx->INF, ctx->adj, ctx->out);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->B * (size_t)ctx->N * (size_t)ctx->N;
    for (size_t i = 0; i < total; ++i) fprintf(f, "%d\n", ctx->out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out);
    free(ctx);
}
