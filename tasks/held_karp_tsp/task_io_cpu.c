#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(int B, int n, const int* costs, int* tour_costs_out);
extern void solution_free(void);

typedef struct {
    int B;
    int n;
    const int* costs;
    int* tour_costs_out;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int B = (int)get_param(data, "B");
    int n = (int)get_param(data, "n");
    const int* costs = get_tensor_int(data, "costs");
    if (!costs) {
        fprintf(stderr, "[task_io] Missing costs tensor\n");
        return NULL;
    }
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->B = B;
    ctx->n = n;
    ctx->costs = costs;
    ctx->tour_costs_out = (int*)calloc((size_t)B, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->B, ctx->n, ctx->costs, ctx->tour_costs_out);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int b = 0; b < ctx->B; ++b) fprintf(f, "%d\n", ctx->tour_costs_out[b]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->tour_costs_out);
    free(ctx);
}
