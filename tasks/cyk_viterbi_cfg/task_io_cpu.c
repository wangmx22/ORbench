#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(int B, int N, int V, int L,
                             const int* binary_scores,
                             const int* unary_scores,
                             const int* tokens,
                             int* parse_scores_out);
extern void solution_free(void);

typedef struct {
    int B, N, V, L;
    const int* binary_scores;
    const int* unary_scores;
    const int* tokens;
    int* parse_scores_out;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->B = (int)get_param(data, "B");
    ctx->N = (int)get_param(data, "N");
    ctx->V = (int)get_param(data, "V");
    ctx->L = (int)get_param(data, "L");
    ctx->binary_scores = get_tensor_int(data, "binary_scores");
    ctx->unary_scores = get_tensor_int(data, "unary_scores");
    ctx->tokens = get_tensor_int(data, "tokens");
    if (!ctx->binary_scores || !ctx->unary_scores || !ctx->tokens) {
        fprintf(stderr, "[task_io] Missing one or more required tensors\n");
        free(ctx);
        return NULL;
    }
    ctx->parse_scores_out = (int*)calloc((size_t)ctx->B, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->B, ctx->N, ctx->V, ctx->L,
                     ctx->binary_scores, ctx->unary_scores, ctx->tokens,
                     ctx->parse_scores_out);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int b = 0; b < ctx->B; ++b) fprintf(f, "%d\n", ctx->parse_scores_out[b]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->parse_scores_out);
    free(ctx);
}
