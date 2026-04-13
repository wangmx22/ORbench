// task_io_cpu.c -- euclidean_distance_matrix CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int ref_nb, int query_nb, int dim,
                          const float* ref, const float* query);
extern void solution_compute(int ref_nb, int query_nb, int dim, float* dist);
extern void solution_free(void);

typedef struct {
    int    ref_nb;
    int    query_nb;
    int    dim;
    float* dist;     // [query_nb * ref_nb]
} EDContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int ref_nb   = (int)get_param(data, "ref_nb");
    int query_nb = (int)get_param(data, "query_nb");
    int dim      = (int)get_param(data, "dim");

    const float* ref   = get_tensor_float(data, "ref");
    const float* query = get_tensor_float(data, "query");
    if (!ref || !query) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    EDContext* ctx = (EDContext*)calloc(1, sizeof(EDContext));
    ctx->ref_nb   = ref_nb;
    ctx->query_nb = query_nb;
    ctx->dim      = dim;
    ctx->dist     = (float*)calloc((size_t)query_nb * ref_nb, sizeof(float));

    solution_init(ref_nb, query_nb, dim, ref, query);
    return ctx;
}

void task_run(void* test_data) {
    EDContext* ctx = (EDContext*)test_data;
    solution_compute(ctx->ref_nb, ctx->query_nb, ctx->dim, ctx->dist);
}

void task_write_output(void* test_data, const char* output_path) {
    EDContext* ctx = (EDContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->query_nb * ctx->ref_nb;
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%.6e\n", ctx->dist[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    EDContext* ctx = (EDContext*)test_data;
    solution_free();
    free(ctx->dist);
    free(ctx);
}
