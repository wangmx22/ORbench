// task_io_cpu.c -- black_scholes CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int N,
                          const int* types, const float* strikes, const float* spots,
                          const float* qs, const float* rs, const float* ts,
                          const float* vols);
extern void solution_compute(int N, float* prices);
extern void solution_free(void);

typedef struct {
    int N;
    float* prices;
} BSContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");

    const int*   types   = get_tensor_int(data, "types");
    const float* strikes = get_tensor_float(data, "strikes");
    const float* spots   = get_tensor_float(data, "spots");
    const float* qs      = get_tensor_float(data, "qs");
    const float* rs      = get_tensor_float(data, "rs");
    const float* ts      = get_tensor_float(data, "ts");
    const float* vols    = get_tensor_float(data, "vols");

    if (!types || !strikes || !spots || !qs || !rs || !ts || !vols) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    BSContext* ctx = (BSContext*)calloc(1, sizeof(BSContext));
    ctx->N = N;
    ctx->prices = (float*)calloc((size_t)N, sizeof(float));

    solution_init(N, types, strikes, spots, qs, rs, ts, vols);
    return ctx;
}

void task_run(void* test_data) {
    BSContext* ctx = (BSContext*)test_data;
    solution_compute(ctx->N, ctx->prices);
}

void task_write_output(void* test_data, const char* output_path) {
    BSContext* ctx = (BSContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f\n", ctx->prices[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    BSContext* ctx = (BSContext*)test_data;
    solution_free();
    free(ctx->prices);
    free(ctx);
}
