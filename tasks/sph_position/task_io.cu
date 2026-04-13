// task_io.cu -- sph_position GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int N,
                          const float* posxy_x, const float* posxy_y,
                          const float* posz,
                          const float* movxy_x, const float* movxy_y,
                          const float* movz,
                          float cell_size);
extern void solution_compute(int N,
                             double* out_x, double* out_y, double* out_z,
                             int* out_cell);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N;
    double* out_x;
    double* out_y;
    double* out_z;
    int*    out_cell;
} SPHPosContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");

    const float* pos_x = get_tensor_float(data, "pos_x");
    const float* pos_y = get_tensor_float(data, "pos_y");
    const float* pos_z = get_tensor_float(data, "pos_z");
    const float* mov_x = get_tensor_float(data, "mov_x");
    const float* mov_y = get_tensor_float(data, "mov_y");
    const float* mov_z = get_tensor_float(data, "mov_z");
    float cell_size = (double)get_param(data, "cell_size_x1000000") / 1000000.0f;

    if (!pos_x || !pos_y || !pos_z || !mov_x || !mov_y || !mov_z) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    SPHPosContext* ctx = (SPHPosContext*)calloc(1, sizeof(SPHPosContext));
    ctx->N = N;
    ctx->out_x    = (double*)calloc((size_t)N, sizeof(double));
    ctx->out_y    = (double*)calloc((size_t)N, sizeof(double));
    ctx->out_z    = (double*)calloc((size_t)N, sizeof(double));
    ctx->out_cell = (int*)calloc((size_t)N, sizeof(int));

    solution_init(N, pos_x, pos_y, pos_z, mov_x, mov_y, mov_z, cell_size);
    return ctx;
}

void task_run(void* test_data) {
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    solution_compute(ctx->N, ctx->out_x, ctx->out_y, ctx->out_z, ctx->out_cell);
}

void task_write_output(void* test_data, const char* output_path) {
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.10f %.10f %.10f %d\n",
                ctx->out_x[i], ctx->out_y[i], ctx->out_z[i], ctx->out_cell[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    solution_free();
    free(ctx->out_x);
    free(ctx->out_y);
    free(ctx->out_z);
    free(ctx->out_cell);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
