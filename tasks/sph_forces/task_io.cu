// task_io.cu -- sph_forces GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int N,
                          const float* xs, const float* ys, const float* zs,
                          const float* vxs, const float* vys, const float* vzs,
                          const float* rhos, const float* masses,
                          float h, float cs0, float rhop0, float alpha_visc,
                          const int* cell_begin, const int* cell_end,
                          const int* sorted_idx,
                          int grid_nx, int grid_ny, int grid_nz,
                          float cell_size);
extern void solution_compute(int N,
                             float* ax, float* ay, float* az, float* drhodt);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int    N;
    float* ax;
    float* ay;
    float* az;
    float* drhodt;
} SPHForceContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");

    const float* xs     = get_tensor_float(data, "xs");
    const float* ys     = get_tensor_float(data, "ys");
    const float* zs     = get_tensor_float(data, "zs");
    const float* vxs    = get_tensor_float(data, "vxs");
    const float* vys    = get_tensor_float(data, "vys");
    const float* vzs    = get_tensor_float(data, "vzs");
    const float* rhos   = get_tensor_float(data, "rhos");
    const float* masses = get_tensor_float(data, "masses");
    const int* cell_begin  = get_tensor_int(data, "cell_begin");
    const int* cell_end    = get_tensor_int(data, "cell_end");
    const int* sorted_idx  = get_tensor_int(data, "sorted_idx");

    float h          = (float)get_param(data, "h_x1000000") / 1000000.0f;
    float cs0        = (float)get_param(data, "cs0_x10000") / 10000.0f;
    float rhop0      = (float)get_param(data, "rhop0_x100") / 100.0f;
    float alpha_visc = (float)get_param(data, "alpha_visc_x10000") / 10000.0f;
    int grid_nx      = (int)get_param(data, "grid_nx");
    int grid_ny      = (int)get_param(data, "grid_ny");
    int grid_nz      = (int)get_param(data, "grid_nz");
    float cell_size  = (float)get_param(data, "cell_size_x1000000") / 1000000.0f;

    if (!xs || !ys || !zs || !vxs || !vys || !vzs || !rhos || !masses ||
        !cell_begin || !cell_end || !sorted_idx) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    SPHForceContext* ctx = (SPHForceContext*)calloc(1, sizeof(SPHForceContext));
    ctx->N = N;
    ctx->ax    = (float*)calloc((size_t)N, sizeof(float));
    ctx->ay    = (float*)calloc((size_t)N, sizeof(float));
    ctx->az    = (float*)calloc((size_t)N, sizeof(float));
    ctx->drhodt = (float*)calloc((size_t)N, sizeof(float));

    solution_init(N, xs, ys, zs, vxs, vys, vzs, rhos, masses,
                  h, cs0, rhop0, alpha_visc,
                  cell_begin, cell_end, sorted_idx,
                  grid_nx, grid_ny, grid_nz, cell_size);
    return ctx;
}

void task_run(void* test_data) {
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    solution_compute(ctx->N, ctx->ax, ctx->ay, ctx->az, ctx->drhodt);
}

void task_write_output(void* test_data, const char* output_path) {
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6e %.6e %.6e %.6e\n",
                ctx->ax[i], ctx->ay[i], ctx->az[i], ctx->drhodt[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    solution_free();
    free(ctx->ax);
    free(ctx->ay);
    free(ctx->az);
    free(ctx->drhodt);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
