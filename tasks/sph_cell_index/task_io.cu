// task_io.cu -- sph_cell_index GPU I/O adapter

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
                          float cell_size, int grid_nx, int grid_ny, int grid_nz);
extern void solution_compute(int N, int num_cells,
                             int* sorted_indices, int* cell_begin, int* cell_end);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int  N;
    int  num_cells;
    int* sorted_indices;
    int* cell_begin;
    int* cell_end;
} CellIdxContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");
    int grid_nx = (int)get_param(data, "grid_nx");
    int grid_ny = (int)get_param(data, "grid_ny");
    int grid_nz = (int)get_param(data, "grid_nz");
    int cell_size_raw = (int)get_param(data, "cell_size_x1000000"); float cell_size = (float)cell_size_raw / 1000000.0f;
    int num_cells = grid_nx * grid_ny * grid_nz;

    const float* xs = get_tensor_float(data, "xs");
    const float* ys = get_tensor_float(data, "ys");
    const float* zs = get_tensor_float(data, "zs");

    if (!xs || !ys || !zs) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    CellIdxContext* ctx = (CellIdxContext*)calloc(1, sizeof(CellIdxContext));
    ctx->N = N;
    ctx->num_cells = num_cells;
    ctx->sorted_indices = (int*)calloc((size_t)N, sizeof(int));
    ctx->cell_begin     = (int*)calloc((size_t)num_cells, sizeof(int));
    ctx->cell_end       = (int*)calloc((size_t)num_cells, sizeof(int));

    solution_init(N, xs, ys, zs, cell_size, grid_nx, grid_ny, grid_nz);
    return ctx;
}

void task_run(void* test_data) {
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    solution_compute(ctx->N, ctx->num_cells,
                     ctx->sorted_indices, ctx->cell_begin, ctx->cell_end);
}

void task_write_output(void* test_data, const char* output_path) {
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%d\n", ctx->sorted_indices[i]);
    for (int i = 0; i < ctx->num_cells; i++)
        fprintf(f, "%d %d\n", ctx->cell_begin[i], ctx->cell_end[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    solution_free();
    free(ctx->sorted_indices);
    free(ctx->cell_begin);
    free(ctx->cell_end);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
