// task_io.cu -- hausdorff_distance GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int num_points, int num_spaces,
                          const float* points_xy, const int* space_offsets);
extern void solution_compute(int num_points, int num_spaces, float* results);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int    num_points;
    int    num_spaces;
    float* results;
} HDContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_points = (int)get_param(data, "num_points");
    int num_spaces = (int)get_param(data, "num_spaces");

    const float* points_xy     = get_tensor_float(data, "points_xy");
    const int*   space_offsets = get_tensor_int(data, "space_offsets");
    if (!points_xy || !space_offsets) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    HDContext* ctx = (HDContext*)calloc(1, sizeof(HDContext));
    ctx->num_points = num_points;
    ctx->num_spaces = num_spaces;
    ctx->results    = (float*)calloc((size_t)num_spaces * num_spaces, sizeof(float));

    solution_init(num_points, num_spaces, points_xy, space_offsets);
    return ctx;
}

void task_run(void* test_data) {
    HDContext* ctx = (HDContext*)test_data;
    solution_compute(ctx->num_points, ctx->num_spaces, ctx->results);
}

void task_write_output(void* test_data, const char* output_path) {
    HDContext* ctx = (HDContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->num_spaces * ctx->num_spaces;
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%.6e\n", ctx->results[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    HDContext* ctx = (HDContext*)test_data;
    solution_free();
    free(ctx->results);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
