// task_io.cu — dbscan GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int N, const float* xs, const float* ys,
                          float eps, int minPts);
extern void solution_compute(int N, int* labels);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N;
    int* labels;
} DBSCANContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    DBSCANContext* ctx = (DBSCANContext*)calloc(1, sizeof(DBSCANContext));
    ctx->N = (int)get_param(data, "N");

    int eps_x10000 = (int)get_param(data, "eps_x10000");
    int minPts     = (int)get_param(data, "minPts");
    float eps = (float)eps_x10000 / 10000.0f;

    const float* xs = get_tensor_float(data, "xs");
    const float* ys = get_tensor_float(data, "ys");

    if (!xs || !ys) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    solution_init(ctx->N, xs, ys, eps, minPts);
    ctx->labels = (int*)calloc(ctx->N, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    solution_compute(ctx->N, ctx->labels);
}

void task_write_output(void* test_data, const char* output_path) {
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    FILE* fp = fopen(output_path, "w");
    if (!fp) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(fp, "%d\n", ctx->labels[i]);
    }
    fclose(fp);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    solution_free();
    free(ctx->labels);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
