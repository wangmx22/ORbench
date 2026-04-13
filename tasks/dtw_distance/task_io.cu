// task_io.cu -- dtw_distance GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int num_entries, int num_features,
                          const float* subjects, const float* query);
extern void solution_compute(int num_entries, int num_features, float* distances);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int    num_entries;
    int    num_features;
    float* distances;
} DTWContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_entries  = (int)get_param(data, "num_entries");
    int num_features = (int)get_param(data, "num_features");

    const float* subjects = get_tensor_float(data, "subjects");
    const float* query    = get_tensor_float(data, "query");
    if (!subjects || !query) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    DTWContext* ctx = (DTWContext*)calloc(1, sizeof(DTWContext));
    ctx->num_entries  = num_entries;
    ctx->num_features = num_features;
    ctx->distances    = (float*)calloc((size_t)num_entries, sizeof(float));

    solution_init(num_entries, num_features, subjects, query);
    return ctx;
}

void task_run(void* test_data) {
    DTWContext* ctx = (DTWContext*)test_data;
    solution_compute(ctx->num_entries, ctx->num_features, ctx->distances);
}

void task_write_output(void* test_data, const char* output_path) {
    DTWContext* ctx = (DTWContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_entries; i++)
        fprintf(f, "%.6e\n", ctx->distances[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    DTWContext* ctx = (DTWContext*)test_data;
    solution_free();
    free(ctx->distances);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
