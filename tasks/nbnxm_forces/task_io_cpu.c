// task_io_cpu.c — nbnxm_forces CPU I/O adapter (compute_only, cluster format)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CLUSTER_SIZE 4

extern void solution_compute(
    int N, int num_ci, int num_cj, int num_types, float rcut2,
    const float* x, const float* q, const int* type, const float* nbfp,
    const int* ci_idx, const int* ci_cj_start, const int* ci_cj_end,
    const int* cj_idx, const unsigned int* cj_excl,
    float* f, float* energy_out
);
extern void solution_free(void);

typedef struct {
    int N, num_ci, num_cj, num_types;
    float rcut2;
    const float* x;
    const float* q;
    const int*   type;
    const float* nbfp;
    const int*   ci_idx;
    const int*   ci_cj_start;
    const int*   ci_cj_end;
    const int*   cj_idx;
    const int*   cj_excl;  // stored as int, cast to unsigned
    float* f;
    float  energy[2];
} NbContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    NbContext* ctx = (NbContext*)calloc(1, sizeof(NbContext));
    ctx->N         = (int)get_param(data, "N");
    ctx->num_ci    = (int)get_param(data, "num_ci");
    ctx->num_cj    = (int)get_param(data, "num_cj");
    ctx->num_types = (int)get_param(data, "num_types");
    int rcut_x100  = (int)get_param(data, "rcut_x100");
    float rcut     = (float)rcut_x100 / 100.0f;
    ctx->rcut2     = rcut * rcut;

    ctx->x           = get_tensor_float(data, "x");
    ctx->q           = get_tensor_float(data, "q");
    ctx->type        = get_tensor_int(data, "type");
    ctx->nbfp        = get_tensor_float(data, "nbfp");
    ctx->ci_idx      = get_tensor_int(data, "ci_idx");
    ctx->ci_cj_start = get_tensor_int(data, "ci_cj_start");
    ctx->ci_cj_end   = get_tensor_int(data, "ci_cj_end");
    ctx->cj_idx      = get_tensor_int(data, "cj_idx");
    ctx->cj_excl     = get_tensor_int(data, "cj_excl");

    ctx->f = (float*)calloc(ctx->N * 3, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    NbContext* ctx = (NbContext*)test_data;
    solution_compute(
        ctx->N, ctx->num_ci, ctx->num_cj, ctx->num_types, ctx->rcut2,
        ctx->x, ctx->q, ctx->type, ctx->nbfp,
        ctx->ci_idx, ctx->ci_cj_start, ctx->ci_cj_end,
        ctx->cj_idx, (const unsigned int*)ctx->cj_excl,
        ctx->f, ctx->energy
    );
}

void task_write_output(void* test_data, const char* output_path) {
    NbContext* ctx = (NbContext*)test_data;
    FILE* fp = fopen(output_path, "w");
    if (!fp) return;
    // Line 1-2: energies
    fprintf(fp, "%.6e\n", ctx->energy[0]);
    fprintf(fp, "%.6e\n", ctx->energy[1]);
    // Line 3+: force magnitude per atom
    for (int i = 0; i < ctx->N; i++) {
        float fx = ctx->f[i*3+0];
        float fy = ctx->f[i*3+1];
        float fz = ctx->f[i*3+2];
        fprintf(fp, "%.6e\n", sqrtf(fx*fx + fy*fy + fz*fz));
    }
    fclose(fp);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    NbContext* ctx = (NbContext*)test_data;
    solution_free();
    free(ctx->f);
    free(ctx);
}
