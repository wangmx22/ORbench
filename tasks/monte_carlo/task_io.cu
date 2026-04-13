// task_io.cu -- monte_carlo GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int N, int num_steps, float risk_free, float volatility,
                          float strike, float spot, float time_to_maturity,
                          unsigned int base_seed);
extern void solution_compute(int N, float* payoffs);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N;
    float* payoffs;
} MCContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int N                  = (int)get_param(data, "N");
    int num_steps          = (int)get_param(data, "num_steps");
    int risk_free_x10000   = (int)get_param(data, "risk_free_x10000");
    int volatility_x10000  = (int)get_param(data, "volatility_x10000");
    int strike_x100        = (int)get_param(data, "strike_x100");
    int spot_x100          = (int)get_param(data, "spot_x100");
    int time_x1000         = (int)get_param(data, "time_x1000");
    unsigned int base_seed = (unsigned int)get_param(data, "base_seed");

    float risk_free  = (float)risk_free_x10000 / 10000.0f;
    float volatility = (float)volatility_x10000 / 10000.0f;
    float strike     = (float)strike_x100 / 100.0f;
    float spot       = (float)spot_x100 / 100.0f;
    float time_to_maturity = (float)time_x1000 / 1000.0f;

    MCContext* ctx = (MCContext*)calloc(1, sizeof(MCContext));
    ctx->N = N;
    ctx->payoffs = (float*)calloc((size_t)N, sizeof(float));

    solution_init(N, num_steps, risk_free, volatility, strike, spot,
                  time_to_maturity, base_seed);
    return ctx;
}

void task_run(void* test_data) {
    MCContext* ctx = (MCContext*)test_data;
    solution_compute(ctx->N, ctx->payoffs);
}

void task_write_output(void* test_data, const char* output_path) {
    MCContext* ctx = (MCContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f\n", ctx->payoffs[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    MCContext* ctx = (MCContext*)test_data;
    solution_free();
    free(ctx->payoffs);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
