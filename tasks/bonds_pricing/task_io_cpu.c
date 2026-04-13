// task_io_cpu.c -- bonds_pricing CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int N,
                          const int* issue_year, const int* issue_month, const int* issue_day,
                          const int* maturity_year, const int* maturity_month, const int* maturity_day,
                          const float* rates, float coupon_freq);

extern void solution_compute(int N, float* prices);
extern void solution_free(void);

typedef struct {
    int N;
    float coupon_freq;
    const int* issue_year;
    const int* issue_month;
    const int* issue_day;
    const int* maturity_year;
    const int* maturity_month;
    const int* maturity_day;
    const float* rates;
    float* prices;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");
    int coupon_freq_x100 = (int)get_param(data, "coupon_freq_x100");
    float coupon_freq = (float)coupon_freq_x100 / 100.0f;

    const int* issue_year     = get_tensor_int(data, "issue_year");
    const int* issue_month    = get_tensor_int(data, "issue_month");
    const int* issue_day      = get_tensor_int(data, "issue_day");
    const int* maturity_year  = get_tensor_int(data, "maturity_year");
    const int* maturity_month = get_tensor_int(data, "maturity_month");
    const int* maturity_day   = get_tensor_int(data, "maturity_day");
    const float* rates        = get_tensor_float(data, "rates");

    if (!issue_year || !issue_month || !issue_day ||
        !maturity_year || !maturity_month || !maturity_day || !rates) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->N = N;
    ctx->coupon_freq = coupon_freq;
    ctx->issue_year = issue_year;
    ctx->issue_month = issue_month;
    ctx->issue_day = issue_day;
    ctx->maturity_year = maturity_year;
    ctx->maturity_month = maturity_month;
    ctx->maturity_day = maturity_day;
    ctx->rates = rates;
    ctx->prices = (float*)calloc((size_t)N * 4, sizeof(float));

    solution_init(N, issue_year, issue_month, issue_day,
                  maturity_year, maturity_month, maturity_day,
                  rates, coupon_freq);
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->N, ctx->prices);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f %.6f %.6f %.6f\n",
                ctx->prices[i*4+0], ctx->prices[i*4+1],
                ctx->prices[i*4+2], ctx->prices[i*4+3]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->prices);
    free(ctx);
}
