// task_io_cpu.c -- repo_pricing CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int N,
                          const int* settle_year, const int* settle_month, const int* settle_day,
                          const int* delivery_year, const int* delivery_month, const int* delivery_day,
                          const int* issue_year, const int* issue_month, const int* issue_day,
                          const int* maturity_year, const int* maturity_month, const int* maturity_day,
                          const float* bond_rates, const float* repo_rates,
                          const float* bond_clean_prices, const float* dummy_strikes);
extern void solution_compute(int N, float* prices);
extern void solution_free(void);

typedef struct {
    int N;
    float* prices;
} RepoContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    int N = (int)get_param(data, "N");

    const int* settle_year     = get_tensor_int(data, "settle_year");
    const int* settle_month    = get_tensor_int(data, "settle_month");
    const int* settle_day      = get_tensor_int(data, "settle_day");
    const int* delivery_year   = get_tensor_int(data, "delivery_year");
    const int* delivery_month  = get_tensor_int(data, "delivery_month");
    const int* delivery_day    = get_tensor_int(data, "delivery_day");
    const int* issue_year      = get_tensor_int(data, "issue_year");
    const int* issue_month     = get_tensor_int(data, "issue_month");
    const int* issue_day       = get_tensor_int(data, "issue_day");
    const int* maturity_year   = get_tensor_int(data, "maturity_year");
    const int* maturity_month  = get_tensor_int(data, "maturity_month");
    const int* maturity_day    = get_tensor_int(data, "maturity_day");
    const float* bond_rates       = get_tensor_float(data, "bond_rates");
    const float* repo_rates       = get_tensor_float(data, "repo_rates");
    const float* bond_clean_prices = get_tensor_float(data, "bond_clean_prices");
    const float* dummy_strikes    = get_tensor_float(data, "dummy_strikes");

    if (!settle_year || !settle_month || !settle_day ||
        !delivery_year || !delivery_month || !delivery_day ||
        !issue_year || !issue_month || !issue_day ||
        !maturity_year || !maturity_month || !maturity_day ||
        !bond_rates || !repo_rates || !bond_clean_prices || !dummy_strikes) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    RepoContext* ctx = (RepoContext*)calloc(1, sizeof(RepoContext));
    ctx->N = N;
    ctx->prices = (float*)calloc((size_t)N * 12, sizeof(float));

    solution_init(N, settle_year, settle_month, settle_day,
                  delivery_year, delivery_month, delivery_day,
                  issue_year, issue_month, issue_day,
                  maturity_year, maturity_month, maturity_day,
                  bond_rates, repo_rates, bond_clean_prices, dummy_strikes);
    return ctx;
}

void task_run(void* test_data) {
    RepoContext* ctx = (RepoContext*)test_data;
    solution_compute(ctx->N, ctx->prices);
}

void task_write_output(void* test_data, const char* output_path) {
    RepoContext* ctx = (RepoContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                ctx->prices[i*12+0], ctx->prices[i*12+1],
                ctx->prices[i*12+2], ctx->prices[i*12+3],
                ctx->prices[i*12+4], ctx->prices[i*12+5],
                ctx->prices[i*12+6], ctx->prices[i*12+7],
                ctx->prices[i*12+8], ctx->prices[i*12+9],
                ctx->prices[i*12+10], ctx->prices[i*12+11]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    RepoContext* ctx = (RepoContext*)test_data;
    solution_free();
    free(ctx->prices);
    free(ctx);
}
