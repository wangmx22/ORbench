// harness.cu - Framework-provided timing harness
// LLM implements: void gpu_solve(const char* data_dir)
// Framework compiles: nvcc -O2 harness.cu solution.cu -o solution
//
// Two modes:
//   ./solution <data_dir>               → timing only (fast)
//   ./solution <data_dir> --validate    → timing + write output.bin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ============================================================
// Interface that LLM must implement
// ============================================================

// Called once before timing. Load data, allocate GPU memory, etc.
// Return number of result floats (e.g., V for shortest path distances)
extern int gpu_setup(const char* data_dir);

// The actual GPU computation to be timed.
// May be called multiple times (warmup + trials).
// Must be re-entrant: reset internal state if needed.
extern void gpu_run();

// Copy results to output buffer. Called once after timing.
// Buffer is pre-allocated with size from gpu_setup() return value.
extern void gpu_get_results(float* output, int count);

// Cleanup GPU memory.
extern void gpu_cleanup();

// ============================================================
// Harness main
// ============================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        return 1;
    }

    const char* data_dir = argv[1];
    int do_validate = (argc >= 3 && strcmp(argv[2], "--validate") == 0);

    // ---- Setup ----
    int result_count = gpu_setup(data_dir);
    if (result_count <= 0) {
        fprintf(stderr, "gpu_setup failed (returned %d)\n", result_count);
        return 1;
    }

    // ---- Warmup (3 runs, not timed) ----
    for (int w = 0; w < 3; w++) {
        gpu_run();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---- Timed runs ----
    int num_trials = 10;
    float total_ms = 0;
    float min_ms = 1e9, max_ms = 0;

    for (int t = 0; t < num_trials; t++) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gpu_run();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    float mean_ms = total_ms / num_trials;

    // ---- Print timing (framework parses this) ----
    printf("GPU_TIME_MS: %.3f\n", mean_ms);
    fprintf(stderr, "Timing: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
            mean_ms, min_ms, max_ms, num_trials);

    // ---- Validate: write output.bin ----
    if (do_validate) {
        float* results = (float*)malloc(result_count * sizeof(float));
        gpu_run();  // one final run to get results
        CHECK_CUDA(cudaDeviceSynchronize());
        gpu_get_results(results, result_count);

        char path[512];
        snprintf(path, sizeof(path), "%s/output.bin", data_dir);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(results, sizeof(float), result_count, f);
            fclose(f);
            fprintf(stderr, "Results written to %s (%d floats)\n", path, result_count);
        }
        free(results);
    }

    // ---- Cleanup ----
    gpu_cleanup();

    return 0;
}
