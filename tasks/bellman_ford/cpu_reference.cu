// cpu_reference.cu - Bellman-Ford CPU baseline
// Compile: nvcc -O2 -o cpu_reference cpu_reference.cu  (or gcc)
// Run:     ./cpu_reference <size_name> <data_dir>
//          ./cpu_reference              (defaults: medium size, internal graph)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INF_VAL 1e30f

// ============================================================
// Minimal MT19937 for reproducible graph generation
// ============================================================
struct MT19937 {
    unsigned int mt[624];
    int mti;

    void seed(unsigned int s) {
        mt[0] = s;
        for (mti = 1; mti < 624; mti++)
            mt[mti] = 1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti;
    }

    unsigned int next() {
        unsigned int y;
        static const unsigned int mag01[2] = {0, 0x9908B0DFU};
        if (mti >= 624) {
            int kk;
            for (kk = 0; kk < 227; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk+1] & 0x7FFFFFFFU);
                mt[kk] = mt[kk+397] ^ (y >> 1) ^ mag01[y & 1];
            }
            for (; kk < 623; kk++) {
                y = (mt[kk] & 0x80000000U) | (mt[kk+1] & 0x7FFFFFFFU);
                mt[kk] = mt[kk+(397-624)] ^ (y >> 1) ^ mag01[y & 1];
            }
            y = (mt[623] & 0x80000000U) | (mt[0] & 0x7FFFFFFFU);
            mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1];
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9D2C5680U;
        y ^= (y << 15) & 0xEFC60000U;
        y ^= (y >> 18);
        return y;
    }

    int uniform_int(int n) { return (int)(next() % (unsigned int)n); }
};

// ============================================================
// Graph
// ============================================================
struct CSRGraph {
    int num_nodes, num_edges;
    int* row_offsets;
    int* col_indices;
    float* weights;
};

void build_graph(CSRGraph* g, int V, int E, int seed) {
    g->num_nodes = V;
    g->num_edges = E;

    typedef struct { int u, v; float w; } Edge;
    Edge* edges = (Edge*)malloc(E * sizeof(Edge));

    MT19937 rng;
    rng.seed(seed);

    int count = 0;
    while (count < E) {
        int u = rng.uniform_int(V);
        int v = rng.uniform_int(V);
        if (u == v) continue;
        float w = 1.0f + (rng.next() / (float)0xFFFFFFFF) * 99.0f;
        edges[count].u = u;
        edges[count].v = v;
        edges[count].w = w;
        count++;
    }

    g->row_offsets = (int*)calloc(V + 1, sizeof(int));
    g->col_indices = (int*)malloc(E * sizeof(int));
    g->weights = (float*)malloc(E * sizeof(float));

    for (int i = 0; i < E; i++) g->row_offsets[edges[i].u + 1]++;
    for (int i = 1; i <= V; i++) g->row_offsets[i] += g->row_offsets[i - 1];

    int* oc = (int*)malloc((V + 1) * sizeof(int));
    memcpy(oc, g->row_offsets, (V + 1) * sizeof(int));
    for (int i = 0; i < E; i++) {
        int pos = oc[edges[i].u]++;
        g->col_indices[pos] = edges[i].v;
        g->weights[pos] = edges[i].w;
    }

    free(oc);
    free(edges);
}

void free_graph(CSRGraph* g) {
    free(g->row_offsets);
    free(g->col_indices);
    free(g->weights);
}

// ============================================================
// Bellman-Ford CPU
// ============================================================
void bellman_ford_cpu(const CSRGraph* g, int source, float* dist) {
    for (int i = 0; i < g->num_nodes; i++) dist[i] = INF_VAL;
    dist[source] = 0.0f;

    for (int round = 0; round < g->num_nodes - 1; round++) {
        int updated = 0;
        for (int u = 0; u < g->num_nodes; u++) {
            if (dist[u] >= INF_VAL) continue;
            for (int idx = g->row_offsets[u]; idx < g->row_offsets[u + 1]; idx++) {
                int v = g->col_indices[idx];
                float nd = dist[u] + g->weights[idx];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    updated = 1;
                }
            }
        }
        if (!updated) break;
    }
}

// ============================================================
// Timer
// ============================================================
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
// Read binary files
// ============================================================
int* read_int_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int* data = (int*)malloc(count * sizeof(int));
    size_t nread = fread(data, sizeof(int), count, f);
    if ((int)nread != count) {
        fprintf(stderr, "Expected %d ints from %s, got %zu\n", count, path, nread);
        exit(1);
    }
    fclose(f);
    return data;
}

float* read_float_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    size_t nread = fread(data, sizeof(float), count, f);
    if ((int)nread != count) {
        fprintf(stderr, "Expected %d floats from %s, got %zu\n", count, path, nread);
        exit(1);
    }
    fclose(f);
    return data;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        fprintf(stderr, "  e.g.: %s tasks/bellman_ford/data/large\n", argv[0]);
        fprintf(stderr, "  --validate: write result to output.bin for correctness check\n");
        return 1;
    }

    const char* data_dir = argv[1];
    int do_validate = 0;
    if (argc >= 3 && strcmp(argv[2], "--validate") == 0) {
        do_validate = 1;
    }

    char path[512];

    // Read V, E, source from input.txt
    int V, E, source, seed;
    snprintf(path, sizeof(path), "%s/input.txt", data_dir);
    FILE* meta = fopen(path, "r");
    if (!meta) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    fscanf(meta, "%d %d %d %d", &V, &E, &source, &seed);
    fclose(meta);

    fprintf(stderr, "Graph: V=%d, E=%d, source=%d\n", V, E, source);

    // Read CSR arrays from .bin files
    snprintf(path, sizeof(path), "%s/row_offsets.bin", data_dir);
    int* row_offsets = read_int_bin(path, V + 1);

    snprintf(path, sizeof(path), "%s/col_indices.bin", data_dir);
    int* col_indices = read_int_bin(path, E);

    snprintf(path, sizeof(path), "%s/weights.bin", data_dir);
    float* weights = read_float_bin(path, E);

    // Build CSRGraph struct
    CSRGraph g;
    g.num_nodes = V;
    g.num_edges = E;
    g.row_offsets = row_offsets;
    g.col_indices = col_indices;
    g.weights = weights;

    float* dist = (float*)malloc(V * sizeof(float));

    double t0 = get_time_sec();
    bellman_ford_cpu(&g, source, dist);
    double elapsed_ms = (get_time_sec() - t0) * 1000.0;

    // Always print timing
    printf("GPU_TIME_MS: %.3f\n", elapsed_ms);
    fprintf(stderr, "CPU_TIME_MS: %.3f\n", elapsed_ms);

    // Only write results when validating (avoids slow 500K-value printf)
    if (do_validate) {
        snprintf(path, sizeof(path), "%s/output.bin", data_dir);
        FILE* fout = fopen(path, "wb");
        fwrite(dist, sizeof(float), V, fout);
        fclose(fout);
        fprintf(stderr, "Results written to %s/output.bin\n", data_dir);
    }

    free(dist);
    free(row_offsets);
    free(col_indices);
    free(weights);
    return 0;
}
