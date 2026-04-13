// cpu_reference.c -- SPH cell-linked list construction (CPU baseline)
//
// Faithfully ported from DualSPHysics JCellDivGpu_ker.cu
//   KerCalcBeginEndCell + sorting logic
//
// For each particle: compute cell_id = floor(pos/cell_size) linearized.
// Sort particles by cell_id. Build begin/end arrays for each cell.
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.
//
// Reference: Crespo et al. "DualSPHysics: Open-source parallel CFD solver
// based on SPH", Computer Physics Communications, 2015.

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ===== Module-level static state =====
static int    g_N;
static const float* g_xs;
static const float* g_ys;
static const float* g_zs;
static float  g_cell_size;
static int    g_grid_nx;
static int    g_grid_ny;
static int    g_grid_nz;

// ===== Helper: compute linear cell index from position =====
// Matches DualSPHysics cell computation: cellpart[p] = cx + cy*nx + cz*nx*ny

static int ComputeCellIndex(float x, float y, float z,
                            float cell_size, int nx, int ny)
{
    int cx = (int)floorf(x / cell_size);
    int cy = (int)floorf(y / cell_size);
    int cz = (int)floorf(z / cell_size);
    /* Clamp to valid range */
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cz < 0) cz = 0;
    if (cx >= nx) cx = nx - 1;
    if (cy >= ny) cy = ny - 1;
    return cx + cy * nx + cz * nx * ny;
}

// ===== Sort helper: comparison for qsort =====

typedef struct {
    int cell_id;
    int original_index;
} CellParticlePair;

static int compare_cell_particle(const void* a, const void* b)
{
    const CellParticlePair* pa = (const CellParticlePair*)a;
    const CellParticlePair* pb = (const CellParticlePair*)b;
    if (pa->cell_id < pb->cell_id) return -1;
    if (pa->cell_id > pb->cell_id) return 1;
    /* Stable sort tie-break by original index */
    if (pa->original_index < pb->original_index) return -1;
    if (pa->original_index > pb->original_index) return 1;
    return 0;
}

// ===== Ported from DualSPHysics KerCalcBeginEndCell (CPU version) =====
// Original: __global__ void KerCalcBeginEndCell(unsigned n, unsigned pini,
//   const unsigned* cellpart, int2* begcell)
// CPU adaptation: sequential scan over sorted cell_ids to find begin/end.

static void KerCalcBeginEndCell(int N, const int* sorted_cell_ids,
                                int num_cells, int* cell_begin, int* cell_end)
{
    int i;
    /* Initialize all cells as empty */
    for (i = 0; i < num_cells; i++) {
        cell_begin[i] = -1;
        cell_end[i]   = -1;
    }

    if (N == 0) return;

    /* First particle starts a cell */
    cell_begin[sorted_cell_ids[0]] = 0;

    for (i = 1; i < N; i++) {
        if (sorted_cell_ids[i] != sorted_cell_ids[i - 1]) {
            /* Previous cell ends here */
            cell_end[sorted_cell_ids[i - 1]] = i;
            /* New cell begins here */
            cell_begin[sorted_cell_ids[i]] = i;
        }
    }

    /* Last particle's cell ends at N */
    cell_end[sorted_cell_ids[N - 1]] = N;
}

// ===== Public interface =====

void solution_init(int N,
                   const float* xs, const float* ys, const float* zs,
                   float cell_size, int grid_nx, int grid_ny, int grid_nz)
{
    g_N = N;
    g_xs = xs;
    g_ys = ys;
    g_zs = zs;
    g_cell_size = cell_size;
    g_grid_nx = grid_nx;
    g_grid_ny = grid_ny;
    g_grid_nz = grid_nz;
}

void solution_compute(int N, int num_cells,
                      int* sorted_indices, int* cell_begin, int* cell_end)
{
    int i;

    /* Step 1: Compute cell index for each particle */
    CellParticlePair* pairs = (CellParticlePair*)malloc(N * sizeof(CellParticlePair));
    for (i = 0; i < N; i++) {
        pairs[i].cell_id = ComputeCellIndex(g_xs[i], g_ys[i], g_zs[i],
                                            g_cell_size, g_grid_nx, g_grid_ny);
        pairs[i].original_index = i;
    }

    /* Step 2: Sort by cell_id (matches DualSPHysics SortDataParticles) */
    qsort(pairs, N, sizeof(CellParticlePair), compare_cell_particle);

    /* Step 3: Extract sorted indices and sorted cell_ids */
    int* sorted_cell_ids = (int*)malloc(N * sizeof(int));
    for (i = 0; i < N; i++) {
        sorted_indices[i] = pairs[i].original_index;
        sorted_cell_ids[i] = pairs[i].cell_id;
    }

    /* Step 4: Build begin/end cell arrays (KerCalcBeginEndCell) */
    KerCalcBeginEndCell(N, sorted_cell_ids, num_cells, cell_begin, cell_end);

    free(sorted_cell_ids);
    free(pairs);
}

void solution_free(void)
{
    /* All data owned by task_io; nothing to free here. */
}
