// cpu_reference.c — DBSCAN density-based spatial clustering (CPU baseline)
//
// Faithfully ported from fast-cuda-gpu-dbscan/multicore-cpu/dbscan.cpp
// Preserves original variable names, struct definitions, and algorithm flow.
//
// Key simplification: single-threaded (no OpenMP), brute-force neighbor search.
// This matches the serial DBSCAN algorithm complexity O(N^2).
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ===== Constants from original (common.h / dbscan.cpp) =====
#define UNCLASSIFIED -1
#define NOISE        -2
#define MARKED        0

// ===== Structs from original (multicore-cpu/dbscan.cpp) =====
typedef float real;

typedef struct {
    real x;
    real y;
} point;

// ===== Module-level state =====
static int    g_N;
static real   g_eps;       // epsilon radius
static int    g_minPts;    // minimum points for core
static point* g_points;    // point array (owned, copied from input)

void solution_init(int N, const float* xs, const float* ys,
                   float eps, int minPts)
{
    g_N = N;
    g_eps = eps;
    g_minPts = minPts;

    g_points = (point*)malloc(N * sizeof(point));
    for (int i = 0; i < N; i++) {
        g_points[i].x = xs[i];
        g_points[i].y = ys[i];
    }
}

// ===== Neighbor search — matches processPointCPU from original =====
// Brute-force: check all N points for distance <= eps
static int calcRow(int thisPoint, int numPoints, real radius,
                   char* oneRow, point* point_Data)
{
    real myX = point_Data[thisPoint].x;
    real myY = point_Data[thisPoint].y;
    int nOfNeighbors = 0;

    for (int x = 0; x < numPoints; x++) {
        real otherX = point_Data[x].x;
        real otherY = point_Data[x].y;

        real distX = myX - otherX;
        real distY = myY - otherY;

        real Distance = sqrtf(distX * distX + distY * distY);

        if (Distance <= radius) {
            oneRow[x] = 1;
            if (x != thisPoint) {
                nOfNeighbors++;
            }
        } else {
            oneRow[x] = 0;
        }
    }
    return nOfNeighbors;
}

// ===== Cluster expansion — matches expandCluster from original =====
static int expandCluster(int thisPoint, int* clusterId, int nextClusterId,
                         int numPoints, real Eps, int MinPts,
                         int* numPointsInCluster, char* oneRow, point* point_Data)
{
    int nOfNeighbors = calcRow(thisPoint, numPoints, Eps, oneRow, point_Data);

    if (nOfNeighbors < MinPts) {
        clusterId[thisPoint] = NOISE;
        return 0;  // false
    } else {
        (*numPointsInCluster)++;
        clusterId[thisPoint] = nextClusterId;

        // Build seed queue (simple array-based queue)
        int* seeds = (int*)malloc(numPoints * sizeof(int));
        int seedHead = 0, seedTail = 0;

        for (int i = 0; i < numPoints; i++) {
            if (oneRow[i] == 1) {
                seeds[seedTail++] = i;
                clusterId[i] = nextClusterId;
            }
        }

        while (seedHead < seedTail) {
            int currentPoint = seeds[seedHead++];

            nOfNeighbors = calcRow(currentPoint, numPoints, Eps, oneRow, point_Data);

            if (nOfNeighbors >= MinPts) {
                // Core point: expand its neighbors
                for (int i = 0; i < numPoints; i++) {
                    if (oneRow[i] == 1) {
                        if (clusterId[i] == UNCLASSIFIED || clusterId[i] == NOISE) {
                            if (clusterId[i] == UNCLASSIFIED) {
                                seeds[seedTail++] = i;
                            }
                            clusterId[i] = nextClusterId;
                        }
                    }
                }
            }
            // Non-core seed points keep their cluster assignment (border points)
        }

        free(seeds);
        return 1;  // true
    }
}

// ===== Main DBSCAN — matches clusterThread from original =====
static void clusterThread(int numPoints, point* point_Data,
                          int* clusterId, real radius, int MinPts)
{
    char* oneRow = (char*)malloc(numPoints * sizeof(char));

    for (int i = 0; i < numPoints; i++) {
        clusterId[i] = UNCLASSIFIED;
    }

    int nextClusterId = 1;

    for (int i = 0; i < numPoints; i++) {
        if (clusterId[i] == UNCLASSIFIED) {
            int numPointsInCluster = 0;
            if (expandCluster(i, clusterId, nextClusterId, numPoints,
                              radius, MinPts, &numPointsInCluster,
                              oneRow, point_Data)) {
                nextClusterId++;
            }
        }
    }

    // Finalize: MARKED points that are in a cluster but not core get cluster ID
    // (In original, MARKED points get the cluster ID from their expansion)
    // Fix up: MARKED points should have been assigned during expansion
    // Any remaining MARKED points → assign to cluster 0 (shouldn't happen)
    for (int i = 0; i < numPoints; i++) {
        if (clusterId[i] == MARKED) {
            clusterId[i] = NOISE;  // fallback
        }
    }

    free(oneRow);
}

void solution_compute(int N, int* labels)
{
    clusterThread(N, g_points, labels, g_eps, g_minPts);
}

void solution_free(void)
{
    if (g_points) { free(g_points); g_points = NULL; }
}
