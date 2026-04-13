/**
 * gpu_baseline.cu - DBSCAN GPU baseline (CUDA-DClust+ port)
 *
 * Faithfully ported from resource/fast-cuda-gpu-dbscan/exp.cu
 * with three interface functions: solution_init, solution_compute, solution_free.
 *
 * Reference: Poudel & Gowanlock, "CUDA-DClust+: Revisiting Early
 *            GPU-Accelerated DBSCAN Clustering Designs", IEEE HiPC 2021.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <set>
#include <vector>
#include <map>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/replace.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/unique.h>

using namespace std;

#define RANGE 2
#define UNPROCESSED -1
#define NOISE -2

#define DIMENSION 2
#define TREE_LEVELS (DIMENSION + 1)

#define THREAD_BLOCKS 256
#define THREAD_COUNT 256

#define MAX_SEEDS 128
#define EXTRA_COLLISION_SIZE 512

__managed__ int MINPTS = 4;
__managed__ double EPS = 1.5;
__managed__ int DATASET_COUNT = 400000;
__managed__ int PARTITION_SIZE = 80;

#define POINTS_SEARCHED 9

/////////////////////////////////////////////////////////////////////////////////////////

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

struct __align__(8) IndexStructure {
  int dimension;
  int dataBegin;
  int dataEnd;
  int childFrom;
};

/////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations
/////////////////////////////////////////////////////////////////////////////////////////

__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                                    struct IndexStructure **indexBuckets,
                                    int *dataKey);

__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                        double *minPoints, double *maxPoints,
                                        double *binWidth, int *results,
                                        struct IndexStructure **indexBuckets,
                                        int *dataKey, int *dataValue,
                                        double *upperBounds);

__device__ void insertData(int id, double *dataset,
                                struct IndexStructure **indexBuckets,
                                int *dataKey, int *dataValue,
                                double *upperBounds, double *binWidth,
                                double *minPoints, double *maxPoints);

__device__ void indexConstruction(int level, int *indexTreeMetaData,
                                       double *minPoints, double *binWidth,
                                       struct IndexStructure **indexBuckets,
                                       double *upperBounds);

__device__ void searchPoints(double *data, int chainID, double *dataset,
                                  int *results,
                                  struct IndexStructure **indexBuckets,
                                  int *indexesStack, int *dataValue,
                                  double *upperBounds, double *binWidth,
                                  double *minPoints, double *maxPoints);

bool MonitorSeedPoints(vector<int>& unprocessedPoints, int* runningCluster,
                       int* d_cluster, int* d_seedList, int* d_seedLength,
                       int* d_collisionMatrix, int* d_extraCollision,
                       int* d_results, float* mergeTime, float* newSeedTime);

void GetDbscanResult(int* d_cluster, int* runningCluster, int* clusterCount,
                     int* noiseCount);

__device__ void MarkAsCandidate(int neighborID, int chainID, int* cluster,
                                int* seedList, int* seedLength,
                                int* collisionMatrix, int* extraCollision);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,
                       int *indexesStack, int *dataValue, double *upperBounds,
                       double *binWidth, double *minPoints, double *maxPoints);


/////////////////////////////////////////////////////////////////////////////////////////
// Module-level static state for the three interface functions
/////////////////////////////////////////////////////////////////////////////////////////

static double *d_dataset = NULL;
static int *d_cluster = NULL;
static int *d_seedList = NULL;
static int *d_seedLength = NULL;
static int *d_collisionMatrix = NULL;
static int *d_extraCollision = NULL;
static int *d_indexTreeMetaData = NULL;
static int *d_results = NULL;
static double *d_minPoints = NULL;
static double *d_maxPoints = NULL;
static double *d_binWidth = NULL;
static struct IndexStructure **d_indexBuckets = NULL;
static int *d_indexesStack = NULL;
static int *d_dataKey = NULL;
static int *d_dataValue = NULL;
static double *d_upperBounds = NULL;
static int s_indexedStructureSize = 0;
static vector<int> s_unprocessedPoints;

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// Function implementations (verbatim from exp.cu)
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

/**
**************************************************************************
* MonitorSeedPoints
**************************************************************************
*/
bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results, float *mergeTime, float *newSeedTime) {
  int *localSeedLength;
  localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localSeedList;
  localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  int *localCollisionMatrix;
  localCollisionMatrix =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost));

  int *localExtraCollision;
  localExtraCollision =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE);
  gpuErrchk(cudaMemcpy(localExtraCollision, d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE,
                       cudaMemcpyDeviceToHost));

  ////////////////////////////////////////////////////////////////////////////////////////

  clock_t mergeStart, mergeStop;

  mergeStart = clock();

  int clusterMap[THREAD_BLOCKS];
  set<int> blockSet;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    blockSet.insert(i);
  }

  set<int>::iterator it;

  while (blockSet.empty() == 0) {
    it = blockSet.begin();
    int curBlock = *it;
    set<int> expansionQueue;
    set<int> finalQueue;

    expansionQueue.insert(curBlock);
    finalQueue.insert(curBlock);

    while (expansionQueue.empty() == 0) {
      it = expansionQueue.begin();
      int expandBlock = *it;
      expansionQueue.erase(it);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if (localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      clusterMap[*it] = curBlock;
    }
  }

  int clusterCountMap[THREAD_BLOCKS];
  for (int x = 0; x < THREAD_BLOCKS; x++) {
    clusterCountMap[x] = UNPROCESSED;
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    if (clusterCountMap[clusterMap[x]] != UNPROCESSED) continue;
    clusterCountMap[clusterMap[x]] = (*runningCluster);
    (*runningCluster)++;
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT, x,
                    clusterCountMap[clusterMap[x]]);
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    if (localExtraCollision[x * EXTRA_COLLISION_SIZE] == -1) continue;
    int minCluster = localExtraCollision[x * EXTRA_COLLISION_SIZE];
    thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
                    clusterCountMap[clusterMap[x]], minCluster);
    for (int y = 0; y < EXTRA_COLLISION_SIZE; y++) {
      if (localExtraCollision[x * EXTRA_COLLISION_SIZE + y] == UNPROCESSED)
        break;
      int data = localExtraCollision[x * EXTRA_COLLISION_SIZE + y];
      thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
                      data, minCluster);
    }
  }

  mergeStop = clock();

  *mergeTime += (float)(mergeStop - mergeStart) / CLOCKS_PER_SEC;

  //////////////////////////////////////////////////////////////////////////////////////////

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  clock_t newSeedStart, newSeedStop;

  newSeedStart = clock();

  int complete = 0;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    bool found = false;
    while (!unprocessedPoints.empty()) {
      int lastPoint = unprocessedPoints.back();
      unprocessedPoints.pop_back();

      if (localCluster[lastPoint] == UNPROCESSED) {
        localSeedLength[i] = 1;
        localSeedList[i * MAX_SEEDS] = lastPoint;
        found = true;
        break;
      }
    }

    if (!found) {
      complete++;
    }
  }

  newSeedStop = clock();

  *newSeedTime += (float)(newSeedStop - newSeedStart) / CLOCKS_PER_SEC;
  // FInally, transfer back the CPU memory to GPU and run DBSCAN process

  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  // Free CPU memories

  free(localCluster);
  free(localSeedList);
  free(localSeedLength);
  free(localCollisionMatrix);
  free(localExtraCollision);

  if (complete == THREAD_BLOCKS) {
    return true;
  }

  return false;
}

/**
**************************************************************************
* MarkAsCandidate
**************************************************************************
*/
__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *collisionMatrix, int *extraCollision) {
  register int oldState =
      atomicCAS(&(cluster[neighborID]), UNPROCESSED, chainID);

  if (oldState == UNPROCESSED) {
    register int sl = atomicAdd(&(seedLength[chainID]), 1);
    if (sl < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + sl] = neighborID;
    }
  }

  else if (oldState >= THREAD_BLOCKS) {
    for (int i = 0; i < EXTRA_COLLISION_SIZE; i++) {
      register int changedState =
          atomicCAS(&(extraCollision[chainID * EXTRA_COLLISION_SIZE + i]),
                    UNPROCESSED, oldState);
      if (changedState == UNPROCESSED || changedState == oldState) {
        break;
      }
    }
  }

  else if (oldState != NOISE && oldState != chainID &&
           oldState < THREAD_BLOCKS) {
    collisionMatrix[oldState * THREAD_BLOCKS + chainID] = 1;
    collisionMatrix[chainID * THREAD_BLOCKS + oldState] = 1;
  }

  else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  }
}


/**
**************************************************************************
* GetDbscanResult
**************************************************************************
*/
void GetDbscanResult(int *d_cluster, int *runningCluster, int *clusterCount,
                     int *noiseCount) {
  *noiseCount = thrust::count(thrust::device, d_cluster,
                              d_cluster + DATASET_COUNT, NOISE);
  int *d_localCluster;
  gpuErrchk(cudaMalloc((void **)&d_localCluster, sizeof(int) * DATASET_COUNT));
  thrust::copy(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
               d_localCluster);
  thrust::sort(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT);
  *clusterCount = thrust::unique(thrust::device, d_localCluster,
                                 d_localCluster + DATASET_COUNT) -
                  d_localCluster - 1;

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_localCluster,
                       sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost));
  free(localCluster);

  cudaFree(d_localCluster);
}


/**
**************************************************************************
* DBSCAN kernel
**************************************************************************
*/
__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,
                       int *indexesStack, int *dataValue, double *upperBounds,
                       double *binWidth, double *minPoints, double *maxPoints) {
  // Point ID to expand by a block
  __shared__ int pointID;

  // Neighbors to store of neighbors points exceeds minpoints
  __shared__ int neighborBuffer[64];

  // It counts the total neighbors
  __shared__ int neighborCount;

  // ChainID is basically blockID
  __shared__ int chainID;

  // Store the point from pointID
  __shared__ double point[DIMENSION];

  // Length of the seedlist to check its size
  __shared__ int currentSeedLength;

  __shared__ int resultId;

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
    currentSeedLength = seedLength[chainID];
    pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int x = threadId; x < THREAD_BLOCKS * THREAD_BLOCKS;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    collisionMatrix[x] = UNPROCESSED;
  }
  for (int x = threadId; x < THREAD_BLOCKS * EXTRA_COLLISION_SIZE;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    extraCollision[x] = UNPROCESSED;
  }

  __syncthreads();

  // Complete the seedlist to proceed.

  while (seedLength[chainID] != 0) {
    for (int x = threadId; x < THREAD_BLOCKS * POINTS_SEARCHED;
         x = x + THREAD_BLOCKS * THREAD_COUNT) {
      results[x] = UNPROCESSED;
    }
    __syncthreads();

    // Assign chainID, current seed length and pointID
    if (threadIdx.x == 0) {
      chainID = blockIdx.x;
      currentSeedLength = seedLength[chainID];
      pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
    }
    __syncthreads();

    // Check if the point is already processed
    if (threadIdx.x == 0) {
      seedLength[chainID] = currentSeedLength - 1;
      neighborCount = 0;
      for (int x = 0; x < DIMENSION; x++) {
        point[x] = dataset[pointID * DIMENSION + x];
      }
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    searchPoints(point, chainID, dataset, results, indexBuckets, indexesStack,
                 dataValue, upperBounds, binWidth, minPoints, maxPoints);

    __syncthreads();

    for (int k = 0; k < POINTS_SEARCHED; k++) {
      if (threadIdx.x == 0) {
        resultId = results[chainID * POINTS_SEARCHED + k];
      }
      __syncthreads();

      if (resultId == -1) break;

      for (int i = threadIdx.x + indexBuckets[resultId]->dataBegin;
           i < indexBuckets[resultId]->dataEnd; i = i + THREAD_COUNT) {
        register double comparingPoint[DIMENSION];

        for (int x = 0; x < DIMENSION; x++) {
          comparingPoint[x] = dataset[dataValue[i] * DIMENSION + x];
        }

        register double distance = 0;
        for (int x = 0; x < DIMENSION; x++) {
          distance +=
              (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
        }

        if (distance <= EPS * EPS) {
          register int currentNeighborCount = atomicAdd(&neighborCount, 1);
          if (currentNeighborCount >= MINPTS) {
            MarkAsCandidate(dataValue[i], chainID, cluster, seedList,
                            seedLength, collisionMatrix, extraCollision);
          } else {
            neighborBuffer[currentNeighborCount] = dataValue[i];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    if (neighborCount >= MINPTS) {
      cluster[pointID] = chainID;
      for (int i = threadIdx.x; i < MINPTS; i = i + THREAD_COUNT) {
        MarkAsCandidate(neighborBuffer[i], chainID, cluster, seedList,
                        seedLength, collisionMatrix, extraCollision);
      }
    } else {
      cluster[pointID] = NOISE;
    }

    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////

    if (threadIdx.x == 0 && seedLength[chainID] >= MAX_SEEDS) {
      seedLength[chainID] = MAX_SEEDS - 1;
    }
    __syncthreads();
  }
}


/**
**************************************************************************
* searchPoints
**************************************************************************
*/
__device__ void searchPoints(double *data, int chainID, double *dataset,
                                  int *results,
                                  struct IndexStructure **indexBuckets,
                                  int *indexesStack, int *dataValue,
                                  double *upperBounds, double *binWidth, double *minPoints, double *maxPoints) {

  __shared__ int resultsCount;
  __shared__ int indexBucketSize;
  __shared__ int currentIndex;
  __shared__ int currentIndexSize;
  __shared__ double comparingData;

  if (threadIdx.x == 0) {
    resultsCount = 0;
    indexBucketSize = 1;
    for (int i = 0; i < DIMENSION; i++) {
      indexBucketSize *= 3;
    }
    indexBucketSize = indexBucketSize * chainID;
    currentIndexSize = indexBucketSize;
    indexesStack[currentIndexSize++] = 0;
  }
  __syncthreads();

  while (currentIndexSize > indexBucketSize) {
    if (threadIdx.x == 0) {
      currentIndexSize = currentIndexSize - 1;
      currentIndex = indexesStack[currentIndexSize];
      comparingData = data[indexBuckets[currentIndex]->dimension];
    }
    __syncthreads();

    for (int k = threadIdx.x + indexBuckets[currentIndex]->childFrom;
         k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE;
         k = k + THREAD_COUNT) {
      double leftRange;
      double rightRange;
      if (k == indexBuckets[currentIndex]->childFrom) {
        leftRange =
            upperBounds[k] - binWidth[indexBuckets[currentIndex]->dimension];
      } else {
        leftRange = upperBounds[k - 1];
      }

      rightRange = upperBounds[k];

      if (comparingData >= leftRange && comparingData < rightRange) {
        if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
          int oldResultsCount = atomicAdd(&resultsCount, 1);
          results[chainID * POINTS_SEARCHED + oldResultsCount] = k;

          if (k > indexBuckets[currentIndex]->childFrom) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] = k - 1;
          }

          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] = k + 1;
          }
        } else {
          int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
          indexesStack[oldCurrentIndexSize] = k;
          if (k > indexBuckets[currentIndex]->childFrom) {
            int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
            indexesStack[oldCurrentIndexSize] = k - 1;
          }
          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
            indexesStack[oldCurrentIndexSize] = k + 1;
          }
        }
      }
    }

    __syncthreads();
  }
}


/**
**************************************************************************
* INDEXING_ADJUSTMENT kernel
**************************************************************************
*/
__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                                    struct IndexStructure **indexBuckets,
                                    int *dataKey) {
  __shared__ int indexingRange;
  if (threadIdx.x == 0) {
    indexingRange = indexTreeMetaData[DIMENSION * RANGE + 1] -
                    indexTreeMetaData[DIMENSION * RANGE];
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = threadId; i < indexingRange;
       i = i + THREAD_COUNT * THREAD_BLOCKS) {
    int idx = indexTreeMetaData[DIMENSION * RANGE] + i;

    thrust::pair<int *, int *> dataPositioned;

    dataPositioned = thrust::equal_range(thrust::device, dataKey, dataKey + DATASET_COUNT, idx);

    indexBuckets[idx]->dataBegin = dataPositioned.first - dataKey;
    indexBuckets[idx]->dataEnd = dataPositioned.second - dataKey;
  }
  __syncthreads();
}


/**
**************************************************************************
* INDEXING_STRUCTURE kernel
**************************************************************************
*/
__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                   double *minPoints, double *maxPoints, double *binWidth,
                                   int *results,
                                   struct IndexStructure **indexBuckets,
                                   int *dataKey, int *dataValue,
                                   double *upperBounds) {
  if (blockIdx.x < DIMENSION) {
    indexConstruction(blockIdx.x, indexTreeMetaData, minPoints, binWidth,
                      indexBuckets, upperBounds);
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < DATASET_COUNT;
       i = i + THREAD_COUNT * THREAD_BLOCKS) {
    insertData(i, dataset, indexBuckets, dataKey, dataValue, upperBounds,
               binWidth, minPoints, maxPoints);
  }
  __syncthreads();
}

/**
**************************************************************************
* indexConstruction
**************************************************************************
*/
__device__ void indexConstruction(int level, int *indexTreeMetaData,
                                  double *minPoints, double *binWidth,
                                  struct IndexStructure **indexBuckets,
                                  double *upperBounds) {
  for (int k = threadIdx.x + indexTreeMetaData[level * RANGE + 0];
       k < indexTreeMetaData[level * RANGE + 1]; k = k + THREAD_COUNT) {
    for (int i = 0; i < PARTITION_SIZE; i++) {
      int currentBucketIndex =
          indexTreeMetaData[level * RANGE + 1] + i +
          (k - indexTreeMetaData[level * RANGE + 0]) * PARTITION_SIZE;

      indexBuckets[k]->dimension = level;
      indexBuckets[currentBucketIndex]->dimension = level + 1;

      if (i == 0) {
        indexBuckets[k]->childFrom = currentBucketIndex;
      }

      double rightPoint =
          minPoints[level] + i * binWidth[level] + binWidth[level];

      if (i == PARTITION_SIZE - 1) rightPoint = rightPoint + binWidth[level];

      upperBounds[currentBucketIndex] = rightPoint;
    }
  }
  __syncthreads();
}

/**
**************************************************************************
* insertData
**************************************************************************
*/
__device__ void insertData(int id, double *dataset,
                           struct IndexStructure **indexBuckets, int *dataKey,
                           int *dataValue, double *upperBounds,
                           double *binWidth, double *minPoints, double *maxPoints) {
  int index = 0;
  for (int j = 0; j < DIMENSION; j++) {
    double x = dataset[id * DIMENSION + j];
    int currentIndex = (x - minPoints[j]) / (maxPoints[j] - minPoints[j]) * PARTITION_SIZE + 1;
    index = index * PARTITION_SIZE + currentIndex;
  }

  dataValue[id] = id;
  dataKey[id] = index;
}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// Interface functions
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

extern "C" void solution_init(int N, const float* xs, const float* ys, float eps, int minPts) {
  // Set managed variables
  DATASET_COUNT = N;
  EPS = (double)eps;
  MINPTS = minPts;

  // Choose PARTITION_SIZE based on data range / eps
  // Find min/max to compute range, then pick partition size
  double maxPoints_h[DIMENSION];
  double minPoints_h[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    maxPoints_h[j] = -1e18;
    minPoints_h[j] = 1e18;
  }

  // Convert float xs/ys to double dataset (interleaved x,y format)
  double *importedDataset = (double *)malloc(sizeof(double) * N * DIMENSION);
  for (int i = 0; i < N; i++) {
    importedDataset[i * DIMENSION + 0] = (double)xs[i];
    importedDataset[i * DIMENSION + 1] = (double)ys[i];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      if (importedDataset[i * DIMENSION + j] > maxPoints_h[j]) {
        maxPoints_h[j] = importedDataset[i * DIMENSION + j];
      }
      if (importedDataset[i * DIMENSION + j] < minPoints_h[j]) {
        minPoints_h[j] = importedDataset[i * DIMENSION + j];
      }
    }
  }

  // Choose PARTITION_SIZE: ensure binWidth >= EPS
  {
    double maxRange = 0;
    for (int j = 0; j < DIMENSION; j++) {
      double r = maxPoints_h[j] - minPoints_h[j];
      if (r > maxRange) maxRange = r;
    }
    // partition = range / eps, but we need binWidth >= eps
    // binWidth = range / partition, so partition <= range / eps
    int p = (int)(maxRange / EPS);
    if (p < 2) p = 2;
    if (p > 200) p = 200;
    PARTITION_SIZE = p;
  }

  // Verify binWidth >= EPS
  double binWidth_h[DIMENSION];
  for (int x = 0; x < DIMENSION; x++) {
    binWidth_h[x] = (double)(maxPoints_h[x] - minPoints_h[x]) / PARTITION_SIZE;
  }
  // If any bin is too small, reduce PARTITION_SIZE
  {
    double minBin = 1e18;
    for (int x = 0; x < DIMENSION; x++) {
      if (binWidth_h[x] < minBin) minBin = binWidth_h[x];
    }
    while (minBin < EPS && PARTITION_SIZE > 2) {
      PARTITION_SIZE--;
      for (int x = 0; x < DIMENSION; x++) {
        binWidth_h[x] = (double)(maxPoints_h[x] - minPoints_h[x]) / PARTITION_SIZE;
      }
      minBin = 1e18;
      for (int x = 0; x < DIMENSION; x++) {
        if (binWidth_h[x] < minBin) minBin = binWidth_h[x];
      }
    }
  }

  fprintf(stderr, "DBSCAN init: N=%d, EPS=%.6f, MINPTS=%d, PARTITION_SIZE=%d\n",
          DATASET_COUNT, EPS, MINPTS, PARTITION_SIZE);
  fprintf(stderr, "  binWidth: %.6f %.6f\n", binWidth_h[0], binWidth_h[1]);
  fprintf(stderr, "  range: [%.4f,%.4f] x [%.4f,%.4f]\n",
          minPoints_h[0], maxPoints_h[0], minPoints_h[1], maxPoints_h[1]);

  // CUDA Memory allocation
  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));

  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  // Indexing Memory allocation
  gpuErrchk(cudaMalloc((void **)&d_indexTreeMetaData,
                       sizeof(int) * TREE_LEVELS * RANGE));

  gpuErrchk(cudaMalloc((void **)&d_results,
                       sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  gpuErrchk(cudaMalloc((void **)&d_minPoints, sizeof(double) * DIMENSION));
  gpuErrchk(cudaMalloc((void **)&d_maxPoints, sizeof(double) * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_binWidth, sizeof(double) * DIMENSION));

  gpuErrchk(
      cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  // Assignment with default values
  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));

  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  // Initialize index structure
  // Level Partition
  int treeLevelPartition[TREE_LEVELS] = {1};

  for (int i = 0; i < DIMENSION; i++) {
    treeLevelPartition[i + 1] = PARTITION_SIZE;
  }

  int childItems[TREE_LEVELS];
  int startEndIndexes[TREE_LEVELS * RANGE];

  int mulx = 1;
  for (int k = 0; k < TREE_LEVELS; k++) {
    mulx *= treeLevelPartition[k];
    childItems[k] = mulx;
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    if (i == 0) {
      startEndIndexes[i * RANGE + 0] = 0;
      startEndIndexes[i * RANGE + 1] = 1;
      continue;
    }
    startEndIndexes[i * RANGE + 0] = startEndIndexes[((i - 1) * RANGE) + 1];
    startEndIndexes[i * RANGE + 1] = startEndIndexes[i * RANGE + 0];
    for (int k = 0; k < childItems[i - 1]; k++) {
      startEndIndexes[i * RANGE + 1] += treeLevelPartition[i];
    }
  }

  gpuErrchk(cudaMemcpy(d_minPoints, minPoints_h, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_maxPoints, maxPoints_h, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_binWidth, binWidth_h, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));

  s_indexedStructureSize = startEndIndexes[DIMENSION * RANGE + 1];

  // Allocate memory for index buckets
  struct IndexStructure *d_currentIndexBucket;

  gpuErrchk(cudaMalloc((void **)&d_indexBuckets,
                       sizeof(struct IndexStructure *) * s_indexedStructureSize));

  for (int i = 0; i < s_indexedStructureSize; i++) {
    gpuErrchk(cudaMalloc((void **)&d_currentIndexBucket,
                         sizeof(struct IndexStructure)));
    gpuErrchk(cudaMemcpy(&d_indexBuckets[i], &d_currentIndexBucket,
                         sizeof(struct IndexStructure *),
                         cudaMemcpyHostToDevice));
  }

  // Allocate memory for current indexes stack
  int indexBucketSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  indexBucketSize = indexBucketSize * THREAD_BLOCKS;

  gpuErrchk(
      cudaMalloc((void **)&d_indexesStack, sizeof(int) * indexBucketSize));

  // NOTE: exp.cu frees d_currentIndexBucket here, but that is a local temp pointer.
  // The actual bucket memory is referenced via d_indexBuckets array.
  cudaFree(d_currentIndexBucket);

  // Data key-value pair
  gpuErrchk(cudaMalloc((void **)&d_dataKey, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_dataValue, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_upperBounds,
                       sizeof(double) * s_indexedStructureSize));

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16*1024*1024);

  // Start Indexing
  gpuErrchk(cudaDeviceSynchronize());

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_dataset, d_indexTreeMetaData, d_minPoints, d_maxPoints, d_binWidth, d_results,
      d_indexBuckets, d_dataKey, d_dataValue, d_upperBounds);
  gpuErrchk(cudaDeviceSynchronize());

  // BUG FIX: exp.cu line 390 does cudaFree(d_indexTreeMetaData) BEFORE calling
  // INDEXING_ADJUSTMENT which USES d_indexTreeMetaData. We do NOT free it here.

  // Sorting Data key-value pair
  thrust::sort_by_key(thrust::device, d_dataKey, d_dataKey + DATASET_COUNT,
                      d_dataValue);

  gpuErrchk(cudaDeviceSynchronize());

  INDEXING_ADJUSTMENT<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_indexTreeMetaData, d_indexBuckets, d_dataKey);

  gpuErrchk(cudaDeviceSynchronize());

  free(importedDataset);
}


extern "C" void solution_compute(int N, int* labels) {
  // Reset cluster/seed state
  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));

  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  gpuErrchk(
      cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  // Rebuild unprocessedPoints
  s_unprocessedPoints.clear();
  for (int x = 0; x < DATASET_COUNT; x++) {
    s_unprocessedPoints.push_back(x);
  }

  // Keep track of number of cluster formed without global merge
  int runningCluster = THREAD_BLOCKS;
  // Global cluster count
  int clusterCount = 0;
  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to control the while loop
  bool exitLoop = false;

  float mergeTime = 0.0;
  float newSeedTime = 0.0;

  while (!exitLoop) {
    // Monitor the seed list and return the completion status of points
    int completed =
        MonitorSeedPoints(s_unprocessedPoints, &runningCluster,
                          d_cluster, d_seedList, d_seedLength,
                          d_collisionMatrix, d_extraCollision, d_results, &mergeTime, &newSeedTime);

    // If all points are processed, exit
    if (completed) {
      exitLoop = true;
    }

    if (exitLoop) break;

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_results, d_indexBuckets, d_indexesStack,
        d_dataValue, d_upperBounds, d_binWidth, d_minPoints, d_maxPoints);
    gpuErrchk(cudaDeviceSynchronize());
  }

  // Copy d_cluster to host
  int *hostCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(hostCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  // Remap IDs: find unique cluster IDs (excluding NOISE), assign 1,2,3,...
  // NOISE (-2) -> -2 in labels
  // Collect unique non-noise cluster IDs
  set<int> uniqueClusters;
  for (int i = 0; i < N; i++) {
    if (hostCluster[i] != NOISE) {
      uniqueClusters.insert(hostCluster[i]);
    }
  }

  // Build mapping: original cluster ID -> sequential 1,2,3,...
  std::map<int,int> clusterRemap;
  int nextLabel = 1;
  for (set<int>::iterator it = uniqueClusters.begin(); it != uniqueClusters.end(); ++it) {
    clusterRemap[*it] = nextLabel++;
  }

  for (int i = 0; i < N; i++) {
    if (hostCluster[i] == NOISE) {
      labels[i] = -2;
    } else {
      labels[i] = clusterRemap[hostCluster[i]];
    }
  }

  free(hostCluster);
}


extern "C" void solution_free(void) {
  // Free individual index bucket allocations
  if (d_indexBuckets != NULL) {
    struct IndexStructure *h_ptr;
    for (int i = 0; i < s_indexedStructureSize; i++) {
      gpuErrchk(cudaMemcpy(&h_ptr, &d_indexBuckets[i],
                           sizeof(struct IndexStructure *),
                           cudaMemcpyDeviceToHost));
      cudaFree(h_ptr);
    }
  }

  if (d_dataset) { cudaFree(d_dataset); d_dataset = NULL; }
  if (d_cluster) { cudaFree(d_cluster); d_cluster = NULL; }
  if (d_seedList) { cudaFree(d_seedList); d_seedList = NULL; }
  if (d_seedLength) { cudaFree(d_seedLength); d_seedLength = NULL; }
  if (d_collisionMatrix) { cudaFree(d_collisionMatrix); d_collisionMatrix = NULL; }
  if (d_extraCollision) { cudaFree(d_extraCollision); d_extraCollision = NULL; }

  if (d_indexTreeMetaData) { cudaFree(d_indexTreeMetaData); d_indexTreeMetaData = NULL; }
  if (d_results) { cudaFree(d_results); d_results = NULL; }
  if (d_indexBuckets) { cudaFree(d_indexBuckets); d_indexBuckets = NULL; }
  if (d_indexesStack) { cudaFree(d_indexesStack); d_indexesStack = NULL; }

  if (d_dataKey) { cudaFree(d_dataKey); d_dataKey = NULL; }
  if (d_dataValue) { cudaFree(d_dataValue); d_dataValue = NULL; }
  if (d_upperBounds) { cudaFree(d_upperBounds); d_upperBounds = NULL; }
  if (d_binWidth) { cudaFree(d_binWidth); d_binWidth = NULL; }

  if (d_minPoints) { cudaFree(d_minPoints); d_minPoints = NULL; }
  if (d_maxPoints) { cudaFree(d_maxPoints); d_maxPoints = NULL; }

  s_unprocessedPoints.clear();
  s_indexedStructureSize = 0;
}
