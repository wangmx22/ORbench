// cpu_reference.c — Monte Carlo option pricing (CPU baseline)
//
// Faithfully ported from FinanceBench/Monte-Carlo/CPU/monteCarloKernelsCpu.c
// Preserves original variable names, function structure, and computation flow.
//
// Key change from original: uses deterministic xorshift32 RNG instead of rand()
// to ensure reproducibility across platforms and calls.
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>

// ===== Constants from monteCarloConstants.h =====
#define DEFAULT_SEQ_VAL    1.0f
#define DEFAULT_SEQ_WEIGHT 1.0f
#define START_PATH_VAL     1.0f

// ===== Inverse normal distribution coefficients (from monteCarloKernelsCpu.h) =====
#define A_1 -39.696830286653757
#define A_2  220.94609842452050
#define A_3 -275.92851044696869
#define A_4  138.35775186726900
#define A_5 -30.664798066147160
#define A_6  2.5066282774592392
#define B_1 -54.476098798224058
#define B_2  161.58583685804089
#define B_3 -155.69897985988661
#define B_4  66.801311887719720
#define B_5 -13.280681552885721

typedef float dataType;

// ===== Option struct (from monteCarloStructs.h) =====
typedef struct {
    dataType riskVal;
    dataType divVal;
    dataType voltVal;
    dataType underlyingVal;
    dataType strikeVal;
    dataType discountVal;
} monteCarloOptionStruct;

// ===== Module-level state =====
static int          g_N;
static int          g_sequenceLength;
static unsigned int g_baseSeed;
static monteCarloOptionStruct g_optionStruct;

// ===== Deterministic RNG (replaces rand()) =====
// xorshift32: deterministic, fast, reproducible
static unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ===== Ported functions — names match original monteCarloKernelsCpu.c =====

// function to compute the inverse normal distribution
static dataType compInverseNormDistCpu(dataType x)
{
    dataType z;
    z = x - 0.5;
    dataType r = z * z;
    z = (((((A_1*r+A_2)*r+A_3)*r+A_4)*r+A_5)*r+A_6)*z /
                (((((B_1*r+B_2)*r+B_3)*r+B_4)*r+B_5)*r+1.0);
    return z;
}

static dataType interestRateCompoundFactCpu(dataType t, dataType rate)
{
    // assuming "continuous" option
    return exp(rate * t);
}

static dataType interestRateDiscountFactCpu(dataType t, dataType rate)
{
    return 1.0 / interestRateCompoundFactCpu(t, rate);
}

static dataType flatForwardDiscountImplCpu(dataType t, dataType rate)
{
    return interestRateDiscountFactCpu(t, rate);
}

static dataType yieldTermStructDiscountCpu(dataType t, dataType rate)
{
    return flatForwardDiscountImplCpu(t, rate);
}

static dataType interestRateImpliedRateCpu(dataType compound, dataType t)
{
    dataType r = log(compound) / t;
    return r;
}

static dataType yieldTermStructForwardRateCpu(dataType t1, dataType t2, dataType rate)
{
    dataType compound = interestRateDiscountFactCpu(t1, rate) / interestRateDiscountFactCpu(t2, rate);
    return interestRateImpliedRateCpu(compound, t2 - t1);
}

static dataType localVoltLocVolCpu(dataType t, dataType underlyingLevel, monteCarloOptionStruct optionStruct)
{
    return optionStruct.voltVal;
}

static dataType processDiffCpu(dataType t, dataType x, monteCarloOptionStruct optionStruct)
{
    return localVoltLocVolCpu(t, x, optionStruct);
}

static dataType processDriftCpu(dataType t, dataType x, monteCarloOptionStruct optionStruct)
{
    dataType sigma = processDiffCpu(t, x, optionStruct);
    dataType t1 = t + 0.0001;

    return yieldTermStructForwardRateCpu(t, t1, optionStruct.riskVal)
         - yieldTermStructForwardRateCpu(t, t1, optionStruct.divVal)
         - (0.5 * sigma * sigma);
}

static dataType discretizationDriftCpu(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return processDriftCpu(t0, x0, optionStruct) * dt;
}

static dataType discDiffCpu(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return processDiffCpu(t0, x0, optionStruct) * sqrt(dt);
}

static dataType stdDeviationCpu(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return discDiffCpu(t0, x0, dt, optionStruct);
}

static dataType applyCpu(dataType x0, dataType dx)
{
    return (x0 * exp(dx));
}

static dataType discDriftCpu(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return processDriftCpu(t0, x0, optionStruct) * dt;
}

static dataType processEvolveCpu(dataType t0, dataType x0, dataType dt, dataType dw, monteCarloOptionStruct optionStruct)
{
    return applyCpu(x0, discDriftCpu(t0, x0, dt, optionStruct) + stdDeviationCpu(t0, x0, dt, optionStruct) * dw);
}

static dataType getProcessValX0Cpu(monteCarloOptionStruct optionStruct)
{
    return optionStruct.underlyingVal;
}

// initialize the path
static void initializePathCpu(dataType* path, int seqLen)
{
    int i;
    for (i = 0; i < seqLen; i++)
    {
        path[i] = START_PATH_VAL;
    }
}

// get path — uses xorshift32 instead of rand() for determinism
static void getPathCpu(dataType* path, size_t sampleNum, dataType dt,
                       monteCarloOptionStruct optionStruct, int seqLen,
                       unsigned int baseSeed)
{
    path[0] = getProcessValX0Cpu(optionStruct);

    // Deterministic per-sample seed
    unsigned int rngState = baseSeed ^ ((unsigned int)sampleNum * 2654435761u);
    if (rngState == 0) rngState = 1;

    size_t i;
    for (i = 1; i < (size_t)seqLen; i++)
    {
        dataType t = i * dt;
        // Generate uniform random in (0, 1) via xorshift32 (replaces rand()/RAND_MAX)
        unsigned int rval = xorshift32(&rngState);
        dataType randVal = (dataType)(rval) / 4294967296.0;
        // Clamp to avoid extreme tails
        if (randVal < 0.0001) randVal = 0.0001;
        if (randVal > 0.9999) randVal = 0.9999;
        dataType inverseCumRandVal = compInverseNormDistCpu(randVal);
        path[i] = processEvolveCpu(t, path[i-1], dt, inverseCumRandVal, optionStruct);
    }
}

static dataType getPriceCpu(dataType val, dataType strikeVal, dataType discountVal)
{
    dataType diff = strikeVal - val;
    return (diff > 0.0 ? diff : 0.0) * discountVal;
}

// ===== Public interface =====

void solution_init(int N, int num_steps, float risk_free, float volatility,
                   float strike, float spot, float time_to_maturity,
                   unsigned int base_seed)
{
    g_N = N;
    g_sequenceLength = num_steps;
    g_baseSeed = base_seed;

    g_optionStruct.riskVal = risk_free;
    g_optionStruct.divVal = 0.0f;
    g_optionStruct.voltVal = volatility;
    g_optionStruct.underlyingVal = spot;
    g_optionStruct.strikeVal = strike;
    g_optionStruct.discountVal = exp(-risk_free * time_to_maturity);
}

void solution_compute(int N, float* samplePrices)
{
    dataType dt = 1.0 / (dataType)g_sequenceLength;
    int numSample;

    for (numSample = 0; numSample < N; numSample++)
    {
        // declare and initialize the path
        dataType* path = (dataType*)malloc(g_sequenceLength * sizeof(dataType));
        initializePathCpu(path, g_sequenceLength);

        getPathCpu(path, numSample, dt, g_optionStruct, g_sequenceLength, g_baseSeed);
        dataType price = getPriceCpu(path[g_sequenceLength - 1],
                                     g_optionStruct.strikeVal,
                                     g_optionStruct.discountVal);

        samplePrices[numSample] = price;
        free(path);
    }
}

void solution_free(void)
{
    /* Nothing to free — all state is static. */
}
