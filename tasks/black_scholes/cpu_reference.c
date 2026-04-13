// cpu_reference.c — Black-Scholes option pricing (CPU baseline)
//
// Faithfully ported from FinanceBench/Black-Scholes/CPU/
//   blackScholesAnalyticEngineKernelsCpu.c
//   blackScholesAnalyticEngineStructs.h
//   errorFunctConsts.h
//
// Preserves original struct definitions, function names, and computation flow.
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>

// ===== Option type constants (from blackScholesAnalyticEngineStructs.h) =====
#define CALL 0
#define PUT  1

// ===== Structs from blackScholesAnalyticEngineStructs.h =====

typedef struct {
    float rate;
    float freq;
    int comp;
} interestRateStruct;

typedef struct {
    float timeYearFraction;
    float forward;
    float compounding;
    float frequency;
    float intRate;
} yieldTermStruct;

typedef struct {
    float timeYearFraction;
    float following;
    float volatility;
} blackVolStruct;

typedef struct {
    float x0;
    yieldTermStruct dividendTS;
    yieldTermStruct riskFreeTS;
    blackVolStruct blackVolTS;
} blackScholesMertStruct;

typedef struct {
    int type;
    float strike;
} payoffStruct;

typedef struct {
    payoffStruct payoff;
    float yearFractionTime;
    blackScholesMertStruct pricingEngine;
} optionStruct;

typedef struct {
    float strike;
    float forward;
    float stdDev;
    float discount;
    float variance;
    float d1;
    float d2;
    float alpha;
    float beta;
    float DalphaDd1;
    float DbetaDd2;
    float n_d1;
    float cum_d1;
    float n_d2;
    float cum_d2;
    float x;
    float DxDs;
    float DxDstrike;
} blackCalcStruct;

typedef struct {
    float average;
    float sigma;
    float denominator;
    float derNormalizationFactor;
    float normalizationFactor;
} normalDistStruct;

typedef struct {
    int type;
    float strike;
    float spot;
    float q;
    float r;
    float t;
    float vol;
    float value;
    float tol;
} optionInputStruct;

// ===== Error function constants (from errorFunctConsts.h) =====
#define BS_DBL_MIN 1e-30f

#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244008443621048490392848359376887
#endif

#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

#define ERROR_FUNCT_tiny   0.000000000000000000001f
#define ERROR_FUNCT_one    1.00000000000000000000e+00
#define ERROR_FUNCT_erx    8.45062911510467529297e-01
#define ERROR_FUNCT_efx    1.28379167095512586316e-01
#define ERROR_FUNCT_efx8   1.02703333676410069053e+00
#define ERROR_FUNCT_pp0    1.28379167095512558561e-01
#define ERROR_FUNCT_pp1   -3.25042107247001499370e-01
#define ERROR_FUNCT_pp2   -2.84817495755985104766e-02
#define ERROR_FUNCT_pp3   -5.77027029648944159157e-03
#define ERROR_FUNCT_pp4   -2.37630166566501626084e-05
#define ERROR_FUNCT_qq1    3.97917223959155352819e-01
#define ERROR_FUNCT_qq2    6.50222499887672944485e-02
#define ERROR_FUNCT_qq3    5.08130628187576562776e-03
#define ERROR_FUNCT_qq4    1.32494738004321644526e-04
#define ERROR_FUNCT_qq5   -3.96022827877536812320e-06
#define ERROR_FUNCT_pa0   -2.36211856075265944077e-03
#define ERROR_FUNCT_pa1    4.14856118683748331666e-01
#define ERROR_FUNCT_pa2   -3.72207876035701323847e-01
#define ERROR_FUNCT_pa3    3.18346619901161753674e-01
#define ERROR_FUNCT_pa4   -1.10894694282396677476e-01
#define ERROR_FUNCT_pa5    3.54783043256182359371e-02
#define ERROR_FUNCT_pa6   -2.16637559486879084300e-03
#define ERROR_FUNCT_qa1    1.06420880400844228286e-01
#define ERROR_FUNCT_qa2    5.40397917702171048937e-01
#define ERROR_FUNCT_qa3    7.18286544141962662868e-02
#define ERROR_FUNCT_qa4    1.26171219808761642112e-01
#define ERROR_FUNCT_qa5    1.36370839120290507362e-02
#define ERROR_FUNCT_qa6    1.19844998467991074170e-02
#define ERROR_FUNCT_ra0   -9.86494403484714822705e-03
#define ERROR_FUNCT_ra1   -6.93858572707181764372e-01
#define ERROR_FUNCT_ra2   -1.05586262253232909814e+01
#define ERROR_FUNCT_ra3   -6.23753324503260060396e+01
#define ERROR_FUNCT_ra4   -1.62396669462573470355e+02
#define ERROR_FUNCT_ra5   -1.84605092906711035994e+02
#define ERROR_FUNCT_ra6   -8.12874355063065934246e+01
#define ERROR_FUNCT_ra7   -9.81432934416914548592e+00
#define ERROR_FUNCT_sa1    1.96512716674392571292e+01
#define ERROR_FUNCT_sa2    1.37657754143519042600e+02
#define ERROR_FUNCT_sa3    4.34565877475229228821e+02
#define ERROR_FUNCT_sa4    6.45387271733267880336e+02
#define ERROR_FUNCT_sa5    4.29008140027567833386e+02
#define ERROR_FUNCT_sa6    1.08635005541779435134e+02
#define ERROR_FUNCT_sa7    6.57024977031928170135e+00
#define ERROR_FUNCT_sa8   -6.04244152148580987438e-02
#define ERROR_FUNCT_rb0   -9.86494292470009928597e-03
#define ERROR_FUNCT_rb1   -7.99283237680523006574e-01
#define ERROR_FUNCT_rb2   -1.77579549177547519889e+01
#define ERROR_FUNCT_rb3   -1.60636384855821916062e+02
#define ERROR_FUNCT_rb4   -6.37566443368389627722e+02
#define ERROR_FUNCT_rb5   -1.02509513161107724954e+03
#define ERROR_FUNCT_rb6   -4.83519191608651397019e+02
#define ERROR_FUNCT_sb1    3.03380607434824582924e+01
#define ERROR_FUNCT_sb2    3.25792512996573918826e+02
#define ERROR_FUNCT_sb3    1.53672958608443695994e+03
#define ERROR_FUNCT_sb4    3.19985821950859553908e+03
#define ERROR_FUNCT_sb5    2.55305040643316442583e+03
#define ERROR_FUNCT_sb6    4.74528541206955367215e+02
#define ERROR_FUNCT_sb7   -2.24409524465858183362e+01

// ===== Module-level state =====
static int g_N;
static const int*   g_types;
static const float* g_strikes;
static const float* g_spots;
static const float* g_qs;
static const float* g_rs;
static const float* g_ts;
static const float* g_vols;

// ===== Functions ported from blackScholesAnalyticEngineKernelsCpu.c =====
// Names preserved exactly as in original.

static float interestRateCompoundFactorCpu(float t, yieldTermStruct currYieldTermStruct)
{
    return (exp((currYieldTermStruct.forward) * t));
}

static float interestRateDiscountFactorCpu(float t, yieldTermStruct currYieldTermStruct)
{
    return 1.0f / interestRateCompoundFactorCpu(t, currYieldTermStruct);
}

static float getBlackVolBlackVarCpu(blackVolStruct volTS)
{
    float vol = volTS.volatility;
    return vol * vol * volTS.timeYearFraction;
}

static float getDiscountOnDividendYieldCpu(float yearFraction, yieldTermStruct dividendYieldTermStruct)
{
    float intDiscountFactor = interestRateDiscountFactorCpu(yearFraction, dividendYieldTermStruct);
    return intDiscountFactor;
}

static float getDiscountOnRiskFreeRateCpu(float yearFraction, yieldTermStruct riskFreeRateYieldTermStruct)
{
    return interestRateDiscountFactorCpu(yearFraction, riskFreeRateYieldTermStruct);
}

static float errorFunctCpu(normalDistStruct normDist, float x)
{
    float R, S, P, Q, s, y, z, r, ax;

    ax = fabs(x);

    if (ax < 0.84375) {
        if (ax < 3.7252902984e-09) {
            if (ax < BS_DBL_MIN * 16)
                return 0.125 * (8.0 * x + (ERROR_FUNCT_efx8) * x);
            return x + (ERROR_FUNCT_efx) * x;
        }
        z = x * x;
        r = ERROR_FUNCT_pp0 + z * (ERROR_FUNCT_pp1 + z * (ERROR_FUNCT_pp2 + z * (ERROR_FUNCT_pp3 + z * ERROR_FUNCT_pp4)));
        s = ERROR_FUNCT_one + z * (ERROR_FUNCT_qq1 + z * (ERROR_FUNCT_qq2 + z * (ERROR_FUNCT_qq3 + z * (ERROR_FUNCT_qq4 + z * ERROR_FUNCT_qq5))));
        y = r / s;
        return x + x * y;
    }
    if (ax < 1.25) {
        s = ax - ERROR_FUNCT_one;
        P = ERROR_FUNCT_pa0 + s * (ERROR_FUNCT_pa1 + s * (ERROR_FUNCT_pa2 + s * (ERROR_FUNCT_pa3 + s * (ERROR_FUNCT_pa4 + s * (ERROR_FUNCT_pa5 + s * ERROR_FUNCT_pa6)))));
        Q = ERROR_FUNCT_one + s * (ERROR_FUNCT_qa1 + s * (ERROR_FUNCT_qa2 + s * (ERROR_FUNCT_qa3 + s * (ERROR_FUNCT_qa4 + s * (ERROR_FUNCT_qa5 + s * ERROR_FUNCT_qa6)))));
        if (x >= 0) return ERROR_FUNCT_erx + P / Q;
        else return -1 * ERROR_FUNCT_erx - P / Q;
    }
    if (ax >= 6) {
        if (x >= 0) return ERROR_FUNCT_one - ERROR_FUNCT_tiny;
        else return ERROR_FUNCT_tiny - ERROR_FUNCT_one;
    }

    s = ERROR_FUNCT_one / (ax * ax);

    if (ax < 2.85714285714285) {
        R = ERROR_FUNCT_ra0 + s * (ERROR_FUNCT_ra1 + s * (ERROR_FUNCT_ra2 + s * (ERROR_FUNCT_ra3 + s * (ERROR_FUNCT_ra4 + s * (ERROR_FUNCT_ra5 + s * (ERROR_FUNCT_ra6 + s * ERROR_FUNCT_ra7))))));
        S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sa1 + s * (ERROR_FUNCT_sa2 + s * (ERROR_FUNCT_sa3 + s * (ERROR_FUNCT_sa4 + s * (ERROR_FUNCT_sa5 + s * (ERROR_FUNCT_sa6 + s * (ERROR_FUNCT_sa7 + s * ERROR_FUNCT_sa8)))))));
    } else {
        R = ERROR_FUNCT_rb0 + s * (ERROR_FUNCT_rb1 + s * (ERROR_FUNCT_rb2 + s * (ERROR_FUNCT_rb3 + s * (ERROR_FUNCT_rb4 + s * (ERROR_FUNCT_rb5 + s * ERROR_FUNCT_rb6)))));
        S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sb1 + s * (ERROR_FUNCT_sb2 + s * (ERROR_FUNCT_sb3 + s * (ERROR_FUNCT_sb4 + s * (ERROR_FUNCT_sb5 + s * (ERROR_FUNCT_sb6 + s * ERROR_FUNCT_sb7))))));
    }

    r = exp(-ax * ax - 0.5625 + R / S);
    if (x >= 0) return ERROR_FUNCT_one - r / ax;
    else return r / ax - ERROR_FUNCT_one;
}

static float cumNormDistOpCpu(normalDistStruct normDist, float z)
{
    z = (z - normDist.average) / normDist.sigma;
    float result = 0.5 * (1.0 + errorFunctCpu(normDist, z * M_SQRT_2));
    return result;
}

static float gaussianFunctNormDistCpu(normalDistStruct normDist, float x)
{
    float deltax = x - normDist.average;
    float exponent = -(deltax * deltax) / normDist.denominator;
    return exponent <= -690.0 ? 0.0 :
        normDist.normalizationFactor * exp(exponent);
}

static float cumNormDistDerivCpu(normalDistStruct normDist, float x)
{
    float xn = (x - normDist.average) / normDist.sigma;
    return gaussianFunctNormDistCpu(normDist, xn) / normDist.sigma;
}

static void initCumNormDistCpu(normalDistStruct* currCumNormDist)
{
    currCumNormDist->average = 0.0f;
    currCumNormDist->sigma = 1.0f;
    currCumNormDist->normalizationFactor = M_SQRT_2 * M_1_SQRTPI / currCumNormDist->sigma;
    currCumNormDist->derNormalizationFactor = currCumNormDist->sigma * currCumNormDist->sigma;
    currCumNormDist->denominator = 2.0 * currCumNormDist->derNormalizationFactor;
}

static void initBlackCalcVarsCpu(blackCalcStruct* blackCalculator, payoffStruct payoff)
{
    blackCalculator->d1 = log(blackCalculator->forward / blackCalculator->strike) / blackCalculator->stdDev + 0.5 * blackCalculator->stdDev;
    blackCalculator->d2 = blackCalculator->d1 - blackCalculator->stdDev;

    normalDistStruct currCumNormDist;
    initCumNormDistCpu(&currCumNormDist);

    blackCalculator->cum_d1 = cumNormDistOpCpu(currCumNormDist, blackCalculator->d1);
    blackCalculator->cum_d2 = cumNormDistOpCpu(currCumNormDist, blackCalculator->d2);
    blackCalculator->n_d1 = cumNormDistDerivCpu(currCumNormDist, blackCalculator->d1);
    blackCalculator->n_d2 = cumNormDistDerivCpu(currCumNormDist, blackCalculator->d2);

    blackCalculator->x = payoff.strike;
    blackCalculator->DxDstrike = 1.0;
    blackCalculator->DxDs = 0.0;

    switch (payoff.type) {
    case CALL:
        blackCalculator->alpha     =  blackCalculator->cum_d1;
        blackCalculator->DalphaDd1 =  blackCalculator->n_d1;
        blackCalculator->beta      = -1.0f * blackCalculator->cum_d2;
        blackCalculator->DbetaDd2  = -1.0f * blackCalculator->n_d2;
        break;
    case PUT:
        blackCalculator->alpha     = -1.0 + blackCalculator->cum_d1;
        blackCalculator->DalphaDd1 =        blackCalculator->n_d1;
        blackCalculator->beta      =  1.0 - blackCalculator->cum_d2;
        blackCalculator->DbetaDd2  = -1.0f * blackCalculator->n_d2;
        break;
    }
}

static void initBlackCalculatorCpu(blackCalcStruct* blackCalc, payoffStruct payoff,
                                   float forwardPrice, float stdDev, float riskFreeDiscount)
{
    blackCalc->strike = payoff.strike;
    blackCalc->forward = forwardPrice;
    blackCalc->stdDev = stdDev;
    blackCalc->discount = riskFreeDiscount;
    blackCalc->variance = stdDev * stdDev;

    initBlackCalcVarsCpu(blackCalc, payoff);
}

static float getResultValCpu(blackCalcStruct blackCalculator)
{
    float result = blackCalculator.discount * (blackCalculator.forward *
                    blackCalculator.alpha + blackCalculator.x * blackCalculator.beta);
    return result;
}

// global function to retrieve the output value for an option
// (matches original getOutValOptionCpu signature)
static void getOutValOptionCpu(optionInputStruct* options, float* outputVals,
                               int optionNum, int numVals)
{
    if (optionNum < numVals) {
        optionInputStruct threadOption = options[optionNum];

        payoffStruct currPayoff;
        currPayoff.type = threadOption.type;
        currPayoff.strike = threadOption.strike;

        yieldTermStruct qTS;
        qTS.timeYearFraction = threadOption.t;
        qTS.forward = threadOption.q;

        yieldTermStruct rTS;
        rTS.timeYearFraction = threadOption.t;
        rTS.forward = threadOption.r;

        blackVolStruct volTS;
        volTS.timeYearFraction = threadOption.t;
        volTS.volatility = threadOption.vol;

        blackScholesMertStruct stochProcess;
        stochProcess.x0 = threadOption.spot;
        stochProcess.dividendTS = qTS;
        stochProcess.riskFreeTS = rTS;
        stochProcess.blackVolTS = volTS;

        optionStruct currOption;
        currOption.payoff = currPayoff;
        currOption.yearFractionTime = threadOption.t;
        currOption.pricingEngine = stochProcess;

        float variance = getBlackVolBlackVarCpu(currOption.pricingEngine.blackVolTS);
        float dividendDiscount = getDiscountOnDividendYieldCpu(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
        float riskFreeDiscount = getDiscountOnRiskFreeRateCpu(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
        float spot = currOption.pricingEngine.x0;

        float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

        blackCalcStruct blackCalc;
        initBlackCalculatorCpu(&blackCalc, currOption.payoff, forwardPrice, sqrt(variance), riskFreeDiscount);

        float resultVal = getResultValCpu(blackCalc);
        outputVals[optionNum] = resultVal;
    }
}

// ===== Public interface =====

void solution_init(int N,
                   const int* types, const float* strikes, const float* spots,
                   const float* qs, const float* rs, const float* ts,
                   const float* vols)
{
    g_N = N;
    g_types = types;
    g_strikes = strikes;
    g_spots = spots;
    g_qs = qs;
    g_rs = rs;
    g_ts = ts;
    g_vols = vols;
}

void solution_compute(int N, float* prices)
{
    // Build optionInputStruct array from flat arrays (bridge to original interface)
    optionInputStruct* options = (optionInputStruct*)malloc(N * sizeof(optionInputStruct));
    int numOption;
    for (numOption = 0; numOption < N; numOption++) {
        options[numOption].type   = g_types[numOption];
        options[numOption].strike = g_strikes[numOption];
        options[numOption].spot   = g_spots[numOption];
        options[numOption].q      = g_qs[numOption];
        options[numOption].r      = g_rs[numOption];
        options[numOption].t      = g_ts[numOption];
        options[numOption].vol    = g_vols[numOption];
        options[numOption].value  = 0.0f;
        options[numOption].tol    = 1.0e-4f;
    }

    // Run on CPU (matches original loop in blackScholesAnalyticEngine.c)
    for (numOption = 0; numOption < N; numOption++) {
        getOutValOptionCpu(options, prices, numOption, N);
    }

    free(options);
}

void solution_free(void)
{
    /* All data owned by task_io; nothing to free here. */
}
