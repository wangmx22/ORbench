// cpu_reference.c -- bonds_pricing CPU baseline
//
// Faithful port of FinanceBench bondsKernelsCpu.c / bondsStructs.h.
// Pure C. NO C++, NO references, NO new, NO file I/O, NO main.
//
// C++ function overloads from the original are renamed:
//   closeCpu(x,y)   -> closeCpu(x,y)  (calls closeCpu2 with n=42)
//   closeCpu(x,y,n) -> closeCpu2(x,y,n)
//   interestRateCompoundFactorCpu(ir, t)       -> interestRateCompoundFactorCpu(ir, t)
//   interestRateCompoundFactorCpu(ir, d1,d2,dc) -> interestRateCompoundFactorCpuDate(ir, d1,d2,dc)

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef double dataType;

#define SIMPLE_INTEREST 0
#define COMPOUNDED_INTEREST 1
#define CONTINUOUS_INTEREST 2
#define SIMPLE_THEN_COMPOUNDED_INTEREST 3

#define ANNUAL_FREQ 1
#define SEMIANNUAL_FREQ 2

#define USE_EXACT_DAY 0
#define USE_SERIAL_NUMS 1

#define QL_EPSILON_GPU 0.000000000000000001f

#define COMPUTE_AMOUNT -1

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

#define ACCURACY 1.0e-8


typedef struct
{
	int month;
	int day;
	int year;
	int dateSerialNum;
} bondsDateStruct;


typedef struct
{
	bondsDateStruct startDate;
	bondsDateStruct maturityDate;
	float rate;
} bondStruct;


typedef struct
{
	dataType rate;
	dataType freq;
	int comp;
	int dayCounter;
} intRateStruct;


typedef struct
{
	dataType forward;
	dataType compounding;
	dataType frequency;
	intRateStruct intRate;
	bondsDateStruct refDate;
	bondsDateStruct calDate;
	int dayCounter;
} bondsYieldTermStruct;


typedef struct
{
	bondsDateStruct paymentDate;
	bondsDateStruct accrualStartDate;
	bondsDateStruct accrualEndDate;
	dataType amount;
} couponStruct;


typedef struct
{
	couponStruct* legs;
	intRateStruct intRate;
	int nominal;
	int dayCounter;
} cashFlowsStruct;


typedef struct
{
	dataType* dirtyPrice;
	dataType* accruedAmountCurrDate;
	dataType* cleanPrice;
	dataType* bondForwardVal;
} resultsStruct;


typedef struct
{
	bondsYieldTermStruct* discountCurve;
	bondsYieldTermStruct* repoCurve;
	bondsDateStruct* currDate;
	bondsDateStruct* maturityDate;
	dataType* bondCleanPrice;
	bondStruct* bond;
	dataType* dummyStrike;
} inArgsStruct;


typedef struct
{
	dataType npv;
	int dayCounter;
	int comp;
	dataType freq;
	bool includecurrDateFlows;
	bondsDateStruct currDate;
	bondsDateStruct npvDate;
} irrFinderStruct;


typedef struct
{
	dataType root_;
	dataType xMin_;
	dataType xMax_;
	dataType fxMin_;
	dataType fxMax_;
	int maxEvaluations_;
	int evaluationNumber_;
	dataType lowerBound_;
	dataType upperBound_;
	bool lowerBoundEnforced_;
	bool upperBoundEnforced_;
} solverStruct;


// ===== Module-level state for solution_init/solution_compute interface =====
static int g_N;
static const int* g_issue_year;
static const int* g_issue_month;
static const int* g_issue_day;
static const int* g_maturity_year;
static const int* g_maturity_month;
static const int* g_maturity_day;
static const float* g_rates;
static float g_coupon_freq;


// ===== Forward declarations =====
int monthLengthKernelCpu(int month, bool leapYear);
int monthOffsetKernelCpu(int m, bool leapYear);
int yearOffsetKernelCpu(int y);
bool isLeapKernelCpu(int y);
bondsDateStruct intializeDateKernelCpu(int d, int m, int y);
dataType yearFractionCpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
int dayCountCpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
dataType couponNotionalCpu(void);
dataType bondNotionalCpu(void);
dataType fixedRateCouponNominalCpu(void);
bool eventHasOccurredCpu(bondsDateStruct currDate, bondsDateStruct eventDate);
bool cashFlowHasOccurredCpu(bondsDateStruct refDate, bondsDateStruct eventDate);
bondsDateStruct advanceDateCpu(bondsDateStruct date, int numMonthsAdvance);
int getNumCashFlowsCpu(inArgsStruct inArgs, int bondNum);
void getBondsResultsCpu(inArgsStruct inArgs, resultsStruct results, int totNumRuns);
dataType getDirtyPriceCpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType getAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType discountingBondEngineCalculateSettlementValueCpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType bondAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType bondFunctionsAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType cashFlowsAccruedAmountCpu(cashFlowsStruct cashFlows, bool includecurrDateFlows, bondsDateStruct currDate, int numLegs, inArgsStruct inArgs, int bondNum);
dataType fixedRateCouponAccruedAmountCpu(cashFlowsStruct cashFlows, int numLeg, bondsDateStruct d, inArgsStruct inArgs, int bondNum);
dataType cashFlowsNpvCpu(cashFlowsStruct cashFlows, bondsYieldTermStruct discountCurve, bool includecurrDateFlows, bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);
dataType bondsYieldTermStructureDiscountCpu(bondsYieldTermStruct ytStruct, bondsDateStruct t);
dataType flatForwardDiscountImplCpu(intRateStruct intRate, dataType t);
dataType interestRateDiscountFactorCpu(intRateStruct intRate, dataType t);
dataType interestRateCompoundFactorCpu(intRateStruct intRate, dataType t);
dataType interestRateCompoundFactorCpuDate(intRateStruct intRate, bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
dataType fixedRateCouponAmountCpu(cashFlowsStruct cashFlows, int numLeg);
dataType cashFlowsNpvYieldCpu(cashFlowsStruct cashFlows, intRateStruct y, bool includecurrDateFlows, bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);
dataType fOpCpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);
dataType fDerivativeCpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);
bool closeCpu2(dataType x, dataType y, int n);
bool closeCpu(dataType x, dataType y);
dataType enforceBoundsCpu(dataType x);
dataType solveImplCpu(solverStruct solver, irrFinderStruct f, dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs);
dataType solverSolveCpu(solverStruct solver, irrFinderStruct f, dataType accuracy, dataType guess, dataType step, cashFlowsStruct cashFlows, int numLegs);
dataType modifiedDurationCpu(cashFlowsStruct cashFlows, intRateStruct y, bool includecurrDateFlows, bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs);
dataType getBondYieldCpu(dataType cleanPrice, int dc, int comp, dataType freq, bondsDateStruct settlement, dataType accuracy, int maxEvaluations, inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType getBondFunctionsYieldCpu(dataType cleanPrice, int dc, int comp, dataType freq, bondsDateStruct settlement, dataType accuracy, int maxEvaluations, inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs);
dataType getCashFlowsYieldCpu(cashFlowsStruct leg, dataType npv, int dayCounter, int compounding, dataType frequency, bool includecurrDateFlows, bondsDateStruct currDate, bondsDateStruct npvDate, int numLegs, dataType accuracy, int maxIterations, dataType guess);
couponStruct cashFlowsNextCashFlowCpu(cashFlowsStruct cashFlows, bondsDateStruct currDate, int numLegs);
int cashFlowsNextCashFlowNumCpu(cashFlowsStruct cashFlows, bondsDateStruct currDate, int numLegs);
dataType interestRateImpliedRateCpu(dataType compound, int comp, dataType freq, dataType t);
dataType getMarketRepoRateCpu(bondsDateStruct d, int comp, dataType freq, bondsDateStruct referenceDate, inArgsStruct inArgs, int bondNum);


// ===== Implementations (faithful to bondsKernelsCpu.c) =====

int monthLengthKernelCpu(int month, bool leapYear)
{
	int MonthLength[] = {
			31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
	};

	int MonthLeapLength[] = {
			31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
	};

	return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}


int monthOffsetKernelCpu(int m, bool leapYear)
{
	int MonthOffset[] = {
			0,  31,  59,  90, 120, 151,   // Jan - Jun
			181, 212, 243, 273, 304, 334,   // Jun - Dec
			365     // used in dayOfMonth to bracket day
	};

	int MonthLeapOffset[] = {
			0,  31,  60,  91, 121, 152,   // Jan - Jun
			182, 213, 244, 274, 305, 335,   // Jun - Dec
			366     // used in dayOfMonth to bracket day
		};

	return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}


int yearOffsetKernelCpu(int y)
{
        // the list of all December 31st in the preceding year
        // e.g. for 1901 yearOffset[1] is 366, that is, December 31 1900
        int YearOffset[] = {
            // 1900-1909
                0,  366,  731, 1096, 1461, 1827, 2192, 2557, 2922, 3288,
            // 1910-1919
             3653, 4018, 4383, 4749, 5114, 5479, 5844, 6210, 6575, 6940,
            // 1920-1929
             7305, 7671, 8036, 8401, 8766, 9132, 9497, 9862,10227,10593,
            // 1930-1939
            10958,11323,11688,12054,12419,12784,13149,13515,13880,14245,
            // 1940-1949
            14610,14976,15341,15706,16071,16437,16802,17167,17532,17898,
            // 1950-1959
            18263,18628,18993,19359,19724,20089,20454,20820,21185,21550,
            // 1960-1969
            21915,22281,22646,23011,23376,23742,24107,24472,24837,25203,
            // 1970-1979
            25568,25933,26298,26664,27029,27394,27759,28125,28490,28855,
            // 1980-1989
            29220,29586,29951,30316,30681,31047,31412,31777,32142,32508,
            // 1990-1999
            32873,33238,33603,33969,34334,34699,35064,35430,35795,36160,
            // 2000-2009
            36525,36891,37256,37621,37986,38352,38717,39082,39447,39813,
            // 2010-2019
            40178,40543,40908,41274,41639,42004,42369,42735,43100,43465,
            // 2020-2029
            43830,44196,44561,44926,45291,45657,46022,46387,46752,47118,
            // 2030-2039
            47483,47848,48213,48579,48944,49309,49674,50040,50405,50770,
            // 2040-2049
            51135,51501,51866,52231,52596,52962,53327,53692,54057,54423,
            // 2050-2059
            54788,55153,55518,55884,56249,56614,56979,57345,57710,58075,
            // 2060-2069
            58440,58806,59171,59536,59901,60267,60632,60997,61362,61728,
            // 2070-2079
            62093,62458,62823,63189,63554,63919,64284,64650,65015,65380,
            // 2080-2089
            65745,66111,66476,66841,67206,67572,67937,68302,68667,69033,
            // 2090-2099
            69398,69763,70128,70494,70859,71224,71589,71955,72320,72685,
            // 2100-2109
            73050,73415,73780,74145,74510,74876,75241,75606,75971,76337,
            // 2110-2119
            76702,77067,77432,77798,78163,78528,78893,79259,79624,79989,
            // 2120-2129
            80354,80720,81085,81450,81815,82181,82546,82911,83276,83642,
            // 2130-2139
            84007,84372,84737,85103,85468,85833,86198,86564,86929,87294,
            // 2140-2149
            87659,88025,88390,88755,89120,89486,89851,90216,90581,90947,
            // 2150-2159
            91312,91677,92042,92408,92773,93138,93503,93869,94234,94599,
            // 2160-2169
            94964,95330,95695,96060,96425,96791,97156,97521,97886,98252,
            // 2170-2179
            98617,98982,99347,99713,100078,100443,100808,101174,101539,101904,
            // 2180-2189
            102269,102635,103000,103365,103730,104096,104461,104826,105191,105557,
            // 2190-2199
            105922,106287,106652,107018,107383,107748,108113,108479,108844,109209,
            // 2200
            109574
        };

        return YearOffset[y-1900];
}


bool isLeapKernelCpu(int y)
{
        bool YearIsLeap[] = {
            // 1900 is leap in agreement with Excel's bug
            // 1900 is out of valid date range anyway
            // 1900-1909
             true,false,false,false, true,false,false,false, true,false,
            // 1910-1919
            false,false, true,false,false,false, true,false,false,false,
            // 1920-1929
             true,false,false,false, true,false,false,false, true,false,
            // 1930-1939
            false,false, true,false,false,false, true,false,false,false,
            // 1940-1949
             true,false,false,false, true,false,false,false, true,false,
            // 1950-1959
            false,false, true,false,false,false, true,false,false,false,
            // 1960-1969
             true,false,false,false, true,false,false,false, true,false,
            // 1970-1979
            false,false, true,false,false,false, true,false,false,false,
            // 1980-1989
             true,false,false,false, true,false,false,false, true,false,
            // 1990-1999
            false,false, true,false,false,false, true,false,false,false,
            // 2000-2009
             true,false,false,false, true,false,false,false, true,false,
            // 2010-2019
            false,false, true,false,false,false, true,false,false,false,
            // 2020-2029
             true,false,false,false, true,false,false,false, true,false,
            // 2030-2039
            false,false, true,false,false,false, true,false,false,false,
            // 2040-2049
             true,false,false,false, true,false,false,false, true,false,
            // 2050-2059
            false,false, true,false,false,false, true,false,false,false,
            // 2060-2069
             true,false,false,false, true,false,false,false, true,false,
            // 2070-2079
            false,false, true,false,false,false, true,false,false,false,
            // 2080-2089
             true,false,false,false, true,false,false,false, true,false,
            // 2090-2099
            false,false, true,false,false,false, true,false,false,false,
            // 2100-2109
            false,false,false,false, true,false,false,false, true,false,
            // 2110-2119
            false,false, true,false,false,false, true,false,false,false,
            // 2120-2129
             true,false,false,false, true,false,false,false, true,false,
            // 2130-2139
            false,false, true,false,false,false, true,false,false,false,
            // 2140-2149
             true,false,false,false, true,false,false,false, true,false,
            // 2150-2159
            false,false, true,false,false,false, true,false,false,false,
            // 2160-2169
             true,false,false,false, true,false,false,false, true,false,
            // 2170-2179
            false,false, true,false,false,false, true,false,false,false,
            // 2180-2189
             true,false,false,false, true,false,false,false, true,false,
            // 2190-2199
            false,false, true,false,false,false, true,false,false,false,
            // 2200
            false
        };

	return YearIsLeap[y-1900];
}


bondsDateStruct intializeDateKernelCpu(int d, int m, int y)
{
	bondsDateStruct currDate;

	currDate.day = d;
	currDate.month = m;
	currDate.year = y;

	bool leap = isLeapKernelCpu(y);
	int offset = monthOffsetKernelCpu(m,leap);

	currDate.dateSerialNum = d + offset + yearOffsetKernelCpu(y);

	return currDate;
}


dataType yearFractionCpu(bondsDateStruct d1,
					bondsDateStruct d2, int dayCounter)
{
	return dayCountCpu(d1, d2, dayCounter) / 360.0;
}


int dayCountCpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter)
{
	if (dayCounter == USE_EXACT_DAY)
	{
		int dd1 = d1.day, dd2 = d2.day;
		int mm1 = d1.month, mm2 = d2.month;
		int yy1 = d1.year, yy2 = d2.year;

		if (dd2 == 31 && dd1 < 30)
		{
			dd2 = 1; mm2++;
		}

		return 360*(yy2-yy1) + 30*(mm2-mm1-1) + MAX(0, 30-dd1) + MIN(30, dd2);
	}

	else
	{
		return (d2.dateSerialNum - d1.dateSerialNum);
	}
}


dataType couponNotionalCpu(void)
{
	return 100.0;
}

dataType bondNotionalCpu(void)
{
	return 100.0;
}


dataType fixedRateCouponNominalCpu(void)
{
	return 100.0;
}

bool eventHasOccurredCpu(bondsDateStruct currDate, bondsDateStruct eventDate)
{
         return eventDate.dateSerialNum > currDate.dateSerialNum;
}


bool cashFlowHasOccurredCpu(bondsDateStruct refDate, bondsDateStruct eventDate)
{
        return eventHasOccurredCpu(refDate, eventDate);
}


bondsDateStruct advanceDateCpu(bondsDateStruct date, int numMonthsAdvance)
{
	int d = date.day;
	int m = date.month+numMonthsAdvance;
	int y = date.year;

	while (m > 12)
	{
		m -= 12;
		y += 1;
	}

	while (m < 1)
	{
		m += 12;
		y -= 1;
	}

	int length = monthLengthKernelCpu(m, isLeapKernelCpu(y));
	if (d > length)
		d = length;

	bondsDateStruct newDate = intializeDateKernelCpu(d, m, y);

	return newDate;
}

int getNumCashFlowsCpu(inArgsStruct inArgs, int bondNum)
{
	int numCashFlows = 0;

	//bondsDateStruct endDate = inArgs.bond[bondNum].maturityDate;
	bondsDateStruct currCashflowDate = inArgs.bond[bondNum].maturityDate;

	while (currCashflowDate.dateSerialNum > inArgs.bond[bondNum].startDate.dateSerialNum)
	{
		numCashFlows++;
		currCashflowDate = advanceDateCpu(currCashflowDate, -6);
	}

	return numCashFlows+1;
}


dataType interestRateCompoundFactorCpu(intRateStruct intRate, dataType t)
{
	switch (intRate.comp)
	{
          case SIMPLE_INTEREST:
            return 1.0 + intRate.rate*t;
          case COMPOUNDED_INTEREST:
            return pow(1.0f+intRate.rate/intRate.freq, intRate.freq*t);
          case CONTINUOUS_INTEREST:
            return exp(intRate.rate*t);
          //case SimpleThenCompounded:
          //  if (t<=1.0/Real(freq_))
          //      return 1.0 + intRate.rate*t;
          //  else
          //      return pow(1.0+r_/freq_, freq_*t);
	}

	return 0.0f;
}


// Overloaded version in original C++ code: interestRateCompoundFactorCpu(intRate, d1, d2, dayCounter)
// Renamed to interestRateCompoundFactorCpuDate for pure C
dataType interestRateCompoundFactorCpuDate(intRateStruct intRate, bondsDateStruct d1,
				                           bondsDateStruct d2, int dayCounter)
{
	dataType t = yearFractionCpu(d1, d2, dayCounter);
	return interestRateCompoundFactorCpu(intRate, t);
}


dataType interestRateDiscountFactorCpu(intRateStruct intRate, dataType t)
{
	return 1.0/interestRateCompoundFactorCpu(intRate, t);
}


dataType flatForwardDiscountImplCpu(intRateStruct intRate, dataType t)
{
	return interestRateDiscountFactorCpu(intRate, t);
}


dataType bondsYieldTermStructureDiscountCpu(bondsYieldTermStruct ytStruct, bondsDateStruct t)
{
	ytStruct.intRate.rate = ytStruct.forward;
	ytStruct.intRate.freq = ytStruct.frequency;
	ytStruct.intRate.comp = ytStruct.compounding;
	return flatForwardDiscountImplCpu(ytStruct.intRate, yearFractionCpu(ytStruct.refDate, t, ytStruct.dayCounter));
}


dataType fixedRateCouponAmountCpu(cashFlowsStruct cashFlows, int numLeg)
{
	if (cashFlows.legs[numLeg].amount == COMPUTE_AMOUNT)
	{
		return fixedRateCouponNominalCpu()*(interestRateCompoundFactorCpuDate(cashFlows.intRate, cashFlows.legs[numLeg].accrualStartDate,
                                              cashFlows.legs[numLeg].accrualEndDate, cashFlows.dayCounter) - 1.0);
	}
	else
	{
		return cashFlows.legs[numLeg].amount;
	}
}


dataType cashFlowsNpvCpu(cashFlowsStruct cashFlows,
                        bondsYieldTermStruct discountCurve,
                        bool includecurrDateFlows,
                        bondsDateStruct currDate,
                        bondsDateStruct npvDate,
						int numLegs)
{
       npvDate = currDate;

        dataType totalNPV = 0.0;

        for (int i=0; i<numLegs; ++i) {



            if (!(cashFlowHasOccurredCpu(cashFlows.legs[i].paymentDate, currDate)))
                totalNPV += fixedRateCouponAmountCpu(cashFlows, i) *
                            bondsYieldTermStructureDiscountCpu(discountCurve, cashFlows.legs[i].paymentDate);
        }

	return totalNPV/bondsYieldTermStructureDiscountCpu(discountCurve, npvDate);
}


dataType cashFlowsNpvYieldCpu(cashFlowsStruct cashFlows,
                        intRateStruct y,
                        bool includecurrDateFlows,
                        bondsDateStruct currDate,
                        bondsDateStruct npvDate,
						int numLegs)
{

        dataType npv = 0.0;
        dataType discount = 1.0;
        bondsDateStruct lastDate;
	bool first = true;

        for (int i=0; i<numLegs; ++i)
	{


        	if (cashFlowHasOccurredCpu(cashFlows.legs[i].paymentDate, currDate))
        		continue;

            	bondsDateStruct couponDate = cashFlows.legs[i].paymentDate;
            	dataType amount = fixedRateCouponAmountCpu(cashFlows, i);
            	if (first)
		{
			first = false;

			if (i > 0)
			{
                    		lastDate = advanceDateCpu(cashFlows.legs[i].paymentDate, -1*6);
                	}
			else
			{
                        	lastDate = cashFlows.legs[i].accrualStartDate;
			}
                	discount *= interestRateDiscountFactorCpu(y, yearFractionCpu(npvDate, couponDate, y.dayCounter));

            	}
		else
		{
                	discount *= interestRateDiscountFactorCpu(y, yearFractionCpu(lastDate, couponDate, y.dayCounter));
            	}

            	lastDate = couponDate;

            	npv += amount * discount;
        }

        return npv;
}


dataType fOpCpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
	intRateStruct yield;

	yield.rate = y;
	yield.comp = f.comp;
	yield.freq = f.freq;
	yield.dayCounter = f.dayCounter;

	dataType NPV = cashFlowsNpvYieldCpu(cashFlows,
                        yield,
                        false,
                        f.currDate,
                        f.npvDate, numLegs);

	return (f.npv - NPV);
}



dataType fDerivativeCpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
	intRateStruct yield;
	yield.rate = y;
	yield.dayCounter = f.dayCounter;
	yield.comp = f.comp;
	yield.freq = f.freq;

    	return modifiedDurationCpu(cashFlows, yield,
                                        f.includecurrDateFlows,
                                        f.currDate, f.npvDate, numLegs);
}


// Overloaded closeCpu(x,y,n) renamed to closeCpu2 for pure C
bool closeCpu2(dataType x, dataType y, int n)
{
	dataType diff = fabs(x-y);
	dataType tolerance = n*QL_EPSILON_GPU;

	return diff <= tolerance*fabs(x) &&
               diff <= tolerance*fabs(y);
}


bool closeCpu(dataType x, dataType y)
{
	return closeCpu2(x,y,42);
}


dataType enforceBoundsCpu(dataType x)
{
	/*if (lowerBoundEnforced_ && x < lowerBound_)
		return lowerBound_;
	if (upperBoundEnforced_ && x > upperBound_)
		return upperBound_;*/
	return x;
}


dataType solveImplCpu(solverStruct solver, irrFinderStruct f,
				dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs)
{
	dataType froot, dfroot, dx, dxold;
	dataType xh, xl;

	// Orient the search so that f(xl) < 0
	if (solver.fxMin_ < 0.0)
	{
		xl = solver.xMin_;
		xh = solver.xMax_;
	}
	else
	{
		xh = solver.xMin_;
		xl = solver.xMax_;
	}

	// the "stepsize before last"
	dxold = solver.xMax_ - solver.xMin_;
	// it was dxold=std::fabs(xMax_-xMin_); in Numerical Recipes
	// here (xMax_-xMin_ > 0) is verified in the constructor

	// and the last step
	dx = dxold;

	froot = fOpCpu(f, solver.root_, cashFlows, numLegs);
	dfroot = fDerivativeCpu(f, solver.root_, cashFlows, numLegs);

	++solver.evaluationNumber_;

	while (solver.evaluationNumber_<=solver.maxEvaluations_)
	{
		// Bisect if (out of range || not decreasing fast enough)
		if ((((solver.root_-xh)*dfroot-froot)*
			((solver.root_-xl)*dfroot-froot) > 0.0)
			|| (fabs(2.0*froot) > fabs(dxold*dfroot)))
		{
			dxold = dx;
			dx = (xh-xl)/2.0;
			solver.root_=xl+dx;
		}
		else
		{
			dxold = dx;
			dx = froot/dfroot;
			solver.root_ -= dx;
		}

		// Convergence criterion
		if (fabs(dx) < xAccuracy)
			return solver.root_;
		froot = fOpCpu(f, solver.root_, cashFlows, numLegs);
		dfroot = fDerivativeCpu(f, solver.root_, cashFlows, numLegs);
		++solver.evaluationNumber_;
		if (froot < 0.0)
			xl=solver.root_;
		else
			xh=solver.root_;
	}

	return solver.root_;
}


dataType solverSolveCpu(solverStruct solver,
					irrFinderStruct f,
					dataType accuracy,
					dataType guess,
					dataType step,
					cashFlowsStruct cashFlows,
					int numLegs)
{
	// check whether we really want to use epsilon
	accuracy = MAX(accuracy, QL_EPSILON_GPU);

	dataType growthFactor = 1.6;
	int flipflop = -1;

	solver.root_ = guess;
	solver.fxMax_ = fOpCpu(f, solver.root_, cashFlows, numLegs);

	// monotonically crescent bias, as in optionValue(volatility)
	if (closeCpu(solver.fxMax_,0.0))
	{
		return solver.root_;
	}
	else if (closeCpu(solver.fxMax_, 0.0))
	{
		solver.xMin_ = /*enforceBounds*/(solver.root_ - step);
		solver.fxMin_ = fOpCpu(f, solver.xMin_, cashFlows, numLegs);
		solver.xMax_ = solver.root_;
	}
	else
	{
		solver.xMin_ = solver.root_;
		solver.fxMin_ = solver.fxMax_;
		solver.xMax_ = /*enforceBounds*/(solver.root_+step);
		solver.fxMax_ = fOpCpu(f, solver.xMax_, cashFlows, numLegs);
	}

	solver.evaluationNumber_ = 2;
	while (solver.evaluationNumber_ <= solver.maxEvaluations_)
	{
		if (solver.fxMin_*solver.fxMax_ <= 0.0)
		{
			if (closeCpu(solver.fxMin_, 0.0))
				return solver.xMin_;
			if (closeCpu(solver.fxMax_, 0.0))
				return solver.xMax_;
			solver.root_ = (solver.xMax_+solver.xMin_)/2.0;
			return solveImplCpu(solver, f, accuracy, cashFlows, numLegs);
		}
		if (fabs(solver.fxMin_) < fabs(solver.fxMax_))
		{
			solver.xMin_ = /*enforceBounds*/(solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
			solver.fxMin_= fOpCpu(f, solver.xMin_, cashFlows, numLegs);
		}
		else if (fabs(solver.fxMin_) > fabs(solver.fxMax_))
		{
			solver.xMax_ = /*enforceBounds*/(solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
			solver.fxMax_= fOpCpu(f, solver.xMax_, cashFlows, numLegs);
		}
		else if (flipflop == -1)
		{
			solver.xMin_ = /*enforceBounds*/(solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
			solver.fxMin_= fOpCpu(f, solver.xMin_, cashFlows, numLegs);
			solver.evaluationNumber_++;
			flipflop = 1;
		}
		else if (flipflop == 1)
		{
			solver.xMax_ = /*enforceBounds*/(solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
			solver.fxMax_= fOpCpu(f, solver.xMax_, cashFlows, numLegs);
			flipflop = -1;
		}
		solver.evaluationNumber_++;
	}

	return 0.0f;
}


dataType modifiedDurationCpu(cashFlowsStruct cashFlows,
								intRateStruct y,
								bool includecurrDateFlows,
								bondsDateStruct currDate,
								bondsDateStruct npvDate,
								int numLegs)
{
	dataType P = 0.0;
	dataType dPdy = 0.0;
	dataType r = y.rate;
	dataType N = y.freq;
	int dc = y.dayCounter;


	for (int i=0; i<numLegs; ++i)
	{


		if (!cashFlowHasOccurredCpu(cashFlows.legs[i].paymentDate, currDate))
		{
			dataType t = yearFractionCpu(npvDate, cashFlows.legs[i].paymentDate, dc);
			dataType c = fixedRateCouponAmountCpu(cashFlows, i);
			dataType B = interestRateDiscountFactorCpu(y, t);

			P += c * B;
			switch (y.comp)
			{
				case SIMPLE_INTEREST:
					dPdy -= c * B*B * t;
					break;
				case COMPOUNDED_INTEREST:
					dPdy -= c * t * B/(1+r/N);
					break;
				case CONTINUOUS_INTEREST:
					dPdy -= c * B * t;
					break;
				case SIMPLE_THEN_COMPOUNDED_INTEREST:
					if (t<=1.0/N)
						dPdy -= c * B*B * t;
					else
						dPdy -= c * t * B/(1+r/N);
					break;
			}
		}
	}

	if (P == 0.0) // no cashflows
	{
		return 0.0;
	}
	return (-1*dPdy)/P; // reverse derivative sign
}


couponStruct cashFlowsNextCashFlowCpu(cashFlowsStruct cashFlows,
                            bondsDateStruct currDate,
							int numLegs)
{
        for (int i = 0; i < numLegs; ++i)
	{


            	if ( ! (cashFlowHasOccurredCpu(cashFlows.legs[i].paymentDate, currDate) ))
                	return cashFlows.legs[i];
        }
        return cashFlows.legs[numLegs-1];
}


int cashFlowsNextCashFlowNumCpu(cashFlowsStruct cashFlows,
                            bondsDateStruct currDate,
							int numLegs)
{
        for (int i = 0; i < numLegs; ++i)
	{

        	if ( ! (cashFlowHasOccurredCpu(cashFlows.legs[i].paymentDate, currDate) ))
                	return i;
        }

	return (numLegs-1);
}


dataType fixedRateCouponAccruedAmountCpu(cashFlowsStruct cashFlows, int numLeg, bondsDateStruct d, inArgsStruct inArgs, int bondNum)
{
	if (d.dateSerialNum <= cashFlows.legs[numLeg].accrualStartDate.dateSerialNum || d.dateSerialNum > inArgs.maturityDate[bondNum].dateSerialNum)
	{
		return 0.0;
	}
	else
	{
		bondsDateStruct endDate = cashFlows.legs[numLeg].accrualEndDate;
		if (d.dateSerialNum < cashFlows.legs[numLeg].accrualEndDate.dateSerialNum)
		{
			endDate = d;
		}

		return fixedRateCouponNominalCpu()*(interestRateCompoundFactorCpuDate(cashFlows.intRate, cashFlows.legs[numLeg].accrualStartDate, endDate, cashFlows.dayCounter) - 1.0);
	}
}


dataType cashFlowsAccruedAmountCpu(cashFlowsStruct cashFlows,
                                  bool includecurrDateFlows,
                                  bondsDateStruct currDate,
								  int numLegs, inArgsStruct inArgs, int bondNum)
{
	int legComputeNum = cashFlowsNextCashFlowNumCpu(cashFlows,
                            				currDate,
							numLegs);

        dataType result = 0.0;

        for (int i = legComputeNum; i < (numLegs); ++i)
	{
                result += fixedRateCouponAccruedAmountCpu(cashFlows, i, currDate, inArgs, bondNum);
        }

        return result;
}


dataType bondFunctionsAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
        return cashFlowsAccruedAmountCpu(cashFlows,
                                        false, date, numLegs, inArgs, bondNum) *
            100.0 / bondNotionalCpu();
}


dataType bondAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalCpu();
	if (currentNotional == 0.0)
		return 0.0;

	return bondFunctionsAccruedAmountCpu(inArgs, date, bondNum, cashFlows, numLegs);
}


dataType getAccruedAmountCpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	return bondAccruedAmountCpu(inArgs, date, bondNum, cashFlows, numLegs);
}


dataType discountingBondEngineCalculateSettlementValueCpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	bondsDateStruct currDate = inArgs.currDate[bondNum];

	if (currDate.dateSerialNum < inArgs.bond[bondNum].startDate.dateSerialNum)
	{
		currDate = inArgs.bond[bondNum].startDate;
	}

        // a bond's cashflow on settlement date is never taken into account
        return cashFlowsNpvCpu(cashFlows,
                               inArgs.discountCurve[bondNum],
                               false,
                               currDate,
                               currDate,
			       numLegs);
}


dataType getDirtyPriceCpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalCpu();
	return discountingBondEngineCalculateSettlementValueCpu(inArgs, bondNum, cashFlows, numLegs) * 100.0 / currentNotional;
}


dataType getCashFlowsYieldCpu(cashFlowsStruct leg,
                          dataType npv,
                          int dayCounter,
                          int compounding,
                          dataType frequency,
                          bool includecurrDateFlows,
                          bondsDateStruct currDate,
                          bondsDateStruct npvDate,
						  int numLegs,
                          dataType accuracy,
                          int maxIterations,
                          dataType guess)
{
	//Brent solver;
	solverStruct solver;
	solver.maxEvaluations_ = maxIterations;
	irrFinderStruct objFunction;

	objFunction.npv = npv;
	objFunction.dayCounter = dayCounter;
	objFunction.comp = compounding;
	objFunction.freq = frequency;
	objFunction.includecurrDateFlows = includecurrDateFlows;
	objFunction.currDate = currDate;
	objFunction.npvDate = npvDate;

	return solverSolveCpu(solver, objFunction, accuracy, guess, guess/10.0, leg, numLegs);
}


dataType getBondFunctionsYieldCpu(dataType cleanPrice,
                     int dc,
                     int comp,
                     dataType freq,
                     bondsDateStruct settlement,
                     dataType accuracy,
                     int maxEvaluations,
					 inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
        dataType dirtyPrice = cleanPrice + bondFunctionsAccruedAmountCpu(currInArgs, settlement, bondNum, cashFlows, numLegs);
        dirtyPrice /= 100.0 / bondNotionalCpu();

        return getCashFlowsYieldCpu(cashFlows, dirtyPrice,
                                dc, comp, freq,
                                false, settlement, settlement, numLegs,
                                accuracy, maxEvaluations, 0.05f);
}


dataType getBondYieldCpu(dataType cleanPrice,
                     int dc,
                     int comp,
                     dataType freq,
                     bondsDateStruct settlement,
                     dataType accuracy,
                     int maxEvaluations,
		     inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalCpu();

	if (currentNotional == 0.0)
		return 0.0;

	if (currInArgs.bond[bondNum].startDate.dateSerialNum > settlement.dateSerialNum)
	{
		settlement = currInArgs.bond[bondNum].startDate;
	}

	return getBondFunctionsYieldCpu(cleanPrice, dc, comp, freq,
								settlement, accuracy, maxEvaluations,
								currInArgs, bondNum, cashFlows, numLegs);
}


dataType interestRateImpliedRateCpu(dataType compound,
                              int comp,
                              dataType freq,
                              dataType t)
{
        dataType r = 0.0f;
        if (compound==1.0)
	{
            r = 0.0;
        }
	else
	{
            switch (comp)
	    {
              case SIMPLE_INTEREST:
                r = (compound - 1.0)/t;
                break;
              case COMPOUNDED_INTEREST:
                r = (pow((dataType)compound, 1.0f/((freq)*t))-1.0f)*(freq);
                break;
	    }
        }

        return r;
}


dataType getMarketRepoRateCpu(bondsDateStruct d,
                                   int comp,
                                   dataType freq,
								   bondsDateStruct referenceDate,
								   inArgsStruct inArgs, int bondNum)
{
	dataType compound = 1.0/bondsYieldTermStructureDiscountCpu(inArgs.repoCurve[bondNum], d);
    return interestRateImpliedRateCpu(compound,
                                   comp, freq,
                                   yearFractionCpu(referenceDate, d, inArgs.repoCurve[bondNum].dayCounter));
}


void getBondsResultsCpu(inArgsStruct inArgs, resultsStruct results, int totNumRuns)
{
	for (int bondNum=0; bondNum < totNumRuns; bondNum++)
	{
		int numLegs  = getNumCashFlowsCpu(inArgs, bondNum);
		cashFlowsStruct cashFlows;
		cashFlows.legs = (couponStruct*)malloc(numLegs*sizeof(couponStruct));

		cashFlows.intRate.dayCounter = USE_EXACT_DAY;
		cashFlows.intRate.rate = inArgs.bond[bondNum].rate;
		cashFlows.intRate.freq = ANNUAL_FREQ;
		cashFlows.intRate.comp = SIMPLE_INTEREST;
		cashFlows.dayCounter = USE_EXACT_DAY;
		cashFlows.nominal = 100.0;

		//bondsDateStruct currPaymentDate;
		bondsDateStruct currStartDate = advanceDateCpu(inArgs.bond[bondNum].maturityDate, (numLegs - 1)*-6);;
		bondsDateStruct currEndDate = advanceDateCpu(currStartDate, 6);

		for (int cashFlowNum = 0; cashFlowNum < numLegs-1; cashFlowNum++)
		{
			cashFlows.legs[cashFlowNum].paymentDate = currEndDate;


			cashFlows.legs[cashFlowNum].accrualStartDate = currStartDate;
			cashFlows.legs[cashFlowNum].accrualEndDate = currEndDate;

			cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

			currStartDate = currEndDate;
			currEndDate = advanceDateCpu(currEndDate, 6);
		}

		cashFlows.legs[numLegs-1].paymentDate = inArgs.bond[bondNum].maturityDate;
		cashFlows.legs[numLegs-1].accrualStartDate = inArgs.currDate[bondNum];
		cashFlows.legs[numLegs-1].accrualEndDate = inArgs.currDate[bondNum];
		cashFlows.legs[numLegs-1].amount = 100.0;


		results.bondForwardVal[bondNum] = getBondYieldCpu(inArgs.bondCleanPrice[bondNum],
                     USE_EXACT_DAY,
                     COMPOUNDED_INTEREST,
                     2.0,
                     inArgs.currDate[bondNum],
                     ACCURACY,
                     100,
		     inArgs, bondNum, cashFlows, numLegs);
		inArgs.discountCurve[bondNum].forward = results.bondForwardVal[bondNum];
		results.dirtyPrice[bondNum] = getDirtyPriceCpu(inArgs, bondNum, cashFlows, numLegs);
		results.accruedAmountCurrDate[bondNum] = getAccruedAmountCpu(inArgs, inArgs.currDate[bondNum], bondNum, cashFlows, numLegs);
		results.cleanPrice[bondNum] = results.dirtyPrice[bondNum] - results.accruedAmountCurrDate[bondNum];


		free(cashFlows.legs);
	}
}


// ===== Public interface (solution_init / solution_compute / solution_free) =====

void solution_init(int N,
                   const int* issue_year, const int* issue_month, const int* issue_day,
                   const int* maturity_year, const int* maturity_month, const int* maturity_day,
                   const float* rates, float coupon_freq) {
    g_N = N;
    g_issue_year = issue_year;
    g_issue_month = issue_month;
    g_issue_day = issue_day;
    g_maturity_year = maturity_year;
    g_maturity_month = maturity_month;
    g_maturity_day = maturity_day;
    g_rates = rates;
    g_coupon_freq = coupon_freq;
}

void solution_compute(int N, float* prices) {
    // Build inArgsStruct from flat arrays, matching bondsEngine.c setup
    inArgsStruct inArgsHost;

    inArgsHost.discountCurve = (bondsYieldTermStruct*)malloc(N*sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct*)malloc(N*sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct*)malloc(N*sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct*)malloc(N*sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType*)malloc(N*sizeof(dataType));
    inArgsHost.bond = (bondStruct*)malloc(N*sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType*)malloc(N*sizeof(dataType));

    for (int numBond = 0; numBond < N; numBond++)
    {
        dataType repoRate = 0.07;
        int repoCompounding = SIMPLE_INTEREST;
        dataType repoCompoundFreq = 1;

        bondsDateStruct bondIssueDate = intializeDateKernelCpu(g_issue_day[numBond], g_issue_month[numBond], g_issue_year[numBond]);
        bondsDateStruct bondMaturityDate = intializeDateKernelCpu(g_maturity_day[numBond], g_maturity_month[numBond], g_maturity_year[numBond]);

        bondsDateStruct todaysDate = intializeDateKernelCpu(bondMaturityDate.day-1, bondMaturityDate.month, bondMaturityDate.year);

        bondStruct bond;
        bond.startDate = bondIssueDate;
        bond.maturityDate = bondMaturityDate;
        bond.rate = g_rates[numBond];

        dataType bondCouponFrequency = (dataType)g_coupon_freq;

        dataType bondCleanPrice = 89.97693786;

        bondsYieldTermStruct bondCurve;

        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.forward = -0.1f;  // dummy rate
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;
        bondCurve.dayCounter = USE_EXACT_DAY;

        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;

        dataType dummyStrike = 91.5745;

        bondsYieldTermStruct repoCurve;
        repoCurve.refDate = todaysDate;
        repoCurve.calDate = todaysDate;
        repoCurve.forward = repoRate;
        repoCurve.compounding = repoCompounding;
        repoCurve.frequency = repoCompoundFreq;
        repoCurve.dayCounter = USE_SERIAL_NUMS;

        inArgsHost.discountCurve[numBond] = bondCurve;
        inArgsHost.repoCurve[numBond] = repoCurve;
        inArgsHost.currDate[numBond] = todaysDate;
        inArgsHost.maturityDate[numBond] = bondMaturityDate;
        inArgsHost.bondCleanPrice[numBond] = bondCleanPrice;
        inArgsHost.bond[numBond] = bond;
        inArgsHost.dummyStrike[numBond] = dummyStrike;
    }

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType*)malloc(N*sizeof(dataType));
    resultsHost.accruedAmountCurrDate = (dataType*)malloc(N*sizeof(dataType));
    resultsHost.cleanPrice = (dataType*)malloc(N*sizeof(dataType));
    resultsHost.bondForwardVal = (dataType*)malloc(N*sizeof(dataType));

    getBondsResultsCpu(inArgsHost, resultsHost, N);

    // Extract results into output float array (4 floats per bond)
    for (int i = 0; i < N; i++) {
        prices[i * 4 + 0] = (float)resultsHost.dirtyPrice[i];
        prices[i * 4 + 1] = (float)resultsHost.accruedAmountCurrDate[i];
        prices[i * 4 + 2] = (float)resultsHost.cleanPrice[i];
        prices[i * 4 + 3] = (float)resultsHost.bondForwardVal[i];
    }

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountCurrDate);
    free(resultsHost.cleanPrice);
    free(resultsHost.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

void solution_free(void) {
    /* No persistent allocations */
}
