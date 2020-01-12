#include <stdlib.h>
#include <math.h>
#include <string>
#include <cassert>
#include <iostream>
#include <chrono>
#include <fstream>
#include <memory>

//Precision to use for calculations
#define FPTYPE float
#define NUM_RUNS 100


typedef struct OptionData_ {
  FPTYPE s;          // spot price
  FPTYPE strike;     // strike price
  FPTYPE r;          // risk-free interest rate
  FPTYPE divq;       // dividend rate
  FPTYPE v;          // volatility
  FPTYPE t;          // time to maturity or option expiration in years 
                     //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
  char OptionType;   // Option type.  "P"=PUT, "C"=CALL
  FPTYPE divs;       // dividend vals (not used in this test)
  FPTYPE DGrefval;   // DerivaGem Reference Value
} OptionData;

inline OptionData *data;
inline FPTYPE *prices;
inline int numOptions;

inline int*    otype;
inline FPTYPE* sptprice;
inline FPTYPE* strike;
inline FPTYPE* rate;
inline FPTYPE* volatility;
inline FPTYPE* otime;
inline int numError {0};

inline FPTYPE* BUFFER {nullptr};
inline int* BUFFER2 {nullptr};

////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244 
////////////////////////////////////////////////////////////////////////////////
#define inv_sqrt_2xPI 0.39894228040143270286

inline FPTYPE CNDF( FPTYPE InputX ) {
    int sign;

    FPTYPE OutputX;
    FPTYPE xInput;
    FPTYPE xNPrimeofX;
    FPTYPE expValues;
    FPTYPE xK2;
    FPTYPE xK2_2, xK2_3;
    FPTYPE xK2_4, xK2_5;
    FPTYPE xLocal, xLocal_1;
    FPTYPE xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
      InputX = -InputX;
      sign = 1;
    } 
    else {
      sign = 0;
    }

    xInput = InputX;
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = std::exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
    
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;
    
    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    
    return OutputX;
} 



inline FPTYPE BlkSchlsEqEuroNoDiv( 
  FPTYPE sptprice, FPTYPE strike, FPTYPE rate,
  FPTYPE volatility, FPTYPE time, int otype, float timet) {

  FPTYPE OptionPrice;

  // local private working variables for the calculation
  //FPTYPE xStockPrice;  // These two variables are not used 
  //FPTYPE xStrikePrice;
  FPTYPE xRiskFreeRate;
  FPTYPE xVolatility;
  FPTYPE xTime;
  FPTYPE xSqrtTime;

  FPTYPE logValues;
  FPTYPE xLogTerm;
  FPTYPE xD1; 
  FPTYPE xD2;
  FPTYPE xPowerTerm;
  FPTYPE xDen;
  FPTYPE d1;
  FPTYPE d2;
  FPTYPE FutureValueX;
  FPTYPE NofXd1;
  FPTYPE NofXd2;
  FPTYPE NegNofXd1;
  FPTYPE NegNofXd2;    

  //xStockPrice = sptprice;
  //xStrikePrice = strike;
  xRiskFreeRate = rate;
  xVolatility = volatility;

  xTime = time;
  xSqrtTime = std::sqrt(xTime);

  logValues = std::log(sptprice/strike);

  xLogTerm = logValues;

  xPowerTerm = xVolatility * xVolatility;
  xPowerTerm = xPowerTerm * 0.5;

  xD1 = xRiskFreeRate + xPowerTerm;
  xD1 = xD1 * xTime;
  xD1 = xD1 + xLogTerm;

  xDen = xVolatility * xSqrtTime;
  xD1 = xD1 / xDen;
  xD2 = xD1 -  xDen;

  d1 = xD1;
  d2 = xD2;

  NofXd1 = CNDF( d1 );
  NofXd2 = CNDF( d2 );

  FutureValueX = strike * ( std::exp( -(rate)*(time) ) );        
  if (otype == 0) {            
    OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
  } 
  else { 
    NegNofXd1 = (1.0 - NofXd1);
    NegNofXd2 = (1.0 - NofXd2);
    OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
  }

  return OptionPrice;
}


inline void check_error(unsigned i, FPTYPE price) {
  FPTYPE priceDelta = data[i].DGrefval - price;
  if(std::fabs(priceDelta) >= 1e-4 ){
    printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
        i, price, data[i].DGrefval, priceDelta);
    numError ++;
  }
}


// Sequential version
inline void bs_seq(FPTYPE *seq_prices) {

  int i, j;
  FPTYPE price;

  for (j=0; j<NUM_RUNS; j++) {
    for (i=0; i<numOptions; i++) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */
      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
          rate[i], volatility[i], otime[i], 
          otype[i], 0);
      seq_prices[i] = price;

#ifdef ERR_CHK 
      check_error(i, seq_prices[i]);
#endif
    }
  }
}


// Write prices to output file
inline void dump(const std::string& output_file) {
  std::ofstream ofs(output_file);

  if(!ofs) {
    std::cerr << "ERROR: Unable to open file "  << output_file << std::endl;
    return ;
  }

  ofs << numOptions << '\n';

  for(auto i=0; i<numOptions; i++) {
    ofs << prices[i] << '\n';
  }
}




// Read input option data from file
inline bool parse_options(const std::string& option_file) {
  FILE* file = fopen(option_file.data(), "r");
  if(file == NULL) {
    std::cerr << "ERROR: Unable to open file " << option_file << std::endl;
    return false;
  }

  if(fscanf(file, "%i", &numOptions) != 1) {
    std::cerr << "ERROR: Unable to read from file " << option_file << std::endl;
    fclose(file);
    return false;
  }

  // Allocate spaces for the option data
  data = static_cast<OptionData*>(malloc(numOptions*sizeof(OptionData)));
  prices = static_cast<FPTYPE*>(malloc(numOptions*sizeof(FPTYPE)));

  for (int i = 0; i < numOptions; ++ i) {
    int num = fscanf(file, "%f %f %f %f %f %f %c %f %f", 
                           &data[i].s, &data[i].strike, &data[i].r, 
                           &data[i].divq, &data[i].v, &data[i].t, 
                           &data[i].OptionType, &data[i].divs, &data[i].DGrefval);
    if(num != 9) {
      std::cerr << "ERROR: Unable to read from file " << option_file << std::endl;
      fclose(file);
      return false;
    }
  }

  const int PAD {256};
  const int LINESIZE {64};

  BUFFER = static_cast<FPTYPE *>(malloc(5 * numOptions * sizeof(FPTYPE) + PAD));
  sptprice = reinterpret_cast<FPTYPE *>(((unsigned long long)BUFFER + PAD) & ~(LINESIZE - 1));
  strike = sptprice + numOptions;
  rate = strike + numOptions;
  volatility = rate + numOptions;
  otime = volatility + numOptions;

  BUFFER2 = static_cast<int *>(malloc(numOptions * sizeof(FPTYPE) + PAD));
  //otype = (int *) (((unsigned long long)BUFFER2 + PAD) & ~(LINESIZE - 1));
  otype = reinterpret_cast<int *>(((unsigned long long)BUFFER2 + PAD) & ~(LINESIZE - 1));

  for(auto i=0; i<numOptions; i++) {
    otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
    sptprice[i]   = data[i].s;
    strike[i]     = data[i].strike;
    rate[i]       = data[i].r;
    volatility[i] = data[i].v;    
    otime[i]      = data[i].t;
  }

  fclose(file);
  return true;
  //std::cout << "Size of data: " << numOptions * (sizeof(OptionData) + sizeof(int)) << std::endl;
}

std::chrono::microseconds measure_time_taskflow(unsigned);
std::chrono::microseconds measure_time_tbb(unsigned);
std::chrono::microseconds measure_time_omp(unsigned);
