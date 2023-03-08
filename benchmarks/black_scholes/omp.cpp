#include "common.hpp"
#include <omp.h>

//int bs_omp(void *tid_ptr) {
//  printf("OMP version\n");
//
//  int i, j;
//  float price;
//  float priceDelta;
//  int tid = *(int *)tid_ptr;
//
//  int start = tid * (numOptions / nThreads);
//  int end = start + (numOptions / nThreads);
//
//  for (j=0; j<NUM_RUNS; j++) {
//    #pragma omp parallel for private(i, price, priceDelta)
//    for(i=0; i<numOptions; i++) {
//    //for (i=start; i<end; i++) {
//      /* Calling main function to calculate option value based on
//       * Black & Scholes's equation.
//       */
//      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
//          rate[i], volatility[i], otime[i],
//          otype[i], 0);
//      prices[i] = price;
//
//#ifdef ERR_CHK
//      priceDelta = data[i].DGrefval - price;
//      if( fabs(priceDelta) >= 1e-4 ){
//        printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
//            i, price, data[i].DGrefval, priceDelta);
//        numError ++;
//      }
//#endif
//    }
//  }
//
//  return 0;
//}



void bs_omp_parallel_for(unsigned num_threads) {
  omp_set_num_threads(num_threads);

  int i, j;
  float price;

  for (j=0; j<NUM_RUNS; j++) {
    #pragma omp parallel for private(i, price) schedule(static)
    for (i=0; i<numOptions; i++) {
      /* Calling main function to calculate option value based on
       * Black & Scholes's equation.
       */
      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
          rate[i], volatility[i], otime[i],
          otype[i], 0);
      prices[i] = price;

#ifdef ERR_CHK
      check_error(i, price);
#endif
    }
  }
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  bs_omp_parallel_for(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


