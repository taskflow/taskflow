#include "common.hpp"
#include <taskflow/taskflow.hpp> 


void bs_taskflow(unsigned num_threads) {
  tf::Taskflow flow;

  flow.parallel_for(
    0, numOptions, 1, [&](unsigned i) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */
       auto price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                    rate[i], volatility[i], otime[i], 
                                    otype[i], 0);
       prices[i] = price;
#ifdef ERR_CHK 
      check_error(i, price);
#endif
    }, num_threads 
  );

  tf::Executor(num_threads).run_n(flow, NUM_RUNS).wait();
}


std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  bs_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
