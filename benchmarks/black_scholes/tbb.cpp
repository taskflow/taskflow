#include "common.hpp" 
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

struct mainWork {
  mainWork() {}
  mainWork(mainWork &w, tbb::split) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    float price;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i++) {
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
};

void bs_tbb(unsigned num_threads) {
	int j;
	tbb::affinity_partitioner a;
  tbb::task_scheduler_init init(num_threads);

	mainWork doall;
	for (j=0; j<NUM_RUNS; j++) {
		tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
	}
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  bs_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
