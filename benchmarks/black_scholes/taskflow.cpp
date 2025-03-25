#include "common.hpp"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

void bs_taskflow(unsigned num_threads) {

  static tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  auto init = taskflow.placeholder();

  auto loop = taskflow.for_each_index(
    0, numOptions, 1, [&](int i) {
    auto price = BlkSchlsEqEuroNoDiv(
      sptprice[i], strike[i],
      rate[i], volatility[i], otime[i],
      otype[i], 0
    );

    prices[i] = price;
#ifdef ERR_CHK
    check_error(i, price);
#endif
  }, tf::StaticPartitioner());

  auto cond = taskflow.emplace([i=0] () mutable{
    return ++i == NUM_RUNS ? -1 : 0;
  });

  init.precede(loop);
  loop.precede(cond);
  cond.precede(loop);

  executor.run(taskflow).wait();
}


std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  bs_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


