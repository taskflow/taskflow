#include <thread>
#include "dnn.hpp"

#define BENCHMARK(LIB, num_threads)                                    \
  {                                                                    \
    auto dnn {build_dnn()};                                            \
    std::cout << "Benchmark " #LIB << '\n';                            \
    auto t1 = std::chrono::high_resolution_clock::now();               \
    run_##LIB(dnn, num_threads);                                       \
    auto t2 = std::chrono::high_resolution_clock::now();               \
    std::cout << "Benchmark runtime: " << time_diff(t1, t2) << " s\n"; \
    dnn.validate(); \
  }

int main(int argc, char *argv[]){


  int sel = 0;
  if(argc > 1) {
    if(::strcmp(argv[1], "taskflow") == 0) {
      sel = 1;
    }
    else if(::strcmp(argv[1], "tbb") == 0) {
      sel = 2;
    }
    else if(::strcmp(argv[1], "omp") == 0) {
      sel = 3;
    }
  }

  auto num_threads = std::thread::hardware_concurrency();
  if(argc > 2) {
    assert(false);
    num_threads = std::atoi(argv[2]);
  }

  switch(sel) {
    case 1:  BENCHMARK(taskflow,   num_threads); break; 
    case 2:  BENCHMARK(tbb,        num_threads); break;
    case 3:  BENCHMARK(omp,        num_threads); break;
    default: BENCHMARK(sequential, num_threads); break;
  };

  return EXIT_SUCCESS;
}

