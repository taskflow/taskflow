// This program demonstrates how to perform parallel inclusive scan.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/scan.hpp>

int main(int argc, char* argv[]) {

  if(argc != 3) {
    std::cerr << "usage: ./inclusive_scan num_workers num_elements\n"; 
    std::exit(EXIT_FAILURE);
  }

  size_t W = std::atoi(argv[1]);
  size_t N = std::atoi(argv[2]);

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<std::string> elements(N), scan_seq(N), scan_par(N);
  for(size_t i=0; i<N; i++) {
    elements[i] = std::to_string(i);
  }
  
  // sequential inclusive scan
  {
    std::cout << "sequential inclusive scan ... ";
    auto beg = std::chrono::steady_clock::now();
    std::inclusive_scan(
      elements.begin(), elements.end(), scan_seq.begin(), std::plus<std::string>{}
    );
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-beg).count()
              << "ns\n";
  }

  // create a parallel inclusive scan task
  {
    std::cout << "parallel   inclusive scan ... ";
    auto beg = std::chrono::steady_clock::now();
    taskflow.inclusive_scan(
      elements.begin(), elements.end(), scan_par.begin(), std::plus<std::string>{}
    );
    executor.run(taskflow).wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-beg).count()
              << "ns\n";
  }

  // verify the result
  for(size_t i=0; i<N; i++) {
    printf(
      "scan_seq[%lu]=%.*s..., scan_par[%lu]=%.*s...\n",
      i, 10, scan_seq[i].c_str(), i, 10, scan_par[i].c_str()
    );
    if(scan_seq[i] != scan_par[i]) {
      throw std::runtime_error("incorrect result");
    }
  }

  printf("correct result\n");

  return 0;
}
