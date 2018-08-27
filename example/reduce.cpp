// This example demonstrates how to use 'reduce' method.

#include "taskflow.hpp"

#include <chrono>
#include <limits.h>

struct Data {
  int a {::rand()};
  int b {::rand()};
  int transform() const { 
    return a*a + 2*a*b + b*b;
  }
};

// Procedure: reduce
// This procedure demonstrates 
void reduce() {

  std::vector<int> data;
  for(int i=0; i<40000000; ++i) {
    data.push_back(::rand());
  }
  
  std::cout << "Benchmark: reduce" << std::endl;

  // sequential method
  auto sbeg = std::chrono::steady_clock::now();
  auto smin = std::numeric_limits<int>::max();
  for(auto& d : data) {
    smin = std::min(smin, d);
  }
  auto send = std::chrono::steady_clock::now();
  std::cout << "[sequential] reduce: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(send - sbeg).count()
            << std::endl;

  // taskflow
  auto tbeg = std::chrono::steady_clock::now();
  tf::Taskflow tf;
  auto tmin = std::numeric_limits<int>::max();
  tf.reduce(data.begin(), data.end(), tmin, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
  tf.wait_for_all();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "[taskflow] reduce: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbeg).count()
            << std::endl;
  
  // assertion
  assert(tmin == smin);
}

// Procedure: transform_reduce
void transform_reduce() {

  std::cout << "Benchmark: transform_reduce" << std::endl;
  
  std::vector<Data> data(40000000);
  
  // sequential method
  auto sbeg = std::chrono::steady_clock::now();
  auto smin = std::numeric_limits<int>::max();
  for(auto& d : data) {
    smin = std::min(smin, d.transform());
  }
  auto send = std::chrono::steady_clock::now();
  std::cout << "[sequential] transform_reduce " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(send - sbeg).count()
            << std::endl;
  
  // taskflow
  auto tbeg = std::chrono::steady_clock::now();
  tf::Taskflow tf;
  auto tmin = std::numeric_limits<int>::max();
  tf.transform_reduce(data.begin(), data.end(), tmin, 
    [] (int l, int r) { return std::min(l, r); },
    [] (const Data& d) { return d.transform(); }
  );
  tf.wait_for_all();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "[taskflow] transform_reduce " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbeg).count()
            << std::endl;

  // assertion
  assert(tmin == smin);
}

// ------------------------------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./reduce [reduce|transform_reduce]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  
  if(std::string_view method(argv[1]); method == "reduce") {
    reduce();
  }
  else if(method == "transform_reduce") {
    transform_reduce();
  }
  else {
    std::cerr << "invalid method " << method << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
