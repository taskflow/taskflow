// This example demonstrates how to create a parallel-reduction task.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>

#define MAX_DATA_SIZE 40000000

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

  std::cout << "Benchmark: reduce" << std::endl;

  std::vector<int> data;
  data.reserve(MAX_DATA_SIZE);
  for(int i=0; i<MAX_DATA_SIZE; ++i) {
    data.push_back(::rand());
  }

  // sequential method
  auto sbeg = std::chrono::steady_clock::now();
  auto smin = std::numeric_limits<int>::max();
  for(auto& d : data) {
    smin = std::min(smin, d);
  }
  auto send = std::chrono::steady_clock::now();
  std::cout << "[sequential] reduce: "
            << std::chrono::duration_cast<std::chrono::microseconds>(send - sbeg).count()
            << " us\n";

  // taskflow
  auto tbeg = std::chrono::steady_clock::now();
  tf::Taskflow taskflow;
  tf::Executor executor;
  auto tmin = std::numeric_limits<int>::max();
  taskflow.reduce(
    data.begin(),
    data.end(),
    tmin,
    [] (int& l, const auto& r) { return std::min(l, r); }
  );
  executor.run(taskflow).get();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "[taskflow] reduce: "
            << std::chrono::duration_cast<std::chrono::microseconds>(tend - tbeg).count()
            << " us\n";

  // assertion
  if(tmin == smin) {
    std::cout << "result is correct" << std::endl;
  }
  else {
    std::cout << "result is incorrect: " << smin << " != " << tmin << std::endl;
  }

  taskflow.dump(std::cout);
}

// Procedure: transform_reduce
void transform_reduce() {

  std::cout << "Benchmark: transform_reduce" << std::endl;

  std::vector<Data> data(MAX_DATA_SIZE);

  // sequential method
  auto sbeg = std::chrono::steady_clock::now();
  auto smin = std::numeric_limits<int>::max();
  for(auto& d : data) {
    smin = std::min(smin, d.transform());
  }
  auto send = std::chrono::steady_clock::now();
  std::cout << "[sequential] transform_reduce "
            << std::chrono::duration_cast<std::chrono::microseconds>(send - sbeg).count()
            << " us\n";

  // taskflow
  auto tbeg = std::chrono::steady_clock::now();
  tf::Taskflow tf;
  auto tmin = std::numeric_limits<int>::max();
  tf.transform_reduce(data.begin(), data.end(), tmin,
    [] (int l, int r) { return std::min(l, r); },
    [] (const Data& d) { return d.transform(); }
  );
  tf::Executor().run(tf).get();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "[taskflow] transform_reduce "
            << std::chrono::duration_cast<std::chrono::microseconds>(tend - tbeg).count()
            << " us\n";

  // assertion
  assert(tmin == smin);
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./reduce [reduce|transform_reduce]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if(std::strcmp(argv[1], "reduce") == 0) {
    reduce();
  }
  else if(std::strcmp(argv[1], "transform_reduce") == 0) {
    transform_reduce();
  }
  else {
    std::cerr << "invalid method " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
