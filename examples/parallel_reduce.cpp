// This example demonstrates how to create a parallel-reduction task.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>

struct Data {
  int a {::rand()};
  int b {::rand()};
  int transform() const {
    return a*a + 2*a*b + b*b;
  }
};

// Procedure: reduce
// This procedure demonstrates
void reduce(size_t N) {

  std::cout << "Benchmark: reduce" << std::endl;

  std::vector<int> data;
  data.reserve(N);
  for(size_t i=0; i<N; ++i) {
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
void transform_reduce(size_t N) {

  std::cout << "Benchmark: transform_reduce" << std::endl;

  std::vector<Data> data(N);

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

void reduce_by_index(size_t N) {
  
  std::cout << "Benchmark: reduce_by_key" << std::endl;

  tf::Executor executor;
  tf::Taskflow taskflow;
  
  std::vector<double> data(N);
  double res = 1.0;

  auto tbeg = std::chrono::steady_clock::now();
  taskflow.reduce_by_index(
    tf::IndexRange<size_t>(0, N, 1),
    // final result
    res,
    // local reducer
    [&](tf::IndexRange<size_t> subrange, std::optional<double> running_total) {
      double residual = running_total ? *running_total : 0.0;
      for(size_t i=subrange.begin(); i<subrange.end(); i+=subrange.step_size()) {
        data[i] = 1.0;
        residual += data[i];
      }
      printf("partial sum = %lf\n", residual);
      return residual;
    },
    // global reducer
    std::plus<double>()
  );
  executor.run(taskflow).wait();
  auto tend = std::chrono::steady_clock::now();
  std::cout << "[taskflow] reduce_by_key "
            << std::chrono::duration_cast<std::chrono::microseconds>(tend - tbeg).count()
            << " us\n";
}

// ----------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 3) {
    std::cerr << "usage: ./reduce [reduce|transform_reduce|reduce_by_index] N" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if(std::strcmp(argv[1], "reduce") == 0) {
    reduce(std::stoul(argv[2]));
  }
  else if(std::strcmp(argv[1], "transform_reduce") == 0) {
    transform_reduce(std::stoul(argv[2]));
  }
  else if(std::strcmp(argv[1], "reduce_by_index") == 0) {
    reduce_by_index(std::stoul(argv[2]));
  }
  else {
    std::cerr << "invalid method " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
