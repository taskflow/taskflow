// This program computes the dot products over a set of independent vectors
// and compares the runtime between sequential, OpenMP, and Taskflow 
// implementations.

#include "taskflow.hpp"
#include <random>
#include <numeric>
#include <fstream>

// ----------------------------------------------------------------------------
// Utility section
// ----------------------------------------------------------------------------

// Function: random_vector
// Generate a vector between 0 and 1
std::vector<float> random_vector(size_t N) {
  
  // avoid global state race condition.
  thread_local std::default_random_engine gen(0);

  std::normal_distribution<float> d{0.0f, 1.0f};

  std::ostringstream oss;
  oss << "|----> generating vector by thread " << std::this_thread::get_id() << "\n";
  std::cout << oss.str();

  std::vector<float> vec(N);
  for(auto& v : vec) {
    v = d(gen);
  }
  return vec;
}

// ----------------------------------------------------------------------------
// Task section
// ----------------------------------------------------------------------------

// Procedure: baseline
void baseline(const std::vector<size_t>& dimensions) {
  
  std::cout << "========== baseline ==========\n";

  auto tbeg = std::chrono::steady_clock::now();

  std::cout << "Generating vector As ...\n";
  std::vector<std::vector<float>> As(dimensions.size());
  for(size_t j=0; j<dimensions.size(); ++j) {
    As[j] = random_vector(dimensions[j]);
  }
  
  std::cout << "Generating vector Bs ...\n";
  std::vector<std::vector<float>> Bs(dimensions.size());
  for(size_t j=0; j<dimensions.size(); ++j) {
    Bs[j] = random_vector(dimensions[j]);
  }
  
  std::cout << "Computing inner product values Cs ...\n";
  std::vector<float> Cs(dimensions.size());
  for(size_t j=0; j<dimensions.size(); ++j) {
    Cs[j] = std::inner_product(As[j].begin(), As[j].end(), Bs[j].begin(), 0.0f);
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "Baseline takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: openmp_parallel
void openmp_parallel(const std::vector<size_t>& dimensions) {
  
  std::cout << "========== OpenMP ==========\n";

  auto tbeg = std::chrono::steady_clock::now();

  std::cout << "Generating vector As ...\n";
  std::vector<std::vector<float>> As(dimensions.size());

  #pragma omp parallel for num_threads(4)
  for(size_t j=0; j<dimensions.size(); ++j) {
    As[j] = random_vector(dimensions[j]);
  }
  
  std::cout << "Generating vector Bs ...\n";
  std::vector<std::vector<float>> Bs(dimensions.size());
  #pragma omp parallel for num_threads(4)
  for(size_t j=0; j<dimensions.size(); ++j) {
    Bs[j] = random_vector(dimensions[j]);
  }
  
  std::cout << "Computing inner product values Cs ...\n";
  std::vector<float> Cs(dimensions.size());
  #pragma omp parallel for num_threads(4)
  for(size_t j=0; j<dimensions.size(); ++j) {
    Cs[j] = std::inner_product(As[j].begin(), As[j].end(), Bs[j].begin(), 0.0f);
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "OpenMP takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: taskflow_parallel
void taskflow_parallel(const std::vector<size_t>& dimensions) {
  
  auto tbeg = std::chrono::steady_clock::now();

  using builder_t  = typename tf::Taskflow::TaskBuilder;

  tf::Taskflow tf(4);
  
  std::cout << "Generating task As ...\n";
  std::vector<std::vector<float>> As(dimensions.size());
  std::vector<builder_t> TaskAs;
  for(size_t j=0; j<dimensions.size(); ++j) {
    TaskAs.push_back(tf.silent_emplace([&, j] () { 
      As[j] = random_vector(dimensions[j]); 
    }));
  }

  std::cout << "Generating task Bs ...\n";
  std::vector<std::vector<float>> Bs(dimensions.size());
  std::vector<builder_t> TaskBs;
  for(size_t j=0; j<dimensions.size(); ++j) {
    TaskBs.push_back(tf.silent_emplace([&, j] () {
      Bs[j] = random_vector(dimensions[j]);
    }));
  }

  std::cout << "Generating task Cs ...\n";
  std::vector<float> Cs(dimensions.size());
  std::vector<builder_t> TaskCs;
  for(size_t j=0; j<dimensions.size(); ++j) {
    TaskCs.push_back(tf.silent_emplace([&, j] () {
      Cs[j] = std::inner_product(As[j].begin(), As[j].end(), Bs[j].begin(), 0.0f);
    }));
  }

  // Build task dependency
  for(size_t j=0; j<dimensions.size(); ++j) {
    TaskCs[j].gather({TaskAs[j], TaskBs[j]});
  }

  tf.wait_for_all();

  auto tend = std::chrono::steady_clock::now();
  std::cout << "Taskflow takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// ------------------------------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 3) {
    std::cerr << "usage: ./matrix N [baseline|openmp|taskflow]\n";
    std::exit(EXIT_FAILURE);
  }
  
  // Create a unbalanced dimension for vector products.
  const auto N = std::stoul(argv[1]);

  std::vector<size_t> vector_sizes(N);
  
  std::default_random_engine engine(0);
  std::uniform_int_distribution dis(1, 4000000);

  std::cout << "vector sizes = [";
  for(size_t i=0; i<N; ++i) {
    vector_sizes[i] = dis(engine);
    if(i) std::cout << ' ';
    std::cout << vector_sizes[i];
  }
  std::cout << "]\n";

  // Run methods
  if(std::string_view method(argv[2]); method == "baseline") {
    baseline(vector_sizes);
  }
  else if(method == "openmp") {
    openmp_parallel(vector_sizes);
  }
  else if(method == "taskflow") {
    taskflow_parallel(vector_sizes);
  }
  else {
    std::cerr << "wrong method, shoud be [baseline|openmp|taskflow]\n";
  }

  return 0;
}


