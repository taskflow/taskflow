// This program computes the dot products over a set of independent vectors
// and compares the runtime between baseline (sequential), OpenMP, C++ thread,
// and Taskflow implementations.

#include "taskflow.hpp"
#include <random>
#include <numeric>
#include <fstream>

using matrix_t = std::vector<std::vector<float>>;

// ----------------------------------------------------------------------------
// Utility section
// ----------------------------------------------------------------------------

// Function: random_matrix
matrix_t random_matrix(size_t N) {
  
  thread_local std::default_random_engine gen(0);
  std::normal_distribution<float> d{0.0f, 1.0f};

  std::ostringstream oss;
  oss << "|----> generating " << N << "x" << N << " matrix by thread " 
      << std::this_thread::get_id() << "\n";
  std::cout << oss.str();
  
  matrix_t mat(N);
  for(auto& r : mat) {
    r.resize(N);
    for(auto& c : r) {
      c = d(gen);
    }
  }
  
  return mat;
}

// Operator: multiplication
matrix_t operator * (const matrix_t& A, const matrix_t& B) {

  if(A.empty() || B.empty() || A[0].size() != B.size()) {
    std::cout << A[0].size() << " " << B.size() << std::endl;
    throw std::runtime_error("Dimension mismatched in matrix multiplication\n");
  }
  
  size_t M, K, N;

  N = A.size();
  K = A[0].size();
  M = B[0].size();

  printf("A[%lux%lu] * B[%lux%lu]\n", N, K, K, M);
  
  // Initialize the matrix
  matrix_t ret(N);
  for(auto& r : ret) {
    r.resize(M);
    for(auto& c : r) {
      c = 0.0f;
    }
  }

  // Matrix multiplication
  for(size_t i=0; i<N; ++i) {
    for(size_t j=0; j<M; ++j) {
      for(size_t k=0; k<K; ++k) {
        ret[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  
  return ret;
}

// ----------------------------------------------------------------------------
// Task section
// ----------------------------------------------------------------------------

// Procedure: baseline
void baseline(const std::vector<size_t>& D) {
  
  std::cout << "========== baseline ==========\n";

  auto tbeg = std::chrono::steady_clock::now();

  std::cout << "Generating matrix As ...\n";
  std::vector<matrix_t> As(D.size());
  for(size_t j=0; j<D.size(); ++j) {
    As[j] = random_matrix(D[j]);
  }
  
  std::cout << "Generating matrix Bs ...\n";
  std::vector<matrix_t> Bs(D.size());
  for(size_t j=0; j<D.size(); ++j) {
    Bs[j] = random_matrix(D[j]);
  }
  
  std::cout << "Computing matrix product values Cs ...\n";
  std::vector<matrix_t> Cs(D.size());
  for(size_t j=0; j<D.size(); ++j) {
    Cs[j] = As[j] * Bs[j];
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "Baseline takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: openmp
void openmp(const std::vector<size_t>& D) {
  
  std::cout << "========== OpenMP ==========\n";

  auto tbeg = std::chrono::steady_clock::now();

  std::cout << "Generating matrix As ...\n";
  std::vector<matrix_t> As(D.size());
  #pragma omp parallel for
  for(int j=0; j<(int)D.size(); ++j) {
    As[j] = random_matrix(D[j]);
  }
  
  std::cout << "Generating matrix Bs ...\n";
  std::vector<matrix_t> Bs(D.size());
  #pragma omp parallel for
  for(int j=0; j<(int)D.size(); ++j) {
    Bs[j] = random_matrix(D[j]);
  }
  
  std::cout << "Computing matrix product values Cs ...\n";
  std::vector<matrix_t> Cs(D.size());
  #pragma omp parallel for
  for(int j=0; j<(int)D.size(); ++j) {
    Cs[j] = As[j] * Bs[j];
  }

  auto tend = std::chrono::steady_clock::now();

  std::cout << "OpenMP takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: cppthread
void cppthread(const std::vector<size_t>& D) {
  
  std::cout << "========== CppThread ==========\n";

  auto tbeg = std::chrono::steady_clock::now();

  tf::Threadpool tpl(std::thread::hardware_concurrency());

  std::cout << "Generating matrix As ...\n";
  std::vector<matrix_t> As(D.size());
  std::vector<std::future<void>> futures;

  for(size_t j=0; j<D.size(); ++j) {
    futures.push_back(tpl.async([&, j] () { As[j] = random_matrix(D[j]); }));
  }
  
  std::cout << "Generating matrix Bs ...\n";
  std::vector<matrix_t> Bs(D.size());
  for(size_t j=0; j<D.size(); ++j) {
    futures.push_back(tpl.async([&, j] () { Bs[j] = random_matrix(D[j]); }));
  }

  std::cout << "Synchronizing As and Bs ...\n";
  for(auto& fu : futures) {
    fu.get();
  }
  futures.clear();
  
  std::cout << "Computing matrix product values Cs ...\n";
  std::vector<matrix_t> Cs(D.size());
  for(size_t j=0; j<D.size(); ++j) {
    futures.push_back(tpl.async([&, j] () { Cs[j] = As[j] * Bs[j]; }));
  }
  
  std::cout << "Synchronizing Cs ...\n";
  for(auto& fu : futures) {
    fu.get();
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "CppThread takes " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbeg).count() 
            << " ms\n";
}

// Procedure: taskflow
void taskflow(const std::vector<size_t>& D) {
  
  auto tbeg = std::chrono::steady_clock::now();

  using builder_t  = typename tf::Task;

  tf::Taskflow tf;
  
  std::cout << "Generating task As ...\n";
  std::vector<matrix_t> As(D.size());
  std::vector<builder_t> TaskAs;
  for(size_t j=0; j<D.size(); ++j) {
    TaskAs.push_back(tf.silent_emplace([&, j] () { 
      As[j] = random_matrix(D[j]); 
    }));
  }

  std::cout << "Generating task Bs ...\n";
  std::vector<matrix_t> Bs(D.size());
  std::vector<builder_t> TaskBs;
  for(size_t j=0; j<D.size(); ++j) {
    TaskBs.push_back(tf.silent_emplace([&, j] () {
      Bs[j] = random_matrix(D[j]);
    }));
  }

  std::cout << "Generating task Cs ...\n";
  std::vector<matrix_t> Cs(D.size());
  std::vector<builder_t> TaskCs;
  for(size_t j=0; j<D.size(); ++j) {
    TaskCs.push_back(tf.silent_emplace([&, j] () {
      Cs[j] = As[j] * Bs[j];
    }));
  }

  // Build task dependency
  for(size_t j=0; j<D.size(); ++j) {
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
    std::cerr << "usage: ./matrix [baseline|openmp|cppthread|taskflow] N\n";
    std::exit(EXIT_FAILURE);
  }
  
  // Create a unbalanced dimension for vector products.
  const auto N = std::stoul(argv[2]);

  std::vector<size_t> dimensions(N);
  
  std::default_random_engine engine(0);
  std::uniform_int_distribution dis(1, 1000);

  std::cout << "matrix sizes = [";
  for(size_t i=0; i<dimensions.size(); ++i) {
    dimensions[i] = dis(engine);
    if(i) std::cout << ' ';
    std::cout << dimensions[i];
  }
  std::cout << "]\n";

  // Run methods
  if(std::string_view method(argv[1]); method == "baseline") {
    baseline(dimensions);
  }
  else if(method == "openmp") {
    openmp(dimensions);
  }
  else if(method == "cppthread") {
    cppthread(dimensions);
  }
  else if(method == "taskflow") {
    taskflow(dimensions);
  }
  else {
    std::cerr << "wrong method, shoud be [baseline|openmp|cppthread|taskflow]\n";
  }

  return 0;
}


