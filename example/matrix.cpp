// Cubist programming assignment.
//
// Author: Tsung-Wei Huang
//
// This program is accomplished by my self, without any advice or help from 
// other individuals. This is my own products.
//
// Dependency: taskflow.hpp
// taskflow.hpp is a c++ DAG-based task scheduler. It has been used in my open-source
// projects DtCraft and OpenTimer. 
// Check my github for more details: https://github.com/twhuang-uiuc

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstring>
#include <taskflow.hpp>

// ----------------------------------------------------------------------------
// Utility section
// ----------------------------------------------------------------------------

using matrix_t = std::vector<std::vector<float>>;

// Function: random_matrix
// Generate a matrix between 0 and 1
matrix_t random_matrix(size_t rows, size_t cols) {
  
  matrix_t mat;

  mat.resize(rows);
  for(size_t r=0; r<rows; ++r) {
    mat[r].resize(cols);
    for(size_t c=0; c<cols; ++c) {
      mat[r][c] = ::rand() / static_cast<float>(RAND_MAX);
    }
  }

  return mat;
}

// Procedure: save_matrix
// save a give matrix to a path
void save_matrix(const std::string& path, const matrix_t& mat) {
  
  std::cout << std::string("saving matrix ") + path.c_str() + "\n";
  
  std::ofstream ofs(path);

  if(!ofs.good() || mat.empty()) {
    throw std::invalid_argument("failed to save matrix");
  }

  ofs << mat.size() << ' ' << mat[0].size() << '\n';

  for(size_t r=0; r<mat.size(); ++r) {
    for(size_t c=0; c<mat[r].size(); ++c) {
      ofs << mat[r][c] << ' '; 
    }
    ofs << '\n';
  }
}

// Function: load_matrix
matrix_t load_matrix(const std::string& path) {

  std::cout << std::string("loading matrix ") + path.c_str() + "\n";
  
  std::ifstream ifs(path);

  if(!ifs.good()) {
    throw std::invalid_argument("failed to load matrix");
  }

  size_t rows, cols;

  ifs >> rows >> cols;

  matrix_t mat;
  mat.resize(rows);
  for(size_t r=0; r<rows; ++r) {
    mat[r].resize(cols);
  }

  for(size_t r=0; r<rows; ++r) {
    for(size_t c=0; c<cols; ++c) {
      ifs >> mat[r][c]; 
    }
  }
  
  return mat;
}

// Dummy caculation
matrix_t operator + (const matrix_t& a, auto b) {
  matrix_t res = a;
  return res;
}


// ----------------------------------------------------------------------------
// Task section
// ----------------------------------------------------------------------------

// Procedure: generate_test
void generate_test(size_t N) {
  // generate test data
  auto a = random_matrix(N, N);
  save_matrix("a.csv", a);
  auto b = random_matrix(N, N);
  save_matrix("b.csv", b);
}

// Procedure: function1
matrix_t func1(const matrix_t& x, auto&& j) {
  std::cout << "computing1 ...\n";
  matrix_t dummy = x;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return dummy;
}

// Procedure: function2
auto func2(const matrix_t& a, auto&& b) {
  std::cout << "computing2 ...\n";
  matrix_t dummy = a;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return dummy;
}

// Procedure: sequential
void sequential(size_t N) {

  generate_test(N);

  auto tbeg = std::chrono::steady_clock::now();

  auto a = load_matrix("a.csv");
  for(int j=1; j<=5; ++j) {
    auto tmp = func1(a, j);  
    save_matrix(std::string("a") + std::to_string(j) + ".csv", tmp);
  }

  auto b = load_matrix("b.csv");
  for(int j=1; j<=5; ++j) {
    auto tmp = func2(b, j);
    save_matrix(std::string("b") + std::to_string(j) + ".csv", tmp);
  }

  for(int j=1; j<=5; ++j) {
    auto a = load_matrix(std::string("a") + std::to_string(j) + ".csv");
    auto b = load_matrix(std::string("b") + std::to_string(j) + ".csv");
    auto c = func2(a, b);
    save_matrix(std::string("c") + std::to_string(j) + ".csv", c);
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "sequential version takes " 
            << std::chrono::duration_cast<std::chrono::seconds>(tend-tbeg).count() 
            << " seconds\n";
}

// Procedure: naive_parallel
void naive_parallel(size_t N, size_t num_threads = std::thread::hardware_concurrency()) {
  
  generate_test(N);

  auto tbeg = std::chrono::steady_clock::now();

  auto a = load_matrix("a.csv");
  #pragma omp parallel for num_threads(num_threads)
  for(int j=1; j<=5; ++j) {
    auto tmp = func1(a, j);  
    save_matrix(std::string("a") + std::to_string(j) + ".csv", tmp);
  }

  auto b = load_matrix("b.csv");
  #pragma omp parallel for num_threads(num_threads)
  for(int j=1; j<=5; ++j) {
    auto tmp = func2(b, j);
    save_matrix(std::string("b") + std::to_string(j) + ".csv", tmp);
  }

  #pragma omp parallel for num_threads(num_threads)
  for(int j=1; j<=5; ++j) {
    auto a = load_matrix(std::string("a") + std::to_string(j) + ".csv");
    auto b = load_matrix(std::string("b") + std::to_string(j) + ".csv");
    auto c = func2(a, b);
    save_matrix(std::string("c") + std::to_string(j) + ".csv", c);
  }
  
  auto tend = std::chrono::steady_clock::now();

  std::cout << "naive parallel version takes " 
            << std::chrono::duration_cast<std::chrono::seconds>(tend-tbeg).count() 
            << " seconds\n";
}

// Procedure: parallel
void parallel(size_t N, size_t num_threads = std::thread::hardware_concurrency()) {
  
  generate_test(N);

  auto tbeg = std::chrono::steady_clock::now();

  tf::Taskflow<int> tf(num_threads);

  // Parallelize the following tasks.
  // auto a = load_matrix("a.csv");
  // for(int j=1; j<=5; ++j) {
  //   auto tmp = func1(a, j);  
  //   save_matrix(std::string("a") + std::to_string(j) + ".csv", tmp);
  // }
  // auto b = load_matrix("b.csv");
  // for(int j=1; j<=5; ++j) {
  //   auto tmp = func2(b, j);
  //   save_matrix(std::string("b") + std::to_string(j) + ".csv", tmp);
  // }
  matrix_t a;
  auto load_a  = tf.silent_emplace([&] () { a = load_matrix("a.csv"); });
  auto save_a1 = tf.silent_emplace([&] () { save_matrix("a1.csv", func1(a, 1)); });
  auto save_a2 = tf.silent_emplace([&] () { save_matrix("a2.csv", func1(a, 2)); });
  auto save_a3 = tf.silent_emplace([&] () { save_matrix("a3.csv", func1(a, 3)); });
  auto save_a4 = tf.silent_emplace([&] () { save_matrix("a4.csv", func1(a, 4)); });
  auto save_a5 = tf.silent_emplace([&] () { save_matrix("a5.csv", func1(a, 5)); });

  tf.broadcast(load_a, {save_a1, save_a2, save_a3, save_a4, save_a5});

  matrix_t b;
  auto load_b  = tf.silent_emplace([&] () { b = load_matrix("b.csv"); });
  auto save_b1 = tf.silent_emplace([&] () { save_matrix("b1.csv", func1(b, 1)); });
  auto save_b2 = tf.silent_emplace([&] () { save_matrix("b2.csv", func1(b, 2)); });
  auto save_b3 = tf.silent_emplace([&] () { save_matrix("b3.csv", func1(b, 3)); });
  auto save_b4 = tf.silent_emplace([&] () { save_matrix("b4.csv", func1(b, 4)); });
  auto save_b5 = tf.silent_emplace([&] () { save_matrix("b5.csv", func1(b, 5)); });
  
  tf.broadcast(load_b, {save_b1, save_b2, save_b3, save_b4, save_b5});
  
  // Synchronize
  auto sync = tf.silent_emplace([&]() {std::cout << "a[1:5].csv and b[1:5].csv written\n";});

  tf.gather({save_a1, save_a2, save_a3, save_a4, save_a5, 
             save_b1, save_b2, save_b3, save_b4, save_b5}, sync);

  // Parallelize the following
  // for(int j=1; j<=5; ++j) {
  //   auto a = load_matrix(std::string("a") + std::to_string(j) + ".csv");
  //   auto b = load_matrix(std::string("b") + std::to_string(j) + ".csv");
  //   auto c = func2(a, b);
  //   save_matrix(std::string("c") + std::to_string(j) + ".csv", c);
  // }
  matrix_t a1, a2, a3, a4, a5, b1, b2, b3, b4, b5;
  auto load_a1 = tf.silent_emplace([&](){ a1 = load_matrix("a1.csv"); });
  auto load_a2 = tf.silent_emplace([&](){ a2 = load_matrix("a2.csv"); });
  auto load_a3 = tf.silent_emplace([&](){ a3 = load_matrix("a3.csv"); });
  auto load_a4 = tf.silent_emplace([&](){ a4 = load_matrix("a4.csv"); });
  auto load_a5 = tf.silent_emplace([&](){ a5 = load_matrix("a5.csv"); });
  auto load_b1 = tf.silent_emplace([&](){ a1 = load_matrix("b1.csv"); });
  auto load_b2 = tf.silent_emplace([&](){ a2 = load_matrix("b2.csv"); });
  auto load_b3 = tf.silent_emplace([&](){ a3 = load_matrix("b3.csv"); });
  auto load_b4 = tf.silent_emplace([&](){ a4 = load_matrix("b4.csv"); });
  auto load_b5 = tf.silent_emplace([&](){ a5 = load_matrix("b5.csv"); });
  auto save_c1 = tf.silent_emplace([&](){ save_matrix("c1.csv", func2(a1, b1)); });
  auto save_c2 = tf.silent_emplace([&](){ save_matrix("c2.csv", func2(a2, b2)); });
  auto save_c3 = tf.silent_emplace([&](){ save_matrix("c3.csv", func2(a3, b3)); });
  auto save_c4 = tf.silent_emplace([&](){ save_matrix("c4.csv", func2(a4, b4)); });
  auto save_c5 = tf.silent_emplace([&](){ save_matrix("c5.csv", func2(a5, b5)); });

  tf.broadcast(sync, {load_a1, load_a2, load_a3, load_a4, load_a5,
                      load_b1, load_b2, load_b3, load_b4, load_b5});

  tf.precede(load_a1, save_c1)
    .precede(load_b1, save_c1)
    .precede(load_a2, save_c2)
    .precede(load_b2, save_c2)
    .precede(load_a3, save_c3)
    .precede(load_b3, save_c3)
    .precede(load_a4, save_c4)
    .precede(load_b4, save_c4)
    .precede(load_a5, save_c5)
    .precede(load_b5, save_c5)
    .wait_for_all();

  auto tend = std::chrono::steady_clock::now();
  std::cout << "parallel version takes " 
            << std::chrono::duration_cast<std::chrono::seconds>(tend-tbeg).count() 
            << " seconds\n";
}

// ------------------------------------------------------------------------------------------------

// Function: main
int main(int argc, char* argv[]) {
  
  if(argc != 3) {
    std::cerr << "usage: ./cubist N [seq|naive|taskflow]\n";
    std::exit(EXIT_FAILURE);
  }

  if(std::strcmp(argv[2], "seq") == 0) {
    sequential(std::stoi(argv[1]));
  }
  else if(std::strcmp(argv[2], "naive") == 0) {
    naive_parallel(std::stoi(argv[1]));
  }
  else if(std::strcmp(argv[2], "taskflow") == 0) {
    parallel(std::stof(argv[1]));
  }
  else {
    std::cerr << "wrong method\n";
  }

  return 0;
}











