#include <thread>
#include <iomanip>
#include "dnn.hpp"

// Function: measure_time_taskflow
std::chrono::milliseconds measure_time_taskflow(
  unsigned num_iterations,
  unsigned num_threads
) {
  auto t1 = std::chrono::high_resolution_clock::now();
  run_taskflow(num_iterations, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_omp
std::chrono::milliseconds measure_time_omp(
  unsigned num_iterations,
  unsigned num_threads
) {
  auto t1 = std::chrono::high_resolution_clock::now();
  run_omp(num_iterations, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: measure_time_tbb
std::chrono::milliseconds measure_time_tbb(
  unsigned num_iterations,
  unsigned num_threads
) {
  auto t1 = std::chrono::high_resolution_clock::now();
  run_tbb(num_iterations, num_threads);
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}

// Function: main
int main(int argc, char *argv[]){
  
  unsigned num_threads = std::thread::hardware_concurrency();

  if(argc > 1) {
    num_threads = std::atoi(argv[1]);
  }

  {
    std::string path = std::experimental::filesystem::current_path();
    path = path.substr(0, path.rfind("cpp-taskflow") + 12);
    path += "/benchmark/parallel_dnn/";

    IMAGES = read_mnist_image(path + "./train-images.data");
    LABELS = read_mnist_label(path + "./train-labels.data");
    TEST_IMAGES = read_mnist_image(path + "./t10k-images-idx3-ubyte");
    TEST_LABELS = read_mnist_label(path + "./t10k-labels-idx1-ubyte");  
  }

  ////auto mnist = build_dnn(100); 
  //MNIST_DNN mnist;
  //init_dnn(mnist);

  ////run_sequential(mnist, 4);
  //run_sequential2(mnist, 4);
  //run_sequential2(10, 4);
  //exit(1);

  int rounds {2};

  std::cout << std::setw(12) << "# iterations"
            << std::setw(12) << "OpenMP"
            << std::setw(12) << "TBB"
            << std::setw(12) << "Taskflow"
            << std::setw(12) << "speedup1"
            << std::setw(12) << "speedup2"
            << '\n';

  for(int iter=10; iter<=100; iter+=10) {
    
    double omp_time {0.0};
    double tbb_time {0.0};
    double tf_time  {0.0};

    for(int j=0; j<rounds; ++j) {
      omp_time += measure_time_omp(iter, num_threads).count();
      tbb_time += measure_time_tbb(iter, num_threads).count();
      tf_time += measure_time_taskflow(iter, num_threads).count();
    }
    
    std::cout << std::setw(12) << iter
              << std::setw(12) << omp_time / rounds / 1e3
              << std::setw(12) << tbb_time / rounds / 1e3 
              << std::setw(12) << tf_time  / rounds / 1e3 
              << std::setw(12) << omp_time / tf_time
              << std::setw(12) << tbb_time / tf_time
              << std::endl;
  }

  return EXIT_SUCCESS;
}

