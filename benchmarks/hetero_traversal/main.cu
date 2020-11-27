#include "graph.hpp"
#include <CLI11.hpp>

int main(int argc, char* argv[]) {
  
  CLI::App app{"HeteroTraversal"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_gpus {1};
  app.add_option("-g,--num_gpus", num_gpus, "number of gpus (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");

  unsigned cuda_ratio {2};
  app.add_option(
    "-c,--cuda_ratio", 
    cuda_ratio, 
    "cpu/cuda task ratio (the higher, the fewer cuda tasks (default=2)"
  );

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tf|tbb|omp (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tf" && m != "tbb" && m != "omp") {
          return "model name should be \"tbb\", \"tf\", or \"omp\"";
        }
        return "";
     });

  std::string path;
  app.add_option("-f,--file", path, "file path");

  CLI11_PARSE(app, argc, argv);


  cudaDeviceReset();

  if(path.empty()) {
     
    std::cout << "model=" << model << ' '
              << "num_threads=" << num_threads << ' '
              << "num_gpus=" << num_gpus << ' '
              << "num_rounds=" << num_rounds << ' '
              << std::endl;

    std::cout << std::setw(12) << "|V|+|E|"
              << std::setw(12) << "Runtime"
               << '\n';

    for(int i=10; i<=20010; i += 500) {

      Graph graph(i, 4*i, cuda_ratio);

      //std::ofstream ofs(std::string("graph") + std::to_string(graph.size()) + ".txt");
      //graph.dump(ofs);
      //continue;
      
      double runtime {0.0};
      double elapsed;
      
      for(unsigned j=0; j<=num_rounds; ++j) {

        if(model == "tf") {
          elapsed = measure_time_taskflow(graph, num_threads, num_gpus).count();
        }
        else if(model == "tbb") {
          elapsed = measure_time_tbb(graph, num_threads, num_gpus).count();
        }
        else if(model == "omp") {
          elapsed = measure_time_omp(graph, num_threads, num_gpus).count();
        }
        else {
          throw std::runtime_error("unknown model");
        }
        
        if(j) {
          runtime += elapsed;
        }
      }

      std::cout << std::setw(12) << graph.size() 
                << std::setw(12) << runtime / num_rounds / 1e3
                << std::endl;
    }
  }
  else {

    Graph graph(path);

    double runtime {0.0};
    double elapsed;
    
    for(unsigned j=0; j<=num_rounds; ++j) {

      if(model == "tf") {
        elapsed = measure_time_taskflow(graph, num_threads, num_gpus).count();
      }
      else if(model == "tbb") {
        elapsed = measure_time_tbb(graph, num_threads, num_gpus).count();
      }
      else if(model == "omp") {
        elapsed = measure_time_omp(graph, num_threads, num_gpus).count();
      }
      else {
        throw std::runtime_error("unknown model");
      }
      
      if(j) {
        runtime += elapsed;
        //std::cout << elapsed/1e3 << ',';
      }
    }

    //std::cout << std::setw(12) << graph.size() 
    //          << std::setw(12) << runtime / num_rounds / 1e3
    //          << std::endl;
    std::cout << runtime/num_rounds/1e3 << std::endl;
  }

  return 0;
}



