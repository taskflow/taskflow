// This program demonstrates how to find an element in a vector
// using the CUDA standard algorithms in Taskflow.

#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/find.hpp> 

int main(int argc, char* argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./cuda_find N\n";
    std::exit(EXIT_FAILURE);
  }

  unsigned N = std::atoi(argv[1]);

  // gpu data
  auto gdata = tf::cuda_malloc_shared<int>(N);
  auto gfind = tf::cuda_malloc_shared<unsigned>(1);

  // cpu data
  auto hdata = std::vector<int>(N);

  size_t tgpu{0}, tcpu{0};

  // initialize the data
  for(unsigned i=0; i<N; i++) {
    auto k = rand();
    gdata[i] = k;
    hdata[i] = k;
  }

  // --------------------------------------------------------------------------
  // GPU find
  // --------------------------------------------------------------------------
  auto beg = std::chrono::steady_clock::now();
  tf::cudaStream s;
  tf::cudaDefaultExecutionPolicy p(s);
  tf::cuda_find_if(
    p, gdata, gdata+N, gfind, []__device__(int v) { return v == 100; }
  );
  s.synchronize();
  auto end = std::chrono::steady_clock::now();
  tgpu += std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count();
  
  // --------------------------------------------------------------------------
  // CPU find
  // --------------------------------------------------------------------------
  beg = std::chrono::steady_clock::now();
  auto hiter = std::find_if(
    hdata.begin(), hdata.end(), [=](int v) { return v == 100; }
  );
  end = std::chrono::steady_clock::now();
  tcpu += std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count();
  
  // --------------------------------------------------------------------------
  // verify the result
  // --------------------------------------------------------------------------
  if(unsigned hfind = std::distance(hdata.begin(), hiter); *gfind != hfind) {
    printf("gdata[%u]=%d, hdata[%u]=%d\n", 
      *gfind, gdata[*gfind], hfind, hdata[hfind]
    );
    throw std::runtime_error("incorrect result");
  }

  // output the time
  std::cout << "GPU time: " << tgpu << '\n'
            << "CPU time: " << tcpu << std::endl;
  
  // delete the memory
  tf::cuda_free(gdata);
  tf::cuda_free(gfind);

  return 0;
}
