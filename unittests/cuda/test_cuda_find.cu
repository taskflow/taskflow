#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/find.hpp>

// ----------------------------------------------------------------------------
// cuda_find_if
// ----------------------------------------------------------------------------

template <typename T>
void cuda_find_if() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      tf::cudaStream stream;
      tf::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = tf::cuda_malloc_shared<T>(n);
      auto gfind = tf::cuda_malloc_shared<unsigned>(1);

      // cpu data
      auto hdata = std::vector<T>(n);

      // initialize the data
      for(int i=0; i<n; i++) {
        T k = rand()% 100;
        gdata[i] = k;
        hdata[i] = k;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      tf::cudaStream s;
      tf::cudaDefaultExecutionPolicy p(s);
      tf::cuda_find_if(
        p, gdata, gdata+n, gfind, []__device__(T v) { return v == (T)50; }
      );
      s.synchronize();
      
      // --------------------------------------------------------------------------
      // CPU find
      // --------------------------------------------------------------------------
      auto hiter = std::find_if(
        hdata.begin(), hdata.end(), [=](T v) { return v == (T)50; }
      );
      
      // --------------------------------------------------------------------------
      // verify the result
      // --------------------------------------------------------------------------
      unsigned hfind = std::distance(hdata.begin(), hiter);
      REQUIRE(*gfind == hfind);

      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(gfind) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_find_if.int" * doctest::timeout(300)) {
  cuda_find_if<int>();
}

TEST_CASE("cuda_find_if.float" * doctest::timeout(300)) {
  cuda_find_if<float>();
}
