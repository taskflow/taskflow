#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/find.hpp>

// ----------------------------------------------------------------------------
// cuda_min_max_element
// ----------------------------------------------------------------------------

template <typename T>
void cuda_min_max_element() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      tf::cudaStream stream;
      tf::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = tf::cuda_malloc_shared<T>(n);
      auto min_i = tf::cuda_malloc_shared<unsigned>(1);
      auto max_i = tf::cuda_malloc_shared<unsigned>(1);

      // buffer
      void* buff;
      cudaMalloc(&buff, policy.min_element_bufsz<T>(n));

      for(int i=0; i<n; i++) {
        gdata[i] = rand() % 1000 - 500;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      tf::cudaStream s;
      tf::cudaDefaultExecutionPolicy p(s);

      tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a < b; }, buff
      );
      
      tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a < b; }, buff
      );
      s.synchronize();
      
      auto min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a < b; });
      auto max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a < b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a > b; }, buff
      );
      
      tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a > b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a > b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a > b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);

      // change the comparator
      tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      
      tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      tf::cuda_min_element(
        p, gdata, gdata+n, min_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      
      tf::cuda_max_element(
        p, gdata, gdata+n, max_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      s.synchronize();
      
      min_v = *std::min_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      max_v = *std::max_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      REQUIRE(std::abs(gdata[*min_i]) == std::abs(min_v));
      REQUIRE(std::abs(gdata[*max_i]) == std::abs(max_v));
      
      // deallocate the memory
      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(min_i) == cudaSuccess);
      REQUIRE(cudaFree(max_i) == cudaSuccess);
      REQUIRE(cudaFree(buff)  == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_min_max_element.int" * doctest::timeout(300)) {
  cuda_min_max_element<int>();
}

TEST_CASE("cuda_min_max_element.float" * doctest::timeout(300)) {
  cuda_min_max_element<float>();
}

