#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/merge.hpp>

// ----------------------------------------------------------------------------
// cuda_merge
// ----------------------------------------------------------------------------

template <typename T>
void cuda_merge() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n1=0; n1<=123456; n1 = n1*2 + 1) {
    for(int n2=0; n2<=123456; n2 = n2*2 + 1) {
    
      taskflow.emplace([n1, n2](){

        // gpu data
        auto da = tf::cuda_malloc_shared<T>(n1);
        auto db = tf::cuda_malloc_shared<T>(n2);
        auto dc = tf::cuda_malloc_shared<T>(n1 + n2);

        // host data
        std::vector<T> ha(n1), hb(n2), hc(n1 + n2);

        for(int i=0; i<n1; i++) {
          da[i] = ha[i] = rand()%100;
        }
        for(int i=0; i<n2; i++) {
          db[i] = hb[i] = rand()%100;
        }
        
        std::sort(da, da+n1);
        std::sort(db, db+n2);
        std::sort(ha.begin(), ha.end());
        std::sort(hb.begin(), hb.end());

        // --------------------------------------------------------------------------
        // GPU merge
        // --------------------------------------------------------------------------

        tf::cudaStream stream;
        tf::cudaDefaultExecutionPolicy policy(stream);

        // allocate the buffer
        void* buf;
        REQUIRE(cudaMalloc(&buf, policy.merge_bufsz(n1, n2)) == cudaSuccess);

        tf::cuda_merge(policy, 
          da, da+n1, db, db+n2, dc, tf::cuda_less<T>{}, buf
        );
        stream.synchronize();

        // --------------------------------------------------------------------------
        // CPU merge
        // --------------------------------------------------------------------------
        std::merge(ha.begin(), ha.end(), hb.begin(), hb.end(), hc.begin());

        // --------------------------------------------------------------------------
        // verify the result
        // --------------------------------------------------------------------------

        for(int i=0; i<n1+n2; i++) {
          REQUIRE(dc[i] == hc[i]);
        }
        
        // --------------------------------------------------------------------------
        // deallocate the memory
        // --------------------------------------------------------------------------
        REQUIRE(cudaFree(da) == cudaSuccess);
        REQUIRE(cudaFree(db) == cudaSuccess);
        REQUIRE(cudaFree(dc) == cudaSuccess);
        REQUIRE(cudaFree(buf) == cudaSuccess);
      });
    }
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_merge.int" * doctest::timeout(300)) {
  cuda_merge<int>();
}

TEST_CASE("cuda_merge.float" * doctest::timeout(300)) {
  cuda_merge<float>();
}

// ----------------------------------------------------------------------------
// cuda_merge_by_key
// ----------------------------------------------------------------------------

template <typename T>
void cuda_merge_by_key() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  for(int n1=0; n1<=123456; n1 = n1*2 + 1) {
    for(int n2=0; n2<=123456; n2 = n2*2 + 1) {
    
      taskflow.emplace([n1, n2](){

        // gpu data
        auto da_k = tf::cuda_malloc_shared<T>(n1);
        auto da_v = tf::cuda_malloc_shared<T>(n1);
        auto db_k = tf::cuda_malloc_shared<T>(n2);
        auto db_v = tf::cuda_malloc_shared<T>(n2);
        auto dc_k = tf::cuda_malloc_shared<T>(n1 + n2);
        auto dc_v = tf::cuda_malloc_shared<T>(n1 + n2);

        std::unordered_map<T, T> map;

        for(int i=0; i<n1; i++) {
          da_k[i] = 1 + 2*i;
          da_v[i] = rand();
          map[da_k[i]] = da_v[i];
        }

        for(int i=0; i<n2; i++) {
          db_k[i] = 2*i;
          db_v[i] = rand();
          map[db_k[i]] = db_v[i];
        }

        REQUIRE(map.size() == n1 + n2);
        
        tf::cudaStream stream;
        tf::cudaDefaultExecutionPolicy policy(stream);

        // allocate the buffer
        void* buf;
        REQUIRE(cudaMalloc(&buf, policy.merge_bufsz(n1, n2)) == cudaSuccess);

        tf::cuda_merge_by_key(
          policy, 
          da_k, da_k+n1, da_v,
          db_k, db_k+n2, db_v,
          dc_k, dc_v,
          tf::cuda_less<T>{}, 
          buf
        );
        stream.synchronize();

        // --------------------------------------------------------------------------
        // verify the result
        // --------------------------------------------------------------------------

        REQUIRE(std::is_sorted(dc_k, dc_k+n1+n2));

        for(int i=0; i<n1+n2; i++) {
          REQUIRE(map.find(dc_k[i]) != map.end());
          REQUIRE(dc_v[i] == map[dc_k[i]]);
        }
        
        // --------------------------------------------------------------------------
        // deallocate the memory
        // --------------------------------------------------------------------------
        REQUIRE(cudaFree(da_k) == cudaSuccess);
        REQUIRE(cudaFree(da_v) == cudaSuccess);
        REQUIRE(cudaFree(db_k) == cudaSuccess);
        REQUIRE(cudaFree(db_v) == cudaSuccess);
        REQUIRE(cudaFree(dc_k) == cudaSuccess);
        REQUIRE(cudaFree(dc_v) == cudaSuccess);
        REQUIRE(cudaFree(buf) == cudaSuccess);
      });
    }
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_merge_by_key.int" * doctest::timeout(300)) {
  cuda_merge_by_key<int>();
}

TEST_CASE("cuda_merge_by_key.float" * doctest::timeout(300)) {
  cuda_merge_by_key<float>();
}



