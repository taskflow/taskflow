#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/cuda/cudaflow.hpp>

namespace tf {
enum class cuda_compare_result {
  not_run,
  equal,
  not_equal,
};

template <typename T>
__global__ void assign_ptr_kernel(T* dest, T value) {
  if (threadIdx.x == 0) {
    *dest = value;
  }
}
template <typename T>
__global__ void test_is_equal_kernel(cuda_compare_result* dest, T expected, T const* value) {
  if (threadIdx.x == 0) {
    *dest = (expected == *value) ? cuda_compare_result::equal : cuda_compare_result::not_equal;
  }
}

struct cudaFreeFunctor {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

TEST_CASE("cudaGraph.subgraph.createWithKernel" * doctest::timeout(300)) {
  tf::cudaGraph graph;
  int* input{};
  REQUIRE(cudaMalloc(&input, sizeof(int)) == cudaSuccess);
  auto input_raii = std::unique_ptr<int, cudaFreeFunctor>(input);
  cuda_compare_result* is_equal_device{};
  REQUIRE(cudaMalloc(&is_equal_device, sizeof(*is_equal_device)) == cudaSuccess);
  auto is_equal_device_raii = std::unique_ptr<cuda_compare_result, cudaFreeFunctor>(is_equal_device);

  auto pre_task1 = graph.kernel({}, {}, 0, assign_ptr_kernel<int>, input, 55);
  auto pre_task2 = graph.kernel({}, {}, 0, assign_ptr_kernel<cuda_compare_result>
    , is_equal_device, cuda_compare_result::not_run);
  cuda_compare_result* is_equal_host{};
  REQUIRE(cudaMallocHost(&is_equal_host, sizeof(cuda_compare_result)) == cudaSuccess);
  auto host_raii = std::unique_ptr<cuda_compare_result, decltype([] (auto* ptr) {
    cudaFreeHost(ptr);
  })>(is_equal_host);
  // Add extra dependencies to pre_task:s just in case something is broken with dependencies for sub-grahps.
  auto post_task = graph.memcpy(is_equal_host, is_equal_device, sizeof(*is_equal_host)).succeed(pre_task1, pre_task2);
  // just to be sure that we don't get problems with lifetimes if the subgraph is temporary, we test this.
  {
    cudaGraph sub_graph;
    sub_graph.kernel({}, {}, 0, test_is_equal_kernel<int>, is_equal_device, 55, input);
    graph.sub_graph(sub_graph).succeed(pre_task1, pre_task2);
  }
  cudaStream stream;
  stream.run(cudaGraphExec(graph)).synchronize();
  TF_CHECK_CUDA(
    cudaMemcpy(is_equal_host, is_equal_device, sizeof(cuda_compare_result), cudaMemcpyDefault),
    "Failed to memcpy the end result");
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(*is_equal_host == cuda_compare_result::equal);
}
}
