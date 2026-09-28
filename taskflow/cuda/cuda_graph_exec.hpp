#pragma once

#include "cuda_graph.hpp"


namespace tf {

// ----------------------------------------------------------------------------
// cudaGraphExec
// ----------------------------------------------------------------------------

/**
@class cudaGraphExec

@brief class to create an executable CUDA graph with unique ownership

This class wraps a `cudaGraphExec_t` handle with `std::unique_ptr` to ensure proper
resource management and automatic cleanup.
The executable graph is created by `cudaGraphInstantiate` and destroyed by `cudaGraphExecDestroy`.
*/
class cudaGraphExec : public std::unique_ptr<
  std::remove_pointer_t<cudaGraphExec_t>,
  cudaProxyDeleter<cudaGraphExecDestroy>
> {

  public:

  /**
  @brief constructs an empty executable graph that manages no `cudaGraphExec_t`
  */
  cudaGraphExec() = default;

  /**
  @brief constructs an executable graph that takes ownership of the given `cudaGraphExec_t`
  */
  explicit cudaGraphExec(cudaGraphExec_t exec) : unique_ptr(exec) {
  }

  /**
  @brief instantiates an executable graph from the given CUDA graph using `cudaGraphInstantiate`

  A tf::cudaGraph implicitly converts to `cudaGraph_t` and can be passed directly.
  */
  explicit cudaGraphExec(cudaGraph_t graph) : unique_ptr([graph](){
    cudaGraphExec_t exec;
    TF_CHECK_CUDA(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
      "failed to create an executable graph"
    );
    return exec;
  }()) {
  }

  /**
  @brief implicitly converts to the managed `cudaGraphExec_t`
  */
  operator cudaGraphExec_t () const noexcept { return get(); }

  // ----------------------------------------------------------------------------------------------
  // Update Methods
  // ----------------------------------------------------------------------------------------------

  /**
  @brief updates parameters of a host task

  This method updates the parameter of the given host task (similar to tf::cudaFlow::host).
  */
  template <typename C>
  void host(cudaTask task, C&& callable, void* user_data);
  
  /**
  @brief updates parameters of a kernel task

  The method is similar to tf::cudaFlow::kernel but operates on a task
  of type tf::cudaTaskType::KERNEL.
  The kernel function name must NOT change.
  */
  template <typename F, typename... ArgsT>
  void kernel(
    cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT... args
  );
  
  /**
  @brief updates parameters of a memset task

  The method is similar to tf::cudaFlow::memset but operates on a task
  of type tf::cudaTaskType::MEMSET.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  void memset(cudaTask task, void* dst, int ch, size_t count);

  /**
  @brief updates parameters of a memcpy task

  The method is similar to tf::cudaFlow::memcpy but operates on a task
  of type tf::cudaTaskType::MEMCPY.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  void memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);
  
  /**
  @brief updates parameters of a memset task to a zero task

  The method is similar to tf::cudaFlow::zero but operates on
  a task of type tf::cudaTaskType::MEMSET.

  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  void zero(cudaTask task, T* dst, size_t count);

  /**
  @brief updates parameters of a memset task to a fill task

  The method is similar to tf::cudaFlow::fill but operates on a task
  of type tf::cudaTaskType::MEMSET.

  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  void fill(cudaTask task, T* dst, T value, size_t count);
  
  /**
  @brief updates parameters of a memcpy task to a copy task

  The method is similar to tf::cudaFlow::copy but operates on a task
  of type tf::cudaTaskType::MEMCPY.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T,
    std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
  >
  void copy(cudaTask task, T* tgt, const T* src, size_t num);
  
  //---------------------------------------------------------------------------
  // Algorithm Primitives
  //---------------------------------------------------------------------------

  /**
  @brief updates a single-threaded kernel task

  This method is similar to cudaFlow::single_task but operates
  on an existing task.
  */
  template <typename C>
  void single_task(cudaTask task, C c);
  
  /**
  @brief updates parameters of a `for_each` kernel task created from the CUDA graph of `*this`
  */
  template <typename E = cudaDefaultExecutionPolicy, typename I, typename C>
  void for_each(cudaTask task, I first, I last, C callable);
  
  /**
  @brief updates parameters of a `for_each_index` kernel task created from the CUDA graph of `*this`
  */
  template <typename E = cudaDefaultExecutionPolicy, typename I, typename C>
  void for_each_index(cudaTask task, I first, I last, I step, C callable);

  /**
  @brief updates parameters of a `transform` kernel task created from the CUDA graph of `*this`
  */
  template <typename E = cudaDefaultExecutionPolicy, typename I, typename O, typename C>
  void transform(cudaTask task, I first, I last, O output, C c);

  /**
  @brief updates parameters of a `transform` kernel task created from the CUDA graph of `*this`
  */
  template <typename E = cudaDefaultExecutionPolicy, typename I1, typename I2, typename O, typename C>
  void transform(cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c);

};

// ------------------------------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------------------------------

// Function: host
template <typename C>
void cudaGraphExec::host(cudaTask task, C&& func, void* user_data) {
  cudaHostNodeParams p {func, user_data};
  TF_CHECK_CUDA(
    cudaGraphExecHostNodeSetParams(this->get(), task._native_node, &p),
    "failed to update kernel parameters on ", task
  );
}

// Function: update kernel parameters
template <typename F, typename... ArgsT>
void cudaGraphExec::kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {
  cudaKernelNodeParams p;

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphExecKernelNodeSetParams(this->get(), task._native_node, &p),
    "failed to update kernel parameters on ", task
  );
}

// Function: update copy parameters
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
void cudaGraphExec::copy(cudaTask task, T* tgt, const T* src, size_t num) {
  auto p = cuda_get_copy_parms(tgt, src, num);
  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Function: update memcpy parameters
inline void cudaGraphExec::memcpy(
  cudaTask task, void* tgt, const void* src, size_t bytes
) {
  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Procedure: memset
inline void cudaGraphExec::memset(cudaTask task, void* dst, int ch, size_t count) {
  auto p = cuda_get_memset_parms(dst, ch, count);
  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: fill
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaGraphExec::fill(cudaTask task, T* dst, T value, size_t count) {
  auto p = cuda_get_fill_parms(dst, value, count);
  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: zero
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaGraphExec::zero(cudaTask task, T* dst, size_t count) {
  auto p = cuda_get_zero_parms(dst, count);
  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

}  // end of namespace tf -------------------------------------------------------------------------
