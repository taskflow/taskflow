#pragma once

#include "cuda_graph.hpp"

namespace tf {

/**
@struct is_cudaflow_task

@brief determines if a callable is a cudaflow task
*/
template <typename C>
constexpr bool is_cudaflow_task_v = is_invocable_r_v<void, C, cudaFlow&>;

/**
@class cudaTask

@brief handle to a node in a cudaGraph
*/
class cudaTask {

  friend class cudaFlow;

  public:
    
    /**
    @brief constructs an empty cudaTask
    */
    cudaTask() = default;

    /**
    @brief copy-constructs a cudaTask
    */
    cudaTask(const cudaTask&) = default;

    /**
    @brief copy-assigns a cudaTask
    */
    cudaTask& operator = (const cudaTask&) = default;

    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts... parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& precede(Ts&&... tasks);
    
    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts... parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& succeed(Ts&&... tasks);
    
    /**
    @brief assigns a name to the task

    @param name a @std_string acceptable string

    @return @c *this
    */
    cudaTask& name(const std::string& name);
    
    /**
    @brief queries the name of the task
    */
    const std::string& name() const;

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries if the task is associated with a cudaNode
    */
    bool empty() const;

  private:
    
    cudaTask(cudaNode*);

    cudaNode* _node {nullptr};
    
    /// @private
    template <typename T>
    void _precede(T&&);

    /// @private
    template <typename T, typename... Ts>
    void _precede(T&&, Ts&&...);
    
    /// @private
    template <typename T>
    void _succeed(T&&);

    // @private
    template <typename T, typename... Ts>
    void _succeed(T&&, Ts&&...);
};

// Constructor
inline cudaTask::cudaTask(cudaNode* node) : _node {node} {
}

// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  _precede(std::forward<Ts>(tasks)...);
  return *this;
}

/// @private
// Procedure: precede
template <typename T>
void cudaTask::_precede(T&& other) {
  _node->_precede(other._node);
}

/// @private
// Procedure: _precede
template <typename T, typename... Ts>
void cudaTask::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
}

// Function: succeed
template <typename... Ts>
cudaTask& cudaTask::succeed(Ts&&... tasks) {
  _succeed(std::forward<Ts>(tasks)...);
  return *this;
}

/// @private
// Procedure: _succeed
template <typename T>
void cudaTask::_succeed(T&& other) {
  other._node->_precede(_node);
}

/// @private
// Procedure: _succeed
template <typename T, typename... Ts>
void cudaTask::_succeed(T&& task, Ts&&... others) {
  _succeed(std::forward<T>(task));
  _succeed(std::forward<Ts>(others)...);
}

// Function: empty
inline bool cudaTask::empty() const {
  return _node == nullptr;
}

// Function: name
inline cudaTask& cudaTask::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& cudaTask::name() const {
  return _node->_name;
}

// Function: num_successors
inline size_t cudaTask::num_successors() const {
  return _node->_successors.size();
}

//// Function: kernel
//template <typename F, typename... ArgsT>
//cudaTask& cudaTask::kernel(
//  dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args
//) {
//
//  using traits = function_traits<F>;
//
//  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
//
//  void* arguments[sizeof...(ArgsT)] = { &args... };
//
//  auto& p = _node->_handle.emplace<cudaNode::Kernel>().param;
//
//  p.func = (void*)func;
//  p.gridDim = grid;
//  p.blockDim = block;
//  p.sharedMemBytes = shm;
//  p.kernelParams = arguments;
//  p.extra = nullptr;
//  
//  TF_CHECK_CUDA(
//    ::cudaGraphAddKernelNode(&_node->_node, _node->_graph._handle, nullptr, 0, &p),
//    "failed to create a cudaKernel node"
//  );
//  
//  return *this;
//}
//
//// Function: copy
//template <
//  typename T,
//  std::enable_if_t<!std::is_same<T, void>::value, void>*
//>
//cudaTask& cudaTask::copy(T* tgt, T* src, size_t num) {
//
//  using U = std::decay_t<T>;
//
//  auto& p = _node->_handle.emplace<cudaNode::Copy>().param;
//
//  p.srcArray = nullptr;
//  p.srcPos = ::make_cudaPos(0, 0, 0);
//  p.srcPtr = ::make_cudaPitchedPtr(src, num*sizeof(U), num, 1);
//  p.dstArray = nullptr;
//  p.dstPos = ::make_cudaPos(0, 0, 0);
//  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
//  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
//  p.kind = cudaMemcpyDefault;
//
//  TF_CHECK_CUDA(
//    cudaGraphAddMemcpyNode(&_node->_node, _node->_graph._handle, nullptr, 0, &p),
//    "failed to create a cudaCopy node"
//  );
//
//  return *this;
//}

}  // end of namespace tf -----------------------------------------------------
