#pragma once

#include "cuda_task.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"

namespace tf {

/**
@class cudaFlowCapturerBase

@brief base class of methods to capture CUDA operations through
       CUDA streams
*/
class cudaFlowCapturerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief default constructor
     */
    cudaFlowCapturerBase() = default;

    /**
    @brief default virtual destructor
     */
    virtual ~cudaFlowCapturerBase() = default;
  
    /**
    @brief runs a callable with only a single kernel thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    /**
    @brief captures a sequential CUDA operations from the given callable
    
    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
    @param callable a callable to capture CUDA operations with the stream

    This methods applies a stream created by the flow to capture 
    a sequence of CUDA operations defined in the callable.
    */
    template <typename C>
    cudaTask on(C&& callable);
    
    /**
    @brief copies data between host and device asynchronously through a stream
    
    @param dst destination memory address
    @param src source memory address
    @param count size in bytes to copy
    
    The method captures a @c cudaMemcpyAsync operation through an 
    internal stream.
    */ 
    cudaTask memcpy(void* dst, const void* src, size_t count);

    /**
    @brief captures a copy task of typed data
    
    @tparam T element type (non-void)

    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param num number of elements to copy

    @return cudaTask handle

    A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */
    template <typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    cudaTask copy(T* tgt, const T* src, size_t num);

    /**
    @brief initializes or sets GPU memory to the given value byte by byte
    
    @param devPtr pointer to GPU mempry
    @param value value to set for each byte of the specified memory
    @param count size in bytes to set
    
    The method captures a @c cudaMemsetAsync operation through an
    internal stream to fill the first @c count bytes of the memory area 
    pointed to by @c devPtr with the constant byte value @c value.
    */ 
    cudaTask memset(void* devPtr, int value, size_t count);

    /**
    @brief captures a kernel
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */ 
    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

  private:

    cudaGraph* _graph {nullptr};

    cudaFlowCapturerBase(cudaGraph&);
};

// Constructor
inline cudaFlowCapturerBase::cudaFlowCapturerBase(cudaGraph& g) :
  _graph {&g} {
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturerBase::single_task(C&& callable) {
  return on([c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
  });
}

// Function: capture
template <typename C>
cudaTask cudaFlowCapturerBase::on(C&& callable) {
  auto node = _graph->emplace_back(*_graph,
    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlowCapturerBase::memcpy(
  void* dst, const void* src, size_t count
) {
  return on([dst, src, count] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}
    
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
cudaTask cudaFlowCapturerBase::copy(T* tgt, const T* src, size_t num) {
  return on([tgt, src, num] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline cudaTask cudaFlowCapturerBase::memset(void* ptr, int v, size_t n) {
  return on([ptr, v, n] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}
    
// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlowCapturerBase::kernel(
  dim3 g, dim3 b, size_t shm, F&& f, ArgsT&&... args
) {
  return on([g, b, shm, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, shm, stream>>>(args...);
  });
}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

/**
@class cudaFlowCapturer

@brief class object to construct a CUDA graph through stream capture

A %cudaFlowCapturer inherits all the base methods from tf::cudaFlowCapturerBase 
to construct a CUDA graph through stream capturer. 
This class also defines a factory interface tf::cudaFlowCapturer::make_capturer 
for users to create custom capturers and manages their lifetimes.

*/
class cudaFlowCapturer : public cudaFlowCapturerBase {

  friend class cudaFlow;
  friend class Executor;

  public:
    
    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;
    
    /**
    @brief creates a custom capturer derived from tf::cudaFlowCapturerBase

    @tparam T custom capturer type
    @tparam ArgsT arguments types

    @param args arguments to forward to construct the custom capturer

    This %cudaFlowCapturer object keeps a factory of created custom capturers
    and manages their lifetimes.
     */
    template <typename T, typename... ArgsT>
    T* make_capturer(ArgsT&&... args);

  private:
    
    std::vector<std::unique_ptr<cudaFlowCapturerBase>> _capturers;

    cudaFlowCapturer(cudaGraph&);
    
    cudaGraph_t _capture();
    
    cudaGraphExec_t _executable {nullptr};
    
    //void _create_executable();
    //void _destroy_executable();
};

// constructor
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
  cudaFlowCapturerBase{g} {
}

// Function: empty
inline bool cudaFlowCapturer::empty() const {
  return _graph->empty();
}

//// Procedure: _create_executable
//inline void cudaFlowCapturer::_create_executable() {
//  assert(_executable == nullptr);
//  TF_CHECK_CUDA(
//    cudaGraphInstantiate(
//      &_executable, _graph->_native_handle, nullptr, nullptr, 0
//    ),
//    "failed to create an executable graph"
//  );
//}
//
//// Procedure: _destroy_executable
//inline void cudaFlowCapturer::_destroy_executable() {
//  assert(_executable != nullptr);
//  TF_CHECK_CUDA(
//    cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
//  );
//  _executable = nullptr;
//}

// Function: make_capturer
template <typename T, typename... ArgsT>
T* cudaFlowCapturer::make_capturer(ArgsT&&... args) {

  static_assert(std::is_base_of_v<cudaFlowCapturerBase, T>);

  auto ptr = std::make_unique<T>(std::forward<ArgsT>(args)...);
  ptr->_graph = this->_graph;
  auto raw = ptr.get();
  _capturers.push_back(std::move(ptr));
  return raw;
}

// Procedure
inline cudaGraph_t cudaFlowCapturer::_capture() {

  // acquire per-thread stream and turn it into capture mode
  cudaScopedPerThreadStream stream;
  start_stream_capture(stream);

  // TODO: need an efficient algorithm
  auto ordered = _graph->_toposort();
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  auto g = cease_stream_capture(stream);

  //cuda_dump_graph(std::cout, g);
  
  return g;
}

}  // end of namespace tf -----------------------------------------------------

