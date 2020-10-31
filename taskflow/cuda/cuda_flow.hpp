#pragma once

#include "cuda_task.hpp"
#include "cuda_capturer.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"
#include "cuda_algorithm/cuda_transform.hpp"

namespace tf {

/**
@brief the default number of threads per block in an 1D vector of N elements
*/
constexpr size_t cuda_default_threads_per_block(size_t N) {
  return N >= 256 ? 256 : 128;
}

// ----------------------------------------------------------------------------
// cudaFlow definition
// ----------------------------------------------------------------------------

/**
@class cudaFlow

@brief class for building a CUDA task dependency graph

A %cudaFlow is a high-level interface to manipulate GPU tasks using 
the task dependency graph model.
The class provides a set of methods for creating and launch different tasks
on one or multiple CUDA devices,
for instance, kernel tasks, data transfer tasks, and memory operation tasks.
*/
class cudaFlow {

  friend class Executor;

  public:

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;

    /**
    @brief queries the device associated with the %cudaFlow
    */
    int device() const;

    /**
    @brief queries if the cudaflow is joinable
    */
    bool joinable() const;
    
    /**
    @brief creates a no-operation task

    An empty node performs no operation during execution, 
    but can be used for transitive ordering. 
    For example, a phased execution graph with 2 groups of n nodes 
    with a barrier between them can be represented using an empty node 
    and 2*n dependency edges, 
    rather than no empty node and n^2 dependency edges.
    */
    cudaTask noop();
    
    /*
    @brief creates a host execution task
    
    @tparam C callable type
    
    @param callable a callable object with neither arguments nor return 
    (i.e., constructible from std::function<void()>)

    A host task can only execute CPU-specific functions and cannot do any CUDA calls 
    (e.g., cudaMalloc).
    */
    template <typename C>
    cudaTask host(C&& callable);
    
    /**
    @brief creates a kernel task
    
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
    
    /**
    @brief creates a kernel task on a device
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type
    
    @param d device identifier to launch the kernel
    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel_on(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

    /**
    @brief creates a memset task

    @param dst pointer to the destination device memory area
    @param v value to set for each byte of specified memory
    @param count size in bytes to set
    
    @return cudaTask handle

    A memset task fills the first @c count bytes of device memory area 
    pointed by @c dst with the byte value @c v.
    */
    cudaTask memset(void* dst, int v, size_t count);
    
    /**
    @brief creates a memcpy task
    
    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param bytes bytes to copy

    @return cudaTask handle

    A memcpy task transfers @c bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */ 
    cudaTask memcpy(void* tgt, const void* src, size_t bytes);

    /**
    @brief creates a zero task that zeroes a typed memory block

    @tparam T element type (size of @c T must be either 1, 2, or 4)
    @param dst pointer to the destination device memory area
    @param count number of elements
    
    @return cudaTask handle

    A zero task zeroes the first @c count elements of type @c T 
    in a device memory area pointed by @c dst.
    */
    template <typename T>
    std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
      cudaTask
    > 
    zero(T* dst, size_t count);

    /**
    @brief creates a fill task that fills a typed memory block with a value

    @tparam T element type (size of @c T must be either 1, 2, or 4)
    
    @param dst pointer to the destination device memory area
    @param value value to fill for each element of type @c T
    @param count number of elements
    
    @return cudaTask handle

    A fill task fills the first @c count elements of type @c T with @c value
    in a device memory area pointed by @c dst.
    The value to fill is interpreted in type @c T rather than byte.
    */
    template <typename T>
    std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
      cudaTask
    >
    fill(T* dst, T value, size_t count);
    
    /**
    @brief creates a copy task
    
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
    @brief offloads the %cudaFlow onto a GPU and repeatedly running it until 
    the predicate becomes true
    
    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the present %cudaFlow onto a GPU and
    repeatedly executes it until the predicate returns @c true.

    A offloaded %cudaFlow force the underlying graph to be instantiated.
    After the instantiation, you should not modify the graph topology
    but update node parameters.
    */
    template <typename P>
    void offload_until(P&& predicate);
    
    /**
    @brief offloads the %cudaFlow and executes it by the given times

    @param N number of executions
    */
    void offload_n(size_t N);

    /**
    @brief offloads the %cudaFlow and executes it once
    */
    void offload();

    /**
    @brief offloads the %cudaFlow with the given stop predicate and then 
    joins the execution

    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the present %cudaFlow onto a GPU 
    and repeatedly executes it until the predicate returns @c true.
    When execution finishes, the %cudaFlow is joined. 
    A joined cudaflow becomes @em invalid and cannot take other operations.
    */
    template <typename P>
    void join_until(P&& predicate);

    /**
    @brief offloads the %cudaFlow and executes it by the given times,
           and then joins the execution

    @param N number of executions before join
    */
    void join_n(size_t N);

    /**
    @brief offloads the %cudaFlow and executes it once, 
           and then joins the execution
    */
    void join();

    // ------------------------------------------------------------------------
    // update methods
    // ------------------------------------------------------------------------
  
    // TODO update_kernel_on

    /**
    @brief updates parameters of a kernel task
    */
    template <typename... ArgsT>
    void update_kernel(cudaTask ct, dim3 g, dim3 b, size_t shm, ArgsT&&... args);

    /**
    @brief updates parameters of a copy task
    */
    template <
      typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void update_copy(cudaTask ct, T* tgt, const T* src, size_t num);

    /**
    @brief updates parameters of a memcpy task
    */
    void update_memcpy(cudaTask ct, void* tgt, const void* src, size_t bytes);

    /**
    @brief updates parameters of a memset task
    */
    void update_memset(cudaTask ct, void* dst, int ch, size_t count);

    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------
    
    /**
    @brief runs a callable with only a single kernel thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    /**
    @brief applies a callable to each dereferenced element of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    
    @return cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    for(auto itr = first; itr != last; i++) {
      callable(*itr);
    }
    @endcode
    */
    template <typename I, typename C>
    cudaTask for_each(I first, I last, C&& callable);

    /**
    @brief applies a callable to each index in the range with the step size
    
    @tparam I index type
    @tparam C callable type
    
    @param first beginning index
    @param last last index
    @param step step size
    @param callable the callable to apply to each element in the data array
    
    @return cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    // step is positive [first, last)
    for(auto i=first; i<last; i+=step) {
      callable(i);
    }

    // step is negative [first, last)
    for(auto i=first; i>last; i+=step) {
      callable(i);
    }
    @endcode
    */
    template <typename I, typename C>
    cudaTask for_each_index(I first, I last, I step, C&& callable);
  
    /**
    @brief applies a callable to a source range and stores the result in a target range
    
    @tparam I iterator type
    @tparam C callable type
    @tparam S source types

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable the callable to apply to each element in the range
    @param srcs iterators to the source ranges
    
    @return cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    while (first != last) {
      *first++ = callable(*src1++, *src2++, *src3++, ...);
    }
    @endcode
    */
    template <typename I, typename C, typename... S>
    cudaTask transform(I first, I last, C&& callable, S... srcs);
    
    // TODO: 
    //template <typename T, typename B>
    //cudaTask reduce(T* tgt, size_t N, T& init, B&& op);
    //
    
    // ------------------------------------------------------------------------
    // subflow
    // ------------------------------------------------------------------------
    
    /**
    @brief constructs a subflow graph through capture

    @tparam C callable type constructible from std::function<void(tf::cudaFlowCapturer&)>
    @param callable the callable to construct a capture flow
     */
    template <typename C>
    cudaTask capture(C&& callable);

  private:
    
    //cudaFlow(int, Executor&, cudaGraph&);
    cudaFlow(Executor& executor, cudaGraph&, int);
    ~cudaFlow();
    
    Executor& _executor;
    cudaGraph& _graph;
    
    int _device {-1};

    bool _joinable {true};
    
    cudaGraphExec_t _executable {nullptr};

    void _create_executable();
    void _destroy_executable();
};

// Constructor
inline cudaFlow::cudaFlow(Executor& e, cudaGraph& g, int d) : 
  _executor{e},
  _graph   {g},
  _device  {d} {
}

// Destructor
inline cudaFlow::~cudaFlow() {
  assert(_executable == nullptr);
}

// Procedure: _create_executable
inline void cudaFlow::_create_executable() {
  assert(_executable == nullptr);
  TF_CHECK_CUDA(
    cudaGraphInstantiate(
      &_executable, _graph._native_handle, nullptr, nullptr, 0
    ),
    "failed to create an executable graph"
  );
}

// Procedure: _destroy_executable
inline void cudaFlow::_destroy_executable() {
  assert(_executable != nullptr);
  TF_CHECK_CUDA(
    cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
  );
  _executable = nullptr;
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Function: joinable
inline bool cudaFlow::joinable() const {
  return _joinable;
} 

// Function: device
inline int cudaFlow::device() const {
  return _device;
}

// Function: noop
inline cudaTask cudaFlow::noop() {

  auto node = _graph.emplace_back( 
    _graph, std::in_place_type_t<cudaNode::Empty>{}
  );

  TF_CHECK_CUDA(
    cudaGraphAddEmptyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0
    ),
    "failed to create a no-operation (empty) node"
  );

  return cudaTask(node);
}

// Function: host
template <typename C>
cudaTask cudaFlow::host(C&& c) {
  
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Host>{}, std::forward<C>(c)
  );

  auto& h = std::get<cudaNode::Host>(node->_handle);

  cudaHostNodeParams p;
  p.fn = cudaNode::Host::callback;
  p.userData = &h;

  TF_CHECK_CUDA(
    ::cudaGraphAddHostNode(&node->_native_handle, _graph._native_handle, nullptr, 0, &p),
    "failed to create a host node"
  );
  
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
  );
  
  cudaKernelNodeParams p;
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphAddKernelNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda kernel task"
  );

  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel_on(
  int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
  
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
  );
  
  cudaKernelNodeParams p;
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  cudaScopedDevice ctx(d);
  TF_CHECK_CUDA(
    ::cudaGraphAddKernelNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda kernel task on device ", d
  );

  return cudaTask(node);
}

// Function: zero
template <typename T>
std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
  cudaTask
> 
cudaFlow::zero(T* dst, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  cudaMemsetParams p;
  p.dst = dst;
  p.value = 0;
  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset (zero) task"
  );

  return cudaTask(node);
}
    
// Function: fill
template <typename T>
std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
  cudaTask
>
cudaFlow::fill(T* dst, T value, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  cudaMemsetParams p;
  p.dst = dst;

  // perform bit-wise copy
  p.value = 0;  // crucial
  static_assert(sizeof(T) <= sizeof(p.value), "internal error");
  std::memcpy(&p.value, &value, sizeof(T));

  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;
  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset (fill) task"
  );

  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );
  
  cudaMemcpy3DParms p;
  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
  p.kind = cudaMemcpyDefault;

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memcpy (copy) task"
  );

  return cudaTask(node);
}

// Function: memset
inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  cudaMemsetParams p;
  p.dst = dst;
  p.value = ch;
  p.pitch = 0;
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
  p.elementSize = 1;  // either 1, 2, or 4
  p.width = count;
  p.height = 1;
  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset task"
  );
  
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );
  
  // Parameters in cudaPitchedPtr
  // d   - Pointer to allocated memory
  // p   - Pitch of allocated memory in bytes
  // xsz - Logical width of allocation in elements
  // ysz - Logical height of allocation in elements
  cudaMemcpy3DParms p;
  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_cudaExtent(bytes, 1, 1);
  p.kind = cudaMemcpyDefault;

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memcpy task"
  );

  return cudaTask(node);
}

// ------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------

// Function: update kernel parameters
template <typename... ArgsT>
void cudaFlow::update_kernel(
  cudaTask ct, dim3 g, dim3 b, size_t s, ArgsT&&... args
) {

  if(ct.type() != CUDA_KERNEL_TASK) {
    TF_THROW(ct, " is not a kernel task");
  }

  cudaKernelNodeParams p;
  
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = std::get<cudaNode::Kernel>((ct._node)->_handle).func;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;
  
  //TF_CHECK_CUDA(
  //  cudaGraphKernelNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of kernel task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecKernelNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update kernel parameter on ", ct
  );
} 

// Function: update copy parameters
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void cudaFlow::update_copy(cudaTask ct, T* tgt, const T* src, size_t num) {
  
  if(ct.type() != CUDA_MEMCPY_TASK) {
    TF_THROW(ct, " is not a memcpy task");
  }

  using U = std::decay_t<T>;

  cudaMemcpy3DParms p;

  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
  p.kind = cudaMemcpyDefault;

  //TF_CHECK_CUDA(
  //  cudaGraphMemcpyNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memcpy task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memcpy parameter on ", ct
  );
}

// Function: update memcpy parameters
inline void cudaFlow::update_memcpy(
  cudaTask ct, void* tgt, const void* src, size_t bytes
) {
  
  if(ct.type() != CUDA_MEMCPY_TASK) {
    TF_THROW(ct, " is not a memcpy task");
  }

  cudaMemcpy3DParms p;

  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_cudaExtent(bytes, 1, 1);
  p.kind = cudaMemcpyDefault;

  //TF_CHECK_CUDA(
  //  cudaGraphMemcpyNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memcpy task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(_executable, ct._node->_native_handle, &p),
    "failed to update memcpy parameter on ", ct
  );
}

inline
void cudaFlow::update_memset(cudaTask ct, void* dst, int ch, size_t count) {

  if(ct.type() != CUDA_MEMSET_TASK) {
    TF_THROW(ct, " is not a memset task");
  }

  cudaMemsetParams p;
  p.dst = dst;
  p.value = ch;
  p.pitch = 0;
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
  p.elementSize = 1;  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  //TF_CHECK_CUDA(
  //  cudaGraphMemsetNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memset task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memset parameter on ", ct
  );
}

// ----------------------------------------------------------------------------
// Generic Algorithm API
// ----------------------------------------------------------------------------
    
// Function: single_task
template <typename C>
cudaTask cudaFlow::single_task(C&& c) {
  return kernel(
    1, 1, 0, cuda_single_task<C>, std::forward<C>(c)
  );
}

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlow::for_each(I first, I last, C&& c) {
  
  size_t N = std::distance(first, last);
  size_t B = cuda_default_threads_per_block(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each<I, C>, first, N, std::forward<C>(c)
  );
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlow::for_each_index(I beg, I end, I inc, C&& c) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?

  size_t N = distance(beg, end, inc);
  size_t B = cuda_default_threads_per_block(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each_index<I, C>, beg, inc, N, std::forward<C>(c)
  );
}

// Function: transform
template <typename I, typename C, typename... S>
cudaTask cudaFlow::transform(I first, I last, C&& c, S... srcs) {
  
  // TODO: special case when N is 0?
  
  size_t N = std::distance(first, last);
  size_t B = cuda_default_threads_per_block(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_transform<I, C, S...>, 
    first, N, std::forward<C>(c), srcs...
  );
}

//template <typename T, typename B>>
//cudaTask cudaFlow::reduce(T* tgt, size_t N, T& init, B&& op) {
  //if(N == 0) {
    //return noop();
  //}
  //size_t B = cuda_default_threads_per_block(N);
//}

// ----------------------------------------------------------------------------
// cudaFlowCapturer
// ----------------------------------------------------------------------------

// Function: capture
template <typename C>
cudaTask cudaFlow::capture(C&& c) {

  // insert a subflow node
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Subflow>{}
  );
  
  // construct a capturer flow from the callable
  auto& node_handle = std::get<cudaNode::Subflow>(node->_handle);
  cudaFlowCapturer capturer(node_handle.graph);

  c(capturer);
  
  // obtain the optimized captured graph
  auto captured = capturer._capture();
  //cuda_dump_graph(std::cout, captured);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, captured
    ), 
    "failed to add a cudaFlow capturer task"
  );
  
  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");

  return cudaTask(node);
}


}  // end of namespace tf -----------------------------------------------------


