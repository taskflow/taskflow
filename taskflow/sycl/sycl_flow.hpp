#pragma once

#include "sycl_task.hpp"

/** 
@file sycl_flow.hpp
@brief syclFlow include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: syclFlow
// ----------------------------------------------------------------------------

/**
@class syclFlow

@brief class for building a SYCL task dependency graph

*/
class syclFlow {

  friend class Executor;

  struct External {
    syclGraph graph;
  };

  struct Internal {
    Executor& executor;
    Internal(Executor& e) : executor {e} {}
  };

  using handle_t = std::variant<External, Internal>;



  public:

    /**
    @brief constructs a standalone %syclFlow from the given queue

    A standalone %syclFlow does not go through any taskflow and
    can be run by the caller thread using explicit offload methods 
    (e.g., tf::syclFlow::offload).
    */
    syclFlow(sycl::queue& queue);
    
    /**
    @brief destroys the %syclFlow 
     */
    ~syclFlow() = default;

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;

    /**
    @brief queries the number of tasks
    */
    size_t num_tasks() const;
    
    /**
    @brief dumps the %syclFlow graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;

    /**
    @brief acquires the underlying queue
    */
    sycl::queue& queue();
    
    /**
    @brief clear the associated graph
    */
    void clear();

    // ------------------------------------------------------------------------
    // Generic device operations
    // ------------------------------------------------------------------------

    /**
    @brief creates a task that launches the given command group function object

    @tparam F type of command group function object
    @param func function object that is constructible from 
                std::function<void(sycl::handler&)>
    
    Creates a task that is associated from the given command group.
    In SYCL, each command group function object is given a unique 
    command group handler object to perform all the necessary work 
    required to correctly process data on a device using a kernel.
    */
    template <typename F>
    syclTask on(F&& func);
    
    /**
    @brief creates a memcpy task that copies untyped data in bytes
    
    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param bytes bytes to copy

    @return a tf::syclTask handle

    A memcpy task transfers @c bytes of data from a source locationA @c src
    to a target location @c tgt. Both @c src and @c tgt may be either host 
    or USM pointers.
    */ 
    syclTask memcpy(void* tgt, const void* src, size_t bytes);
    
    /**
    @brief creates a memset task that fills untyped data with a byte value

    @param ptr pointer to the destination device memory area
    @param value value to set for each byte of specified memory
    @param bytes number of bytes to set
    
    @return a tf::syclTask handle

    Fills @c bytes of memory beginning at address @c ptr with @c value. 
    @c ptr must be a USM allocation. 
    @c value is interpreted as an unsigned char.
    */
    syclTask memset(void* ptr, int value, size_t bytes);
    
    /**
    @brief creates a fill task that fills typed data with the given value

    @tparam T trivially copyable value type

    @param ptr pointer to the memory to fill
    @param pattern pattern value to fill into the memory
    @param count number of items to fill the value

    Creates a task that fills the specified memory with the 
    specified value.
    */
    template <typename T>
    syclTask fill(void* ptr, const T& pattern, size_t count);
    
    /**
    @brief creates a copy task that copies typed data from a source to a target
           memory block

    @tparam T trivially copyable value type
    
    @param target pointer to the memory to fill
    @param source pointer to the pattern value to fill into the memory
    @param count number of items to fill the value
    
    Creates a task that copies @c count items of type @c T from a source memory
    location to a target memory location.
    */
    template <typename T,
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    syclTask copy(T* target, const T* source, size_t count);
    
    /**
    @brief creates a kernel task

    @tparam ArgsT arguments types

    @param args arguments to forward to the parallel_for methods defined 
                in the handler object

    Creates a kernel task from a parallel_for method through the handler 
    object associated with a command group.
    */
    template <typename...ArgsT>
    syclTask parallel_for(ArgsT&&... args);
    
    // ------------------------------------------------------------------------
    // algorithms
    // ------------------------------------------------------------------------
    
    /**
    @brief invokes a SYCL kernel function using only one thread

    @tparam F kernel function type
    @param func kernel function

    Creates a task that launches the given function object using only one
    kernel thread. 
    */
    template <typename F>
    syclTask single_task(F&& func);
    
    /**
    @brief applies a callable to each dereferenced element of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    
    @return a tf::syclTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    for(auto itr = first; itr != last; itr++) {
      callable(*itr);
    }
    @endcode
    */
    template <typename I, typename C>
    syclTask for_each(I first, I last, C&& callable);
    
    /**
    @brief applies a callable to each index in the range with the step size
    
    @tparam I index type
    @tparam C callable type
    
    @param first beginning index
    @param last last index
    @param step step size
    @param callable the callable to apply to each element in the data array
    
    @return a tf::syclTask handle
    
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
    syclTask for_each_index(I first, I last, I step, C&& callable);
    
    /**
    @brief applies a callable to a source range and stores the result in a target range
    
    @tparam I iterator type
    @tparam C callable type
    @tparam S source types

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable the callable to apply to each element in the range
    @param srcs iterators to the source ranges
    
    @return a tf::syclTask handle
    
    This method is equivalent to the parallel execution of the following 
    loop on a SYCL device:
    
    @code{.cpp}
    while (first != last) {
      *first++ = callable(*src1++, *src2++, *src3++, ...);
    }
    @endcode
    */
    template <typename I, typename C, typename... S>
    syclTask transform(I first, I last, C&& callable, S... srcs);
    
    /**
    @brief performs parallel reduction over a range of items
    
    @tparam I input iterator type
    @tparam T value type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param result pointer to the result with an initialized value
    @param op binary reduction operator
    
    @return a tf::syclTask handle
    
    This method is equivalent to the parallel execution of the following loop 
    on a SYCL device:
    
    @code{.cpp}
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    syclTask reduce(I first, I last, T* result, C&& op);
    
    /**
    @brief similar to tf::syclFlow::reduce but does not assume any initial
           value to reduce
    
    This method is equivalent to the parallel execution of the following loop 
    on a SYCL device:
    
    @code{.cpp}
    *result = *first++;  // no initial values partitipcate in the loop
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    syclTask uninitialized_reduce(I first, I last, T* result, C&& op);
    
    // ------------------------------------------------------------------------
    // offload methods
    // ------------------------------------------------------------------------

    /**
    @brief offloads the %syclFlow onto a GPU and repeatedly runs it until 
    the predicate becomes true
    
    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Repetitively executes the present %syclFlow through the given queue object
    until the predicate returns @c true.

    By default, if users do not offload the %syclFlow, 
    the executor will offload it once.
    */
    template <typename P>
    void offload_until(P&& predicate);
    
    /**
    @brief offloads the %syclFlow and executes it by the given times

    @param N number of executions
    */
    void offload_n(size_t N);

    /**
    @brief offloads the %syclFlow and executes it once
    */
    void offload();
    
    // ------------------------------------------------------------------------
    // rebind methods
    // ------------------------------------------------------------------------
    
    /**
    @brief rebinds the task to the given command group function object

    Similar to tf::syclFlow::on but operates on an existing task.
    */
    template <typename F>
    void rebind_on(syclTask task, F&& func);

    /**
    @brief rebinds the task to a memcpy task
    
    Similar to tf::syclFlow::memcpy but operates on an existing task.
    */
    void rebind_memcpy(syclTask task, void* tgt, const void* src, size_t bytes);
    
    /**
    @brief rebinds the task to a memset task
    
    Similar to tf::syclFlow::memset but operates on an existing task.
    */
    void rebind_memset(syclTask task, void* ptr, int value, size_t bytes);
    
    /**
    @brief rebinds the task to a fill task

    Similar to tf::syclFlow::fill but operates on an existing task.
    */
    template <typename T>
    void rebind_fill(syclTask task, void* ptr, const T& pattern, size_t count);
    
    /**
    @brief rebinds the task to a copy task

    Similar to tf::syclFlow::copy but operates on an existing task.
    */
    template <typename T,
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void rebind_copy(syclTask task, T* target, const T* source, size_t count);
    
    /**
    @brief rebinds the task to a parallel-for kernel task
    
    Similar to tf::syclFlow::parallel_for but operates on an existing task.
    */
    template <typename...ArgsT>
    void rebind_parallel_for(syclTask task, ArgsT&&... args);

    /**
    @brief rebinds the task to a single-threaded kernel task

    Similar to tf::syclFlow::single_task but operates on an existing task.
    */
    template <typename F>
    void rebind_single_task(syclTask task, F&& func);
    
    /**
    @brief rebinds the task to a for-each task

    Similar to tf::syclFlow::for_each but operates on an existing task.
    */
    template <typename I, typename C>
    void rebind_for_each(syclTask task, I first, I last, C&& callable);
    
    /**
    @brief rebinds the task to a for-each-index task

    Similar to tf::syclFlow::for_each_index but operates on an existing task.
     */
    template <typename I, typename C>
    void rebind_for_each_index(
      syclTask task, I first, I last, I step, C&& callable
    );
    
    /**
    @brief rebinds the task to a transform task
    
    Similar to tf::syclFlow::transform but operates on an existing task.
     */
    template <typename I, typename C, typename... S>
    void rebind_transform(
      syclTask task, I first, I last, C&& callable, S... srcs
    );
    
    /**
    @brief rebinds the task to a reduce task

    Similar to tf::syclFlow::reduce but operates on an existing task.
    */
    template <typename I, typename T, typename C>
    void rebind_reduce(
      syclTask task, I first, I last, T* result, C&& op
    );
    
    /**
    @brief rebinds the task to an unitialized reduce task

    Similar to tf::syclFlow::uninitialized_reduce but operates on an existing task.
    */
    template <typename I, typename T, typename C>
    void rebind_uninitialized_reduce(
      syclTask task, I first, I last, T* result, C&& op
    );

  private:

    syclFlow(Executor&, syclGraph&, sycl::queue&);
    
    const size_t _MAX_WORK_GROUP_SIZE;

    handle_t _handle;
    
    syclGraph& _graph;

    sycl::queue& _queue;
  
    std::vector<syclNode*> _tpg;
    std::queue<syclNode*> _bfs;

    size_t _default_group_size(size_t N) const;

    template <typename I, typename C>
    auto _for_each_cgh(I, I, C&&);

    template <typename I, typename C>
    auto _for_each_index_cgh(I, I, I, C&&);

    template <typename I, typename T, typename C, bool>
    auto _reduce_cgh(I, I, T*, C&&);

    template <typename I, typename C, typename... S>
    auto _transform_cgh(I, I, C&&, S...);
};

// constructor
inline syclFlow::syclFlow(sycl::queue& queue) :
  _MAX_WORK_GROUP_SIZE {
    queue.get_device().get_info<sycl::info::device::max_work_group_size>()
  },
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get<External>(_handle).graph},
  _queue  {queue} {
}

// Construct the syclFlow from executor (internal graph)
inline syclFlow::syclFlow(Executor& e, syclGraph& g, sycl::queue& queue) :
  _MAX_WORK_GROUP_SIZE {
    queue.get_device().get_info<sycl::info::device::max_work_group_size>()
  } ,
  _handle {std::in_place_type_t<Internal>{}, e},
  _graph  {g},
  _queue  {queue} {
}

// Function: _default_group_size
inline size_t syclFlow::_default_group_size(size_t N) const {
  return N <= 32u ? 32u : std::min(_MAX_WORK_GROUP_SIZE, next_pow2(N));
}

// Function: empty
inline bool syclFlow::empty() const {
  return _graph._nodes.empty();
}

// Function: num_tasks
inline size_t syclFlow::num_tasks() const {
  return _graph._nodes.size();
}

// Procedure: dump
inline void syclFlow::dump(std::ostream& os) const {
  _graph.dump(os, nullptr, "");
}

// Function: queue
inline sycl::queue& syclFlow::queue() {
  return _queue;
}

// Procedure: clear
inline void syclFlow::clear() {
  _graph.clear();
}

// Function: memcpy
inline syclTask syclFlow::memcpy(void* tgt, const void* src, size_t bytes) {

  auto node = _graph.emplace_back(_graph, [=](sycl::handler& h){
    h.memcpy(tgt, src, bytes);
  });
  
  return syclTask(node);
}

// Function: memset
inline syclTask syclFlow::memset(void* ptr, int value, size_t bytes) {

  auto node = _graph.emplace_back(_graph, [=](sycl::handler& h){
    h.memset(ptr, value, bytes);
  });
  
  return syclTask(node);
}

// Function: fill
template <typename T>
syclTask syclFlow::fill(void* ptr, const T& pattern, size_t count) {

  auto node = _graph.emplace_back(_graph, [=](sycl::handler& h){
    h.fill(ptr, pattern, count);
  });
  
  return syclTask(node);
}

// Function: copy
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
syclTask syclFlow::copy(T* target, const T* source, size_t count) {
  auto node = _graph.emplace_back(_graph, [=](sycl::handler& h){
    h.memcpy(target, source, count*sizeof(T));
  });
  return syclTask(node);
}

// Function: on
template <typename F>
syclTask syclFlow::on(F&& func) {
  auto node = _graph.emplace_back(_graph, std::forward<F>(func));
  return syclTask(node);
}

// Function: single_task
template <typename F>
syclTask syclFlow::single_task(F&& func) {
  auto node = _graph.emplace_back(_graph,
    [f=std::forward<F>(func)] (sycl::handler& h) mutable {
      h.single_task(f);
    }
  );
  return syclTask(node);
}

// Function: parallel_for
template <typename...ArgsT>
syclTask syclFlow::parallel_for(ArgsT&&... args) {
  auto node = _graph.emplace_back(_graph,
    [args...] (sycl::handler& h) mutable {
      h.parallel_for(args...);
    }
  );
  return syclTask(node);
}

// Procedure: offload_until
template <typename P>
void syclFlow::offload_until(P&& predicate) {
  
  if(!(_graph._state & syclGraph::TOPOLOGY_CHANGED)) {
    goto offload;
  }

  // levelize the graph
  _tpg.clear();

  // insert the first level of nodes into the queue
  for(auto& u : _graph._nodes) {
    u->_level = u->_dependents.size();
    if(u->_level == 0) {
      _bfs.push(u.get());
    }
  }
  
  while(!_bfs.empty()) {
    auto u = _bfs.front();
    _bfs.pop();
    _tpg.push_back(u);
    for(auto v : u->_successors) {
      if(--(v->_level) == 0) {
        v->_level = u->_level + 1;
        _bfs.push(v);
      }
    }
  }

  offload:
  
  // offload the syclFlow graph
  bool in_order = _queue.is_in_order();
  
  while(!predicate()) {

    // traverse node in a topological order
    for(auto u : _tpg) {
      u->_event = _queue.submit([u, in_order](sycl::handler& handler){
        // wait on all predecessors
        if(!in_order) {
          for(auto p : u->_dependents) {
            handler.depends_on(p->_event);
          }
        }
        u->_func(handler);
      });      
    }
    
    // synchronize the execution
    _queue.wait();
  }

  _graph._state |= syclGraph::OFFLOADED;
  _graph._state &= ~syclGraph::TOPOLOGY_CHANGED;
}

// Procedure: offload_n
inline void syclFlow::offload_n(size_t n) {
  offload_until([repeat=n] () mutable { return repeat-- == 0; });
}

// Procedure: offload
inline void syclFlow::offload() {
  offload_until([repeat=1] () mutable { return repeat-- == 0; });
}

// Function: rebind_on
template <typename F>
void syclFlow::rebind_on(syclTask task, F&& func) {
  task._node->_func = std::forward<F>(func);
}
    
// Function: rebind_memcpy
inline void syclFlow::rebind_memcpy(
  syclTask task, void* tgt, const void* src, size_t bytes
) {
  task._node->_func = [=](sycl::handler& h){ h.memcpy(tgt, src, bytes); };
}

// Function: rebind_memset
inline void syclFlow::rebind_memset(
  syclTask task, void* ptr, int value, size_t bytes
) {
  task._node->_func = [=](sycl::handler& h){ h.memset(ptr, value, bytes); };
}

// Function: rebind_fill
template <typename T>
void syclFlow::rebind_fill(
  syclTask task, void* ptr, const T& pattern, size_t count
) {
  task._node->_func = [=](sycl::handler& h){ h.fill(ptr, pattern, count); };
}

// Function: rebind_copy
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void syclFlow::rebind_copy(
  syclTask task, T* target, const T* source, size_t count
) {
  task._node->_func = [=](sycl::handler& h) { 
    h.memcpy(target, source, count*sizeof(T)); 
  };
}

// Function: parallel_for
template <typename...ArgsT>
void syclFlow::rebind_parallel_for(syclTask task, ArgsT&&... args) {
  task._node->_func = [args...] (sycl::handler& h) mutable {
    h.parallel_for(args...);
  };
}
    
// Function: single_task
template <typename F>
void syclFlow::rebind_single_task(syclTask task, F&& func) {
  task._node->_func = [f=std::forward<F>(func)](sycl::handler& h) {
    h.single_task(f);
  };
}

// ############################################################################
// Forward declaration: FlowBuilder
// ############################################################################
    
// FlowBuilder::emplace_on
template <typename C, typename Q, 
  std::enable_if_t<is_syclflow_task_v<C>, void>*
>
Task FlowBuilder::emplace_on(C&& callable, Q& queue) {
  auto n = _graph.emplace_back(
    std::in_place_type_t<Node::syclFlow>{},
    [c=std::forward<C>(callable), &queue] (Executor& e, Node* p) mutable {
      e._invoke_syclflow_task_entry(p, c, queue);
    },
    std::make_unique<syclGraph>()
  );
  return Task(n);
}

// ############################################################################
// Forward declaration: Executor
// ############################################################################

// Procedure: _invoke_syclflow_task_entry (syclFlow)
template <typename C, typename Q,
  std::enable_if_t<is_syclflow_task_v<C>, void>*
>
void Executor::_invoke_syclflow_task_entry(Node* node, C&& c, Q& queue) {

  auto& h = std::get<Node::syclFlow>(node->_handle);

  syclGraph* g = dynamic_cast<syclGraph*>(h.graph.get());
  
  g->clear();

  syclFlow sf(*this, *g, queue);

  c(sf); 

  if(!(g->_state & syclGraph::OFFLOADED)) {
    sf.offload();
  }
}

}  // end of namespace tf -----------------------------------------------------
    

