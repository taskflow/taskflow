#pragma once

#include "task.hpp"

namespace tf {

/** 
@class FlowBuilder

@brief Building methods of a task dependency graph.

*/
class FlowBuilder {

  friend class Task;

  public:
    
    /**
    @brief constructs a flow builder object

    @param graph a task dependency graph to manipulate
    */
    FlowBuilder(Graph& graph);
    
    /**
    @brief creates a static task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function<void()>

    @return Task handle
    */
    template <typename C>
    std::enable_if_t<is_static_task_v<C>, Task> emplace(C&& callable);

    /**
    @brief creates a dynamic task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function<void(Subflow&)>

    @return Task handle
    */
    template <typename C>
    std::enable_if_t<is_dynamic_task_v<C>, Task> emplace(C&& callable);

    /**
    @brief creates a condition task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function<int()>

    @return Task handle
    */
    template <typename C>
    std::enable_if_t<is_condition_task_v<C>, Task> emplace(C&& callable);

#ifdef TF_ENABLE_CUDA
    /**
    @brief creates a cudaflow task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function<void(cudaFlow&)>

    @return Task handle
    */
    template <typename C>
    std::enable_if_t<is_cudaflow_task_v<C>, Task> emplace(C&& callable);
#endif 

    /**
    @brief creates multiple tasks from a list of callable objects at one time
    
    @tparam C... callable types

    @param callables one or multiple callable objects acceptable to std::function

    @return a Task handle
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&... callables);

    /**
    @brief creates a module task from a taskflow

    @param taskflow a taskflow object for the module
    @return a Task handle
    */
    Task composed_of(Taskflow& taskflow);
    
    /**
    @brief constructs a task dependency graph of range-based parallel_for
    
    The task dependency graph applies a callable object 
    to the dereferencing of every iterator 
    in the range [beg, end) chunk by chunk.

    @tparam I input iterator type
    @tparam C callable type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to be applied to 
    @param chunk size (default 1)

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename C>
    std::pair<Task, Task> parallel_for(I beg, I end, C&& callable, size_t chunk=1);
    
    /**
    @brief constructs a task dependency graph of integer index-based parallel_for
    
    The task dependency graph applies a callable object to every index 
    in the range [beg, end) with a step size chunk by chunk.

    @tparam I integer (arithmetic) index type
    @tparam C callable type

    @param beg index of the beginning (inclusive)
    @param end index of the end (exclusive)
    @param step step size 
    @param callable a callable object to be applied to
    @param chunk items per task

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <
      typename I, 
      typename C, 
      std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>* = nullptr
    >
    std::pair<Task, Task> parallel_for(
      I beg, I end, I step, C&& callable, size_t chunk = 1
    );

    /**
    @brief constructs a task dependency graph of floating index-based parallel_for
    
    The task dependency graph applies a callable object to every index 
    in the range [beg, end) with a step size chunk by chunk.

    @tparam I floating (arithmetic) index type
    @tparam C callable type

    @param beg index of the beginning (inclusive)
    @param end index of the end (exclusive)
    @param step step size 
    @param callable a callable object to be applied to
    @param chunk items per task

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <
      typename I, 
      typename C, 
      std::enable_if_t<std::is_floating_point<std::decay_t<I>>::value, void>* = nullptr
    >
    std::pair<Task, Task> parallel_for(
      I beg, I end, I step, C&& callable, size_t chunk = 1
    );
 
    /**
    @brief construct a task dependency graph of parallel reduction
    
    The task dependency graph reduces items in the range [beg, end) to a single result.
    
    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop    binary operator that will be applied in unspecified order to the result
                  of dereferencing the input iterator
    
    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename T, typename B>
    std::pair<Task, Task> reduce(I beg, I end, T& result, B&& bop);
    
    /**
    @brief constructs a task dependency graph of parallel reduction through @std_min
    
    The task dependency graph applies a parallel reduction
    to find the minimum item in the range [beg, end) through @std_min reduction.

    @tparam I input iterator type
    @tparam T data type 

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename T>
    std::pair<Task, Task> reduce_min(I beg, I end, T& result);
    
    /**
    @brief constructs a task dependency graph of parallel reduction through @std_max
    
    The task dependency graph applies a parallel reduction
    to find the maximum item in the range [beg, end) through @std_max reduction.

    @tparam I input iterator type
    @tparam T data type 

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename T>
    std::pair<Task, Task> reduce_max(I beg, I end, T& result);
    
    /** 
    @brief constructs a task dependency graph of parallel transformation and reduction
    
    The task dependency graph transforms each item in the range [beg, end) 
    into a new data type and then reduce the results.

    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator
    @tparam U unary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop    binary function object that will be applied in unspecified order 
                  to the results of @c uop; the return type must be @c T
    @param uop    unary function object that transforms each element 
                  in the input range; the return type must be acceptable as input to @c bop
    
    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename T, typename B, typename U>
    std::pair<Task, Task> transform_reduce(I beg, I end, T& result, B&& bop, U&& uop);
    
    /**
    @brief constructs a task dependency graph of parallel transformation and reduction
    
    The task dependency graph transforms each item in the range [beg, end) 
    into a new data type and then apply two-layer reductions to derive the result.

    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator type
    @tparam P binary operator type
    @tparam U unary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop1   binary function object that will be applied in the second-layer reduction
                  to the results of @c bop2
    @param bop2   binary function object that will be applied in the first-layer reduction
                  to the results of @c uop and the dereferencing of input iterators
    @param uop    unary function object that will be applied to transform an item to a new 
                  data type that is acceptable as input to @c bop2
    
    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename T, typename B, typename P, typename U>
    std::pair<Task, Task> transform_reduce(
      I beg, I end, T& result, B&& bop1, P&& bop2, U&& uop
    );
    
    /**
    @brief creates an empty task

    @return a Task handle
    */
    Task placeholder();
    
    /**
    @brief adds a dependency link from task A to task B
    
    @param A task A
    @param B task B
    */
    void precede(Task A, Task B);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks a vector of tasks
    */
    void linearize(std::vector<Task>& tasks);

    /**
    @brief adds adjacent dependency links to a linear list of tasks

    @param tasks an initializer list of tasks
    */
    void linearize(std::initializer_list<Task> tasks);

    /**
    @brief adds dependency links from one task A to many tasks

    @param A      task A
    @param others a task set which A precedes
    */
    void broadcast(Task A, std::vector<Task>& others);

    /**
    @brief adds dependency links from one task A to many tasks

    @param A      task A
    @param others a task set which A precedes
    */
    void broadcast(Task A, std::initializer_list<Task> others);

    /**
    @brief adds dependency links from many tasks to one task A

    @param others a task set to precede A
    @param A task A
    */
    void gather(std::vector<Task>& others, Task A);

    /**
    @brief adds dependency links from many tasks to one task A

    @param others a task set to precede A
    @param A task A
    */
    void gather(std::initializer_list<Task> others, Task A);
    
  private:

    Graph& _graph;

    template <typename L>
    void _linearize(L&);
};

// Constructor
inline FlowBuilder::FlowBuilder(Graph& graph) :
  _graph {graph} {
}

// ----------------------------------------------------------------------------

/** 
@class Subflow

@brief The building blocks of dynamic tasking.
*/ 
class Subflow : public FlowBuilder {

  public:
    
    /**
    @brief constructs a subflow builder object
    */
    template <typename... Args>
    Subflow(Args&&... args);
    
    /**
    @brief enables the subflow to join its parent task
    */
    void join();

    /**
    @brief enables the subflow to detach from its parent task
    */
    void detach();
    
    /**
    @brief queries if the subflow will be detached from its parent task
    */
    bool detached() const;

    /**
    @brief queries if the subflow will join its parent task
    */
    bool joined() const;

  private:

    bool _detached {false};
};

// Constructor
template <typename... Args>
Subflow::Subflow(Args&&... args) :
  FlowBuilder {std::forward<Args>(args)...} {
}

// Procedure: join
inline void Subflow::join() {
  _detached = false;
}

// Procedure: detach
inline void Subflow::detach() {
  _detached = true;
}

// Function: detached
inline bool Subflow::detached() const {
  return _detached;
}

// Function: joined
inline bool Subflow::joined() const {
  return !_detached;
}

// ----------------------------------------------------------------------------
// Member definition of FlowBuilder
// ----------------------------------------------------------------------------

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: emplace
// emplaces a static task
template <typename C>
std::enable_if_t<is_static_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::StaticWork>{}, std::forward<C>(c)
  );
  return Task(n);
}

// Function: emplace
// emplaces a dynamic task
template <typename C>
std::enable_if_t<is_dynamic_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(nstd::in_place_type_t<Node::DynamicWork>{}, 
  [c=std::forward<C>(c)] (Subflow& fb) mutable {
    // first time execution
    if(fb._graph.empty()) {
      c(fb);
    }
  });
  return Task(n);
}

// Function: emplace 
// emplaces a condition task
template <typename C>
std::enable_if_t<is_condition_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::ConditionWork>{}, std::forward<C>(c)
  );
  return Task(n);
}

#ifdef TF_ENABLE_CUDA
// Function: emplace
// emplaces a cudaflow task
template <typename C>
std::enable_if_t<is_cudaflow_task_v<C>, Task> FlowBuilder::emplace(C&& c) {
  auto n = _graph.emplace_back(
    nstd::in_place_type_t<Node::cudaFlowWork>{}, std::forward<C>(c)
  );
  return Task(n);
}
#endif

//template <typename C>
//Task FlowBuilder::emplace(C&& c) {
//
//  // static task
//  //if constexpr(is_static_task_v<C>) {
//  if constexpr (is_invocable_r_v<void, C> && !is_invocable_r_v<int, C>) {
//    auto n = _graph.emplace_back(
//      nstd::in_place_type_t<Node::StaticWork>{}, std::forward<C>(c)
//    );
//    return Task(n);
//  }
//  // condition task
//  else if constexpr(is_condition_task_v<C>) {
//    auto n = _graph.emplace_back(
//      nstd::in_place_type_t<Node::ConditionWork>{}, std::forward<C>(c)
//    );
//    return Task(n);
//  }
//  // dynamic task
//  else if constexpr(is_dynamic_task_v<C>) {
//    auto n = _graph.emplace_back(nstd::in_place_type_t<Node::DynamicWork>{}, 
//    [c=std::forward<C>(c)] (Subflow& fb) mutable {
//      // first time execution
//      if(fb._graph.empty()) {
//        c(fb);
//      }
//    });
//    return Task(n);
//  }
//  else {
//    static_assert(dependent_false_v<C>, "invalid task work type");
//  }
//}

// Function: composed_of    
inline Task FlowBuilder::composed_of(Taskflow& taskflow) {
  auto node = _graph.emplace_back(
    nstd::in_place_type_t<Node::ModuleWork>{}, &taskflow
  );
  return Task(node);
}

// Procedure: precede
inline void FlowBuilder::precede(Task from, Task to) {
  from._node->_precede(to._node);
}

// Procedure: broadcast
inline void FlowBuilder::broadcast(Task from, std::vector<Task>& tos) {
  for(auto to : tos) {
    from.precede(to);
  }
}

// Procedure: broadcast
inline void FlowBuilder::broadcast(Task from, std::initializer_list<Task> tos) {
  for(auto to : tos) {
    from.precede(to);
  }
}

// Function: gather
inline void FlowBuilder::gather(std::vector<Task>& froms, Task to) {
  for(auto from : froms) {
    to.succeed(from);
  }
}

// Function: gather
inline void FlowBuilder::gather(std::initializer_list<Task> froms, Task to) {
  for(auto from : froms) {
    to.succeed(from);
  }
}

// Function: placeholder
inline Task FlowBuilder::placeholder() {
  auto node = _graph.emplace_back();
  return Task(node);
}

// Function: parallel_for
template <typename I, typename C>
std::pair<Task, Task> FlowBuilder::parallel_for(
  I beg, I end, C&& c, size_t chunk
){
  
  //using category = typename std::iterator_traits<I>::iterator_category;
  
  auto S = placeholder();
  auto T = placeholder();
  
  // default partition equals to the worker count
  if(chunk == 0) {
    chunk = 1;
  }

  size_t remain = std::distance(beg, end);

  while(beg != end) {

    auto e = beg;
    
    auto x = std::min(remain, chunk);
    std::advance(e, x);
    remain -= x;
      
    // Create a task
    auto task = emplace([beg, e, c] () mutable {
      std::for_each(beg, e, c);
    });

    S.precede(task);
    task.precede(T);

    // adjust the pointer
    beg = e;
  }
  
  // special case
  if(S.num_successors() == 0) {
    S.precede(T);
  }
  
  return std::make_pair(S, T); 
}

// Function: parallel_for
template <
  typename I, 
  typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, I s, C&& c, size_t chunk) {
  
  if((s == 0) || (beg < end && s <= 0) || (beg > end && s >=0) ) {
    TF_THROW("invalid range [", beg, ", ", end, ") with step size ", s);
  }

  // source and target 
  auto source = placeholder();
  auto target = placeholder();
  
  if(chunk == 0) {
    chunk = 1;
  }

  // positive case
  if(beg < end) {
    while(beg != end) {
      auto o = static_cast<I>(chunk) * s;
      auto e = std::min(beg + o, end);
      auto task = emplace([=] () mutable {
        for(auto i=beg; i<e; i+=s) {
          c(i);
        }
      });
      source.precede(task);
      task.precede(target);
      beg = e;
    }
  }
  // negative case
  else if(beg > end) {
    while(beg != end) {
      auto o = static_cast<I>(chunk) * s;
      auto e = std::max(beg + o, end);
      auto task = emplace([=] () mutable {
        for(auto i=beg; i>e; i+=s) {
          c(i);
        }
      });
      source.precede(task);
      task.precede(target);
      beg = e;
    }
  }

  if(source.num_successors() == 0) {
    source.precede(target);
  }
    
  return std::make_pair(source, target);
}

// Function: parallel_for
template <typename I, typename C, 
  std::enable_if_t<std::is_floating_point<std::decay_t<I>>::value, void>*
>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, I s, C&& c, size_t chunk) {
  
  if((s == 0) || (beg < end && s <= 0) || (beg > end && s >=0) ) {
    TF_THROW("invalid range [", beg, ", ", end, ") with step size ", s);
  }

  // source and target 
  auto source = placeholder();
  auto target = placeholder();
  
  if(chunk == 0) {
    chunk = 1;
  }

  // positive case
  if(beg < end) {
    size_t N=0;
    I b = beg;
    for(I e=beg; e<end; e+=s) {
      if(++N == chunk) {
        auto task = emplace([=] () mutable {
          for(size_t i=0; i<N; ++i, b+=s) {
            c(b);
          }
        });
        source.precede(task);
        task.precede(target);
        N = 0;
        b = e;
      }
    }

    if(N) {
      auto task = emplace([=] () mutable {
        for(size_t i=0; i<N; ++i, b+=s) {
          c(b);
        }
      });
      source.precede(task);
      task.precede(target);
    }
  }
  else if(beg > end) {
    size_t N=0;
    I b = beg;
    for(I e=beg; e>end; e+=s) {
      if(++N == chunk) {
        auto task = emplace([=] () mutable {
          for(size_t i=0; i<N; ++i, b+=s) {
            c(b);
          }
        });
        source.precede(task);
        task.precede(target);
        N = 0;
        b = e;
      }
    }

    if(N) {
      auto task = emplace([=] () mutable {
        for(size_t i=0; i<N; ++i, b+=s) {
          c(b);
        }
      });
      source.precede(task);
      task.precede(target);
    }
  }

  if(source.num_successors() == 0) {
    source.precede(target);
  }
    
  return std::make_pair(source, target);
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename I, typename T>
std::pair<Task, Task> FlowBuilder::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename I, typename T>
std::pair<Task, Task> FlowBuilder::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename U>
std::pair<Task, Task> FlowBuilder::transform_reduce(
  I beg, I end, T& result, B&& bop, U&& uop
) {

  //using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  //std::vector<std::future<T>> futures;
  auto g_results = std::make_unique<T[]>(w);
  size_t id {0};

  size_t remain = d;

  while(beg != end) {

    auto e = beg;
    
    size_t x = std::min(remain, g);
    std::advance(e, x);
    remain -= x;

    // Create a task 
    auto task = emplace([beg, e, bop, uop, res=&(g_results[id])] () mutable {
      *res = uop(*beg);
      for(++beg; beg != e; ++beg) {
        *res = bop(std::move(*res), uop(*beg));          
      }
    });

    source.precede(task);
    task.precede(target);

    // adjust the pointer
    beg = e;
    id ++;
  }

  // target synchronizer 
  target.work([&result, bop, res=make_moc(std::move(g_results)), w=id] () {
    for(auto i=0u; i<w; i++) {
      result = bop(std::move(result), res.object[i]);
    }
  });

  return std::make_pair(source, target); 
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename P, typename U>
std::pair<Task, Task> FlowBuilder::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {

  //using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  auto g_results = std::make_unique<T[]>(w);

  size_t id {0};
  size_t remain = d;

  while(beg != end) {

    auto e = beg;
    
    size_t x = std::min(remain, g);
    std::advance(e, x);
    remain -= x;
      
    // Create a task 
    auto task = emplace([beg, e, uop, pop,  res= &g_results[id]] () mutable {
      *res = uop(*beg);
      for(++beg; beg != e; ++beg) {
        *res = pop(std::move(*res), *beg);
      }
    });
    source.precede(task);
    task.precede(target);

    // adjust the pointer
    beg = e;
    id ++;
  }

  // target synchronizer 
  target.work([&result, bop, g_results=make_moc(std::move(g_results)), w=id] () {
    for(auto i=0u; i<w; i++) {
      result = bop(std::move(result), std::move(g_results.object[i]));
    }
  });

  return std::make_pair(source, target); 
}

// Procedure: _linearize
template <typename L>
void FlowBuilder::_linearize(L& keys) {

  auto itr = keys.begin();
  auto end = keys.end();

  if(itr == end) {
    return;
  }

  auto nxt = itr;

  for(++nxt; nxt != end; ++nxt, ++itr) {
    itr->_node->_precede(nxt->_node);
  }
}

// Procedure: linearize
inline void FlowBuilder::linearize(std::vector<Task>& keys) {
  _linearize(keys); 
}

// Procedure: linearize
inline void FlowBuilder::linearize(std::initializer_list<Task> keys) {
  _linearize(keys);
}

// Proceduer: reduce
template <typename I, typename T, typename B>
std::pair<Task, Task> FlowBuilder::reduce(I beg, I end, T& result, B&& op) {
  
  //using category = typename std::iterator_traits<I>::iterator_category;
  
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  //T* g_results = static_cast<T*>(malloc(sizeof(T)*w));
  auto g_results = std::make_unique<T[]>(w);
  //std::vector<std::future<T>> futures;
  
  size_t id {0};
  size_t remain = d;

  while(beg != end) {

    auto e = beg;
    
    size_t x = std::min(remain, g);
    std::advance(e, x);
    remain -= x;
    
    // Create a task
    //auto [task, future] = emplace([beg, e, op] () mutable {
    auto task = emplace([beg, e, op, res = &g_results[id]] () mutable {
      *res = *beg;
      for(++beg; beg != e; ++beg) {
        *res = op(std::move(*res), *beg);          
      }
      //auto init = *beg;
      //for(++beg; beg != e; ++beg) {
      //  init = op(std::move(init), *beg);          
      //}
      //return init;
    });
    source.precede(task);
    task.precede(target);
    //futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
    id ++;
  }
  
  // target synchronizer
  //target.work([&result, futures=MoC{std::move(futures)}, op] () {
  //  for(auto& fu : futures.object) {
  //    result = op(std::move(result), fu.get());
  //  }
  //});
  target.work([g_results=make_moc(std::move(g_results)), &result, op, w=id] () {
    for(auto i=0u; i<w; i++) {
      result = op(std::move(result), g_results.object[i]);
    }
  });

  return std::make_pair(source, target); 
}

// ----------------------------------------------------------------------------
// Cyclic Dependency: Task
// ----------------------------------------------------------------------------

// Function: work
// assign a static work
template <typename C>
std::enable_if_t<is_static_task_v<C>, Task>& Task::work(C&& c) {
  _node->_handle.emplace<Node::StaticWork>(std::forward<C>(c));
  return *this;
}

// Function: work
// assigns a dynamic work
template <typename C>
std::enable_if_t<is_dynamic_task_v<C>, Task>& Task::work(C&& c) {
  _node->_handle.emplace<Node::DynamicWork>( 
  [c=std::forward<C>(c)] (Subflow& fb) mutable {
    // first time execution
    if(fb._graph.empty()) {
      c(fb);
    }
  });
  return *this;
}

// Function: work
// assigns a condition work
template <typename C>
std::enable_if_t<is_condition_task_v<C>, Task>& Task::work(C&& c) {
  _node->_handle.emplace<Node::ConditionWork>(std::forward<C>(c));
  return *this;
}


// Function: work
//template <typename C>
//Task& Task::work(C&& c) {
//
//  // static tasking
//  if constexpr(is_static_task_v<C>) {
//    _node->_handle.emplace<Node::StaticWork>(std::forward<C>(c));
//  }
//  // condition tasking
//  else if constexpr(is_condition_task_v<C>) {
//    _node->_handle.emplace<Node::ConditionWork>(std::forward<C>(c));
//  }
//  // dyanmic tasking
//  else if constexpr(is_dynamic_task_v<C>) {
//    _node->_handle.emplace<Node::DynamicWork>( 
//    [c=std::forward<C>(c)] (Subflow& fb) mutable {
//      // first time execution
//      if(fb._graph.empty()) {
//        c(fb);
//      }
//    });
//  }
//  else {
//    static_assert(dependent_false_v<C>, "invalid task work type");
//  }
//
//  return *this;
//}

// ----------------------------------------------------------------------------
// Legacy code
// ----------------------------------------------------------------------------


using SubflowBuilder = Subflow;

}  // end of namespace tf. ---------------------------------------------------


