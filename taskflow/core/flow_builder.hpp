#pragma once

#include "task.hpp"

namespace tf {

/** 
@class FlowBuilder

@brief Building blocks of a task dependency graph.

*/
class FlowBuilder {

  public:
    
    /**
    @brief construct a flow builder object

    @param graph a task dependency graph to manipulate
    */
    FlowBuilder(Graph& graph);
    
    /**
    @brief creates a task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function

    @return Task handle
    */
    template <typename C>
    Task emplace(C&& callable);
    
    /**
    @brief creates multiple tasks from a list of callable objects at one time
    
    @tparam C... callable types

    @param callables one or multiple callable objects acceptable to std::function

    @return a Task handle
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&... callables);
    
    /**
    @brief the same as tf::FlowBuilder::emplace (starting at 2.1.0)
    */
    template <typename C>
    Task silent_emplace(C&& callable);

    /**
    @brief the same as tf::FlowBuilder::emplace (starting at 2.1.0)
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&... callables);
    
    /**
    @brief constructs a task dependency graph of range-based parallel_for
    
    The task dependency graph applies a callable object 
    to the dereferencing of every iterator 
    in the range [beg, end) partition-by-partition.

    @tparam I input iterator type
    @tparam C callable type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to be applied to 
    @param partitions number of partitions

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename C>
    std::pair<Task, Task> parallel_for(I beg, I end, C&& callable, size_t partitions = 0);
    
    /**
    @brief constructs a task dependency graph of index-based parallel_for
    
    The task dependency graph applies a callable object to every index 
    in the range [beg, end) with a step size partition-by-partition.

    @tparam I arithmetic index type
    @tparam C callable type

    @param beg index to the beginning (inclusive)
    @param end index to the end (exclusive)
    @param step step size 
    @param callable a callable object to be applied to
    @param partitions number of partitions

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename C, std::enable_if_t<std::is_arithmetic_v<I>, void>* = nullptr >
    std::pair<Task, Task> parallel_for(I beg, I end, I step, C&& callable, size_t partitions = 0);
    
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
    std::pair<Task, Task> transform_reduce(I beg, I end, T& result, B&& bop1, P&& bop2, U&& uop);
    
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

// Procedure: precede
inline void FlowBuilder::precede(Task from, Task to) {
  from._node->precede(*(to._node));
}

// Procedure: broadcast
inline void FlowBuilder::broadcast(Task from, std::vector<Task>& keys) {
  from.precede(keys);
}

// Procedure: broadcast
inline void FlowBuilder::broadcast(Task from, std::initializer_list<Task> keys) {
  from.precede(keys);
}

// Function: gather
inline void FlowBuilder::gather(std::vector<Task>& keys, Task to) {
  to.gather(keys);
}

// Function: gather
inline void FlowBuilder::gather(std::initializer_list<Task> keys, Task to) {
  to.gather(keys);
}

// Function: placeholder
inline Task FlowBuilder::placeholder() {
  auto& node = _graph.emplace_back();
  return Task(node);
}

// Function: parallel_for
template <typename I, typename C>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, C&& c, size_t p){

  using category = typename std::iterator_traits<I>::iterator_category;
  
  auto S = placeholder();
  auto T = placeholder();
  auto D = std::distance(beg, end);
  
  // special case
  if(D == 0) {
    S.precede(T);
    return std::make_pair(S, T);
  }
  
  // default partition equals to the worker count
  if(p == 0) {
    p = std::max(unsigned{1}, std::thread::hardware_concurrency());
  }

  size_t b = (D + p - 1) / p;           // block size
  size_t r = (D % p) ? D % p : p;       // workers to take b
  size_t w = 0;                         // worker id

  while(beg != end) {

    auto e = beg;
    size_t g = (w++ >= r) ? b - 1 : b;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t x = std::distance(beg, end);
      std::advance(e, std::min(x, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto task = emplace([beg, e, c] () mutable {
      std::for_each(beg, e, c);
    });
    S.precede(task);
    task.precede(T);

    // adjust the pointer
    beg = e;
  }
  
  return std::make_pair(S, T); 
}

/*// Function: parallel_for    
template <typename I, typename C>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, C&& c, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(unsigned{1}, std::thread::hardware_concurrency());
    g = (d + w - 1) / w;
  }

  auto source = placeholder();
  auto target = placeholder();
  
  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task
    auto task = emplace([beg, e, c] () mutable {
      std::for_each(beg, e, c);
    });
    source.precede(task);
    task.precede(target);

    // adjust the pointer
    beg = e;
  }

  return std::make_pair(source, target); 
}*/

// Function: parallel_for
template <
  typename I, 
  typename C, 
  std::enable_if_t<std::is_arithmetic_v<I>, void>*
>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, I s, C&& c, size_t p) {

  using T = std::decay_t<I>;

  if((s == 0) || (beg < end && s <= 0) || (beg > end && s >=0) ) {
    TF_THROW(Error::TASKFLOW, 
      "invalid range [", beg, ", ", end, ") with step size ", s
    );
  }

  // compute the distance
  size_t D;

  if constexpr(std::is_integral_v<T>) {
    if(beg <= end) {  
      D = (end - beg + s - 1) / s;
    }
    else {
      D = (end - beg + s + 1) / s;
    }
  }
  else if constexpr(std::is_floating_point_v<T>) {
    D = static_cast<size_t>(std::ceil((end - beg) / s));
  }
  else {
    static_assert(dependent_false_v<T>, "can't deduce distance");
  }

  // source and target 
  auto source = placeholder();
  auto target = placeholder();

  // special case
  if(D == 0) {
    source.precede(target);
    return std::make_pair(source, target);
  }
  
  // default partition equals to the worker count
  if(p == 0) {
    p = std::max(unsigned{1}, std::thread::hardware_concurrency());
  }
  
  size_t b = (D + p - 1) / p;           // block size
  size_t r = (D % p) ? D % p : p;       // workers to take b
  size_t w = 0;                         // worker id

  // Integer indices
  if constexpr(std::is_integral_v<T>) {
    // positive case
    if(beg < end) {
      while(beg != end) {
        auto g = (w++ >= r) ? b - 1 : b;
        auto o = static_cast<T>(g) * s;
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
        auto g = (w++ >= r) ? b - 1 : b;
        auto o = static_cast<T>(g) * s;
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
  }
  // We enumerate the entire sequence to avoid floating error
  else if constexpr(std::is_floating_point_v<T>) {
    size_t N = 0;
    size_t g = b;
    auto B = beg;
    for(auto i=beg; (beg<end ? i<end : i>end); i+=s, ++N) {
      if(N == g) {
        auto task = emplace([=] () mutable {
          auto b = B;
          for(size_t n=0; n<N; ++n) {
            c(b);
            b += s; 
          }
        });
        N = 0;
        B = i;
        source.precede(task);
        task.precede(target);
        if(++w >= r) {
          g = b - 1;
        }
      }
    }

    // the last pices
    if(N != 0) {
      auto task = emplace([=] () mutable {
        auto b = B;
        for(size_t n=0; n<N; ++n) {
          c(b);
          b += s; 
        }
      });
      source.precede(task);
      task.precede(target);
    }
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
std::pair<Task, Task> FlowBuilder::transform_reduce(I beg, I end, T& result, B&& bop, U&& uop) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  //std::vector<std::future<T>> futures;
  auto g_results = std::make_unique<T[]>(w);
  size_t id {0};

  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
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
  target.work([&result, bop, res=MoC{std::move(g_results)}, w=id] () {
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

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  auto g_results = std::make_unique<T[]>(w);

  size_t id {0};
  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
    // Create a task 
    auto task = emplace([beg, e, uop, pop,  res= &g_results[id]] () mutable {
      *res = uop(*beg);
      for(++beg; beg != e; ++beg) {
        *res = pop(std::move(*res), *beg);
      }
    });
    //auto [task, future] = emplace([beg, e, uop, pop] () mutable {
    //  auto init = uop(*beg);
    //  for(++beg; beg != e; ++beg) {
    //    init = pop(std::move(init), *beg);
    //  }
    //  return init;
    //});
    source.precede(task);
    task.precede(target);
    //futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
    id ++;
  }

  // target synchronizer 
  target.work([&result, bop, g_results=MoC{std::move(g_results)}, w=id] () {
    for(auto i=0u; i<w; i++) {
      result = bop(std::move(result), std::move(g_results.object[i]));
    }
  });
  //target.work([&result, futures=MoC{std::move(futures)}, bop] () {
  //  for(auto& fu : futures.object) {
  //    result = bop(std::move(result), fu.get());
  //  }
  //});

  return std::make_pair(source, target); 
}

//// Function: _estimate_chunk_size
//template <typename I>
//size_t FlowBuilder::_estimate_chunk_size(I beg, I end, I step) {
//
//  using T = std::decay_t<I>;
//      
//  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
//  size_t N = 0;
//
//  if constexpr(std::is_integral_v<T>) {
//    if(beg <= end) {  
//      N = (end - beg + step - 1) / step;
//    }
//    else {
//      N = (end - beg + step + 1) / step;
//    }
//  }
//  else if constexpr(std::is_floating_point_v<T>) {
//    N = static_cast<size_t>(std::ceil((end - beg) / step));
//  }
//  else {
//    static_assert(dependent_false_v<T>, "can't deduce chunk size");
//  }
//
//  return (N + w - 1) / w;
//}


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
    itr->_node->precede(*(nxt->_node));
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
  
  using category = typename std::iterator_traits<I>::iterator_category;
  
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  //T* g_results = static_cast<T*>(malloc(sizeof(T)*w));
  auto g_results = std::make_unique<T[]>(w);
  //std::vector<std::future<T>> futures;
  
  size_t id {0};
  while(beg != end) {

    auto e = beg;
    
    // Case 1: random access iterator
    if constexpr(std::is_same_v<category, std::random_access_iterator_tag>) {
      size_t r = std::distance(beg, end);
      std::advance(e, std::min(r, g));
    }
    // Case 2: non-random access iterator
    else {
      for(size_t i=0; i<g && e != end; ++e, ++i);
    }
      
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
  target.work([g_results=MoC{std::move(g_results)}, &result, op, w=id] () {
    for(auto i=0u; i<w; i++) {
      result = op(std::move(result), g_results.object[i]);
    }
  });

  return std::make_pair(source, target); 
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

// -----

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: emplace
template <typename C>
Task FlowBuilder::emplace(C&& c) {
  // dynamic tasking
  if constexpr(std::is_invocable_v<C, Subflow&>) {
    auto& n = _graph.emplace_back(
    [c=std::forward<C>(c)] (Subflow& fb) mutable {
      // first time execution
      if(fb._graph.empty()) {
        c(fb);
      }
    });
    return Task(n);
  }
  // static tasking
  else if constexpr(std::is_invocable_v<C>) {
    auto& n = _graph.emplace_back(std::forward<C>(c));
    return Task(n);
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

// Function: silent_emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::silent_emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: silent_emplace
template <typename C>
Task FlowBuilder::silent_emplace(C&& c) {
  return emplace(std::forward<C>(c));
}

// ---------- deprecated ----------

//// Function: emplace
//template <typename C>
//auto FlowBuilder::emplace(C&& c) {
//  // subflow task
//  if constexpr(std::is_invocable_v<C, Subflow&>) {
//
//    using R = std::invoke_result_t<C, Subflow&>;
//    std::promise<R> p;
//    auto fu = p.get_future();
//  
//    if constexpr(std::is_same_v<void, R>) {
//      auto& node = _graph.emplace_back([p=MoC(std::move(p)), c=std::forward<C>(c)]
//      (Subflow& fb) mutable {
//        if(fb._graph.empty()) {
//          c(fb);
//          // if subgraph is detached or empty after invoked
//          if(fb.detached() || fb._graph.empty()) {
//            p.get().set_value();
//          }
//        }
//        else {
//          p.get().set_value();
//        }
//      });
//      return std::make_pair(Task(node), std::move(fu));
//    }
//    else {
//      auto& node = _graph.emplace_back(
//      [p=MoC(std::move(p)), c=std::forward<C>(c), r=std::optional<R>()]
//      (Subflow& fb) mutable {
//        if(fb._graph.empty()) {
//          r.emplace(c(fb));
//          if(fb.detached() || fb._graph.empty()) {
//            p.get().set_value(std::move(*r)); 
//          }
//        }
//        else {
//          assert(r);
//          p.get().set_value(std::move(*r));
//        }
//      });
//      return std::make_pair(Task(node), std::move(fu));
//    }
//  }
//  // regular task
//  else if constexpr(std::is_invocable_v<C>) {
//
//    using R = std::invoke_result_t<C>;
//    std::promise<R> p;
//    auto fu = p.get_future();
//
//    if constexpr(std::is_same_v<void, R>) {
//      auto& node = _graph.emplace_back(
//        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
//          c(); 
//          p.get().set_value();
//        }
//      );
//      return std::make_pair(Task(node), std::move(fu));
//    }
//    else {
//      auto& node = _graph.emplace_back(
//        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
//          p.get().set_value(c());
//        }
//      );
//      return std::make_pair(Task(node), std::move(fu));
//    }
//  }
//  else {
//    static_assert(dependent_false_v<C>, "invalid task work type");
//  }
//}

using SubflowBuilder = Subflow;

}  // end of namespace tf. ---------------------------------------------------
