#pragma once

#include "task.hpp"

namespace tf {

// Class: FlowBuilder
class FlowBuilder {

  public:
    
    FlowBuilder(Graph&);
    
    /**
    @brief create a task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object

    @return a @std_pair of Task handle and @std_future
    */
    template <typename C>
    auto emplace(C&& callable);
    
    /**
    @brief create multiple tasks from a list of callable objects at one time
    
    @tparam C callable type

    @param callables... a list of callable objects

    @return a @std_tuple of pairs of Task Handle and @std_future
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&... callables);
    
    /**
    @brief create a task from a given callable object without access to the result
    
    @tparam C callable type
    
    @param callable a callable object 

    @return a Task handle
    */
    template <typename C>
    auto silent_emplace(C&& callable);

    /**
    @brief create multiple tasks from a list of callable objects without access to the results
    
    @tparam C callable type
    
    @param callables... a list of callable objects

    @return a tuple of Task handles
    */
    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&... callables);
    
    /**
    @brief apply a callable object to the dereferencing of every iterator in the range 
           [beg, end) chunk-by-chunk

    @tparam I input iterator type
    @tparam C callable type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param callable a callable object to be applied to 
    @param chunk number of works per thread

    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename C>
    std::pair<Task, Task> parallel_for(I beg, I end, C&& callable, size_t chunk = 0);
    
    /**
    @brief apply a callable object to every index in the range [beg, end) with a step size
           chunk-by-chunk

    @tparam I arithmetic index type
    @tparam C callable type

    @param beg index to the beginning (inclusive)
    @param end index to the end (exclusive)
    @param step step size 
    @param callable a callable object to be applied to
    @param chunk number of works per thread

    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename C, std::enable_if_t<std::is_arithmetic_v<I>, void>* = nullptr >
    std::pair<Task, Task> parallel_for(I beg, I end, I step, C&& callable, size_t chunk = 0);
    
    /**
    @brief reduce items in the range [beg, end) to a single result
    
    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop    binary operator that will be applied in unspecified order to the result
                  of dereferencing the input iterator
    
    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename T, typename B>
    std::pair<Task, Task> reduce(I beg, I end, T& result, B&& bop);
    
    /**
    @brief find the minimum item in the range [beg, end) through @std_min reduction

    @tparam I input iterator type
    @tparam T data type 

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result

    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename T>
    std::pair<Task, Task> reduce_min(I beg, I end, T& result);
    
    /**
    @brief find the maximum item in the range [beg, end) through @std_max reduction

    @tparam I input iterator type
    @tparam T data type 

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result

    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename T>
    std::pair<Task, Task> reduce_max(I beg, I end, T& result);
    
    /** 
    @brief transform each item in the range [beg, end) into a new data type and then
           reduce the results

    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator
    @tparam U unary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop    binary function object that will be applied in unspecified order 
                  to the results of @em uop; the return type must be @em T
    @param uop    unary function object that transforms each element 
                  in the input range; the return type must be acceptable as input to @em bop
    
    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename T, typename B, typename U>
    std::pair<Task, Task> transform_reduce(I beg, I end, T& result, B&& bop, U&& uop);
    
    /**
    @brief transform each item in the range [beg, end) into a new data type and then
           apply two-layer reductions to derive the result

    @tparam I input iterator type
    @tparam T data type
    @tparam B binary operator type
    @tparam P binary operator type
    @tparam U unary operator type

    @param beg    iterator to the beginning (inclusive)
    @param end    iterator to the end (exclusive)
    @param result reference variable to store the final result
    @param bop1   binary function object that will be applied in the second-layer reduction
                  to the results of @em bop2
    @param bop2   binary function object that will be applied in the first-layer reduction
                  to the results of @em uop and the dereferencing of input iterators
    @param uop    unary function object that will be applied to transform an item to a new 
                  data type that is acceptable as input to @em bop2
    
    @return a pair of Task handles to the beginning and end of the graph
    */
    template <typename I, typename T, typename B, typename P, typename U>
    std::pair<Task, Task> transform_reduce(I beg, I end, T& result, B&& bop1, P&& bop2, U&& uop);
    
    /**
    @brief create an empty task

    @return a Task handle
    */
    auto placeholder();
    
    void precede(Task, Task);
    void linearize(std::vector<Task>&);
    void linearize(std::initializer_list<Task>);
    void broadcast(Task, std::vector<Task>&);
    void broadcast(Task, std::initializer_list<Task>);
    void gather(std::vector<Task>&, Task);
    void gather(std::initializer_list<Task>, Task);  
    
  private:

    Graph& _graph;

    template <typename L>
    void _linearize(L&);

    template <typename I>
    size_t _estimate_chunk_size(I, I, I);
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
  from.broadcast(keys);
}

// Procedure: broadcast
inline void FlowBuilder::broadcast(Task from, std::initializer_list<Task> keys) {
  from.broadcast(keys);
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
inline auto FlowBuilder::placeholder() {
  auto& node = _graph.emplace_front();
  return Task(node);
}

// Function: silent_emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}

// Function: parallel_for    
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
    auto task = silent_emplace([beg, e, c] () mutable {
      std::for_each(beg, e, c);
    });
    source.precede(task);
    task.precede(target);

    // adjust the pointer
    beg = e;
  }

  return std::make_pair(source, target); 
}

// Function: parallel_for
template <
  typename I, 
  typename C, 
  std::enable_if_t<std::is_arithmetic_v<I>, void>*
>
std::pair<Task, Task> FlowBuilder::parallel_for(I beg, I end, I s, C&& c, size_t g) {

  using T = std::decay_t<I>;

  if((s == 0) || (beg < end && s <= 0) || (beg > end && s >=0) ) {
    TF_THROW(Error::FLOW_BUILDER, 
      "invalid range [", beg, ", ", end, ") with step size ", s
    );
  }
    
  auto source = placeholder();
  auto target = placeholder();

  if(g == 0) {
    g = _estimate_chunk_size(beg, end, s);
  }

  // Integer indices
  if constexpr(std::is_integral_v<T>) {

    auto offset = static_cast<T>(g) * s;

    // positive case
    if(beg < end) {
      while(beg != end) {
        auto e = std::min(beg + offset, end);
        auto task = silent_emplace([=] () mutable {
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
        auto e = std::max(beg + offset, end);
        auto task = silent_emplace([=] () mutable {
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
    auto B = beg;
    for(auto i=beg; (beg<end ? i<end : i>end); i+=s, ++N) {
      if(N == g) {
        auto task = silent_emplace([=] () mutable {
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
      }
    }

    // the last pices
    if(N != 0) {
      auto task = silent_emplace([=] () mutable {
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

  std::vector<std::future<T>> futures;

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
    auto [task, future] = emplace([beg, e, bop, uop] () mutable {
      auto init = uop(*beg);
      for(++beg; beg != e; ++beg) {
        init = bop(std::move(init), uop(*beg));          
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }

  // target synchronizer
  target.work([&result, futures=MoC{std::move(futures)}, bop] () {
    for(auto& fu : futures.object) {
      result = bop(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename P, typename U>
std::pair<Task, Task> FlowBuilder::transform_reduce(I beg, I end, T& result, B&& bop, P&& pop, U&& uop) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t g = std::max((d + w - 1) / w, size_t{2});

  auto source = placeholder();
  auto target = placeholder();

  std::vector<std::future<T>> futures;

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
    auto [task, future] = emplace([beg, e, uop, pop] () mutable {
      auto init = uop(*beg);
      for(++beg; beg != e; ++beg) {
        init = pop(std::move(init), *beg);
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }

  // target synchronizer
  target.work([&result, futures=MoC{std::move(futures)}, bop] () {
    for(auto& fu : futures.object) {
      result = bop(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}

// Function: _estimate_chunk_size
template <typename I>
size_t FlowBuilder::_estimate_chunk_size(I beg, I end, I step) {

  using T = std::decay_t<I>;
      
  size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
  size_t N = 0;

  if constexpr(std::is_integral_v<T>) {
    if(beg <= end) {  
      N = (end - beg + step - 1) / step;
    }
    else {
      N = (end - beg + step + 1) / step;
    }
  }
  else if constexpr(std::is_floating_point_v<T>) {
    N = std::ceil((end - beg) / step);
  }
  else {
    static_assert(dependent_false_v<T>, "can't deduce chunk size");
  }

  return (N + w - 1) / w;
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

  std::vector<std::future<T>> futures;
  
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
    auto [task, future] = emplace([beg, e, op] () mutable {
      auto init = *beg;
      for(++beg; beg != e; ++beg) {
        init = op(std::move(init), *beg);          
      }
      return init;
    });
    source.precede(task);
    task.precede(target);
    futures.push_back(std::move(future));

    // adjust the pointer
    beg = e;
  }
  
  // target synchronizer
  target.work([&result, futures=MoC{std::move(futures)}, op] () {
    for(auto& fu : futures.object) {
      result = op(std::move(result), fu.get());
    }
  });

  return std::make_pair(source, target); 
}

// ----------------------------------------------------------------------------

// Class: SubflowBuilder
class SubflowBuilder : public FlowBuilder {

  public:
    
    template <typename... Args>
    SubflowBuilder(Args&&...);

    void join();
    void detach();

    bool detached() const;
    bool joined() const;

  private:

    bool _detached {false};
};

// Constructor
template <typename... Args>
SubflowBuilder::SubflowBuilder(Args&&... args) :
  FlowBuilder {std::forward<Args>(args)...} {
}

// Procedure: join
inline void SubflowBuilder::join() {
  _detached = false;
}

// Procedure: detach
inline void SubflowBuilder::detach() {
  _detached = true;
}

// Function: detached
inline bool SubflowBuilder::detached() const {
  return _detached;
}

// Function: joined
inline bool SubflowBuilder::joined() const {
  return !_detached;
}

// Function: emplace
template <typename C>
auto FlowBuilder::emplace(C&& c) {
    
  // subflow task
  if constexpr(std::is_invocable_v<C, SubflowBuilder&>) {

    using R = std::invoke_result_t<C, SubflowBuilder&>;
    std::promise<R> p;
    auto fu = p.get_future();
  
    if constexpr(std::is_same_v<void, R>) {
      auto& node = _graph.emplace_front([p=MoC(std::move(p)), c=std::forward<C>(c)]
      (SubflowBuilder& fb) mutable {
        if(fb._graph.empty()) {
          c(fb);
          if(fb.detached()) {
            p.get().set_value();
          }
        }
        else {
          p.get().set_value();
        }
      });
      return std::make_pair(Task(node), std::move(fu));
    }
    else {
      auto& node = _graph.emplace_front(
      [p=MoC(std::move(p)), c=std::forward<C>(c), r=std::optional<R>()]
      (SubflowBuilder& fb) mutable {
        if(fb._graph.empty()) {
          r.emplace(c(fb));
          if(fb.detached()) {
            p.get().set_value(std::move(*r)); 
          }
        }
        else {
          assert(r);
          p.get().set_value(std::move(*r));
        }
      });
      return std::make_pair(Task(node), std::move(fu));
    }
  }
  // regular task
  else if constexpr(std::is_invocable_v<C>) {

    using R = std::invoke_result_t<C>;
    std::promise<R> p;
    auto fu = p.get_future();

    if constexpr(std::is_same_v<void, R>) {
      auto& node = _graph.emplace_front(
        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
          c(); 
          p.get().set_value();
        }
      );
      return std::make_pair(Task(node), std::move(fu));
    }
    else {
      auto& node = _graph.emplace_front(
        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
          p.get().set_value(c());
        }
      );
      return std::make_pair(Task(node), std::move(fu));
    }
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

// Function: emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: silent_emplace
template <typename C>
auto FlowBuilder::silent_emplace(C&& c) {
  // dynamic tasking
  if constexpr(std::is_invocable_v<C, SubflowBuilder&>) {
    auto& n = _graph.emplace_front(
    [c=std::forward<C>(c)] (SubflowBuilder& fb) {
      // first time execution
      if(fb._graph.empty()) {
        c(fb);
      }
    });
    return Task(n);
  }
  // static tasking
  else if constexpr(std::is_invocable_v<C>) {
    auto& n = _graph.emplace_front(std::forward<C>(c));
    return Task(n);
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

};  // end of namespace tf. ---------------------------------------------------
