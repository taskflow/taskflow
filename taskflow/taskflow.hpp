// MIT License
// 
// Copyright (c) 2018 Tsung-Wei Huang, Chun-Xun Lin, Guannan Guo, and Martin Wong
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <iostream>
#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <optional>

#include "threadpool/threadpool.hpp"

// ============================================================================

// Clang mis-interprets variant's get as a non-friend of variant and cannot
// get compiled correctly. We use the patch: 
// https://gcc.gnu.org/viewcvs/gcc?view=revision&revision=258854
// to get rid of this.
#if defined(__clang__)
  #include "patch/clang_variant.hpp"
#else
  #include <variant>
#endif

// ============================================================================


namespace tf {

// Procedure: throw_re
template <typename... ArgsT>
inline void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss(std::ios_base::out);
  oss << '[' << fname << ':' << line << "] ";
  (oss << ... << std::forward<ArgsT>(args));
  throw std::runtime_error(oss.str());
}

#define TF_THROW(...) throw_re(__FILE__, __LINE__, __VA_ARGS__);

//-----------------------------------------------------------------------------
// Traits
//-----------------------------------------------------------------------------

// Macro to check whether a class has a member function
#define define_has_member(member_name)                                     \
template <typename T>                                                      \
class has_member_##member_name                                             \
{                                                                          \
  typedef char yes_type;                                                   \
  typedef long no_type;                                                    \
  template <typename U> static yes_type test(decltype(&U::member_name));   \
  template <typename U> static no_type  test(...);                         \
  public:                                                                  \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);  \
}

#define has_member(class_, member_name)  has_member_##member_name<class_>::value

// Struct: dependent_false
template <typename... T>
struct dependent_false { 
  static constexpr bool value = false; 
};

template <typename... T>
constexpr auto dependent_false_v = dependent_false<T...>::value;

// Struct: is_iterator
template <typename T, typename = void>
struct is_iterator {
  static constexpr bool value = false;
};

template <typename T>
struct is_iterator<
  T, 
  std::enable_if_t<!std::is_same_v<typename std::iterator_traits<T>::value_type, void>>
> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

// Struct: is_iterable
template <typename T, typename = void>
struct is_iterable : std::false_type {
};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()),
                                  decltype(std::declval<T>().end())>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

//-----------------------------------------------------------------------------
// Taskflow definition
//-----------------------------------------------------------------------------

// Forward declaration
class Node;
class Topology;
class Task;
class FlowBuilder;
class SubflowBuilder;
class Taskflow;
    
using Graph = std::forward_list<Node>;

// ----------------------------------------------------------------------------

// Class: Node
class Node {

  friend class Task;
  friend class Topology;
  friend class Taskflow;

  using StaticWork   = std::function<void()>;
  using DynamicWork  = std::function<void(SubflowBuilder&)>;

  public:

    Node();

    template <typename C>
    explicit Node(C&&);

    const std::string& name() const;
    
    void precede(Node&);

    size_t num_successors() const;
    size_t num_dependents() const;

    std::string dump() const;

  private:
    
    std::string _name;
    std::variant<StaticWork, DynamicWork> _work;
    std::vector<Node*> _successors;
    std::atomic<int> _dependents;

    Graph _subgraph;

    Topology* _topology;

    void _dump(std::ostream&) const;
};

// Constructor
inline Node::Node() {
  _dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
}

// Constructor
template <typename C>
inline Node::Node(C&& c) : _work {std::forward<C>(c)} {
  _dependents.store(0, std::memory_order_relaxed);
  _topology = nullptr;
}

// Procedure: precede
inline void Node::precede(Node& v) {
  _successors.push_back(&v);
  v._dependents.fetch_add(1, std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: dependents
inline size_t Node::num_dependents() const {
  return _dependents.load(std::memory_order_relaxed);
}

// Function: name
inline const std::string& Node::name() const {
  return _name;
}

// Function: dump
inline std::string Node::dump() const {
  std::ostringstream os;  
  _dump(os);
  return os.str();
}

// Function: _dump
inline void Node::_dump(std::ostream& os) const {
  
  if(_name.empty()) os << '\"' << this << '\"';
  else os << std::quoted(_name);
  os << ";\n";

  for(const auto s : _successors) {

    if(_name.empty()) os << '\"' << this << '\"';
    else os << std::quoted(_name);

    os << " -> ";
    
    if(s->name().empty()) os << '\"' << s << '\"';
    else os << std::quoted(s->name());

    os << ";\n";
  }
  
  if(!_subgraph.empty()) {

    os << "subgraph cluster_";
    if(_name.empty()) os << this;
    else os << _name;
    os << " {\n";

    os << "label = \"Subflow_";
    if(_name.empty()) os << this;
    else os << _name;

    os << "\";\n" << "color=blue\n";

    for(const auto& n : _subgraph) {
      n._dump(os);
    }
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  friend class Taskflow;

  public:

    Topology(Graph&&);

    std::string dump() const;

  private:

    Graph _graph;

    std::shared_future<void> _future;

    Node _source;
    Node _target;

    void _dump(std::ostream&) const;
};

// Constructor
inline Topology::Topology(Graph&& t) : 
  _graph(std::move(t)) {

  _source._topology = this;
  _target._topology = this;
  
  std::promise<void> promise;

  _future = promise.get_future().share();

  _target._work = [p=MoC{std::move(promise)}] () mutable { 
    p.get().set_value(); 
  };

  // ensure the topology is connected
  _source.precede(_target);

  // Build the super source and super target.
  for(auto& node : _graph) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _source.precede(node);
    }

    if(node.num_successors() == 0) {
      node.precede(_target);
    }
  }
}

// Procedure: _dump
inline void Topology::_dump(std::ostream& os) const {

  assert(_source._subgraph.empty());
  assert(_target._subgraph.empty());
  
  os << "digraph Topology {\n"
     << _source.dump() 
     << _target.dump();

  for(const auto& node : _graph) {
    os << node.dump();
  }

  os << "}\n";
}
  
// Function: dump
inline std::string Topology::dump() const { 
  std::ostringstream os;
  _dump(os);
  return os.str();
}

// ----------------------------------------------------------------------------

// Class: Task
class Task {

  friend class FlowBuilder;
  friend class Taskflow;

  public:
    
    Task() = default;
    Task(Node&);
    Task(const Task&);
    Task(Task&&);

    Task& operator = (const Task&);

    const std::string& name() const;

    size_t num_successors() const;
    size_t num_dependents() const;

    Task& name(const std::string&);
    Task& precede(Task);
    Task& broadcast(std::vector<Task>&);
    Task& broadcast(std::initializer_list<Task>);
    Task& gather(std::vector<Task>&);
    Task& gather(std::initializer_list<Task>);

    template <typename C>
    Task& work(C&&);
  
    template <typename... Bs>
    Task& broadcast(Bs&&...);

    template <typename... Bs>
    Task& gather(Bs&&...);

  private:

    Node* _node {nullptr};

    template<typename S>
    void _broadcast(S&);

    template<typename S>
    void _gather(S&);
};

// Constructor
inline Task::Task(Node& t) : _node {&t} {
}

// Constructor
inline Task::Task(const Task& rhs) : _node {rhs._node} {
}

// Function: precede
inline Task& Task::precede(Task tgt) {
  _node->precede(*(tgt._node));
  return *this;
}

// Function: broadcast
template <typename... Bs>
inline Task& Task::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename S>
inline void Task::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
inline Task& Task::broadcast(std::vector<Task>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
inline Task& Task::broadcast(
  std::initializer_list<Task> tgts
) {
  _broadcast(tgts);
  return *this;
}

// Function: gather
template <typename... Bs>
inline Task& Task::gather(Bs&&... tgts) {
  (tgts.precede(*this), ...);
  return *this;
}

// Procedure: _gather
template <typename S>
inline void Task::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
inline Task& Task::gather(std::vector<Task>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
inline Task& Task::gather(std::initializer_list<Task> tgts) {
  _gather(tgts);
  return *this;
}

// Operator =
inline Task& Task::operator = (const Task& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
inline Task::Task(Task&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename C>
inline Task& Task::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
inline Task& Task::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& Task::name() const {
  return _node->_name;
}

// Function: num_dependents
inline size_t Task::num_dependents() const {
  return _node->_dependents.load(std::memory_order_relaxed);
}

// Function: num_successors
inline size_t Task::num_successors() const {
  return _node->_successors.size();
}

// ----------------------------------------------------------------------------

// Class: FlowBuilder
class FlowBuilder {

  public:

    FlowBuilder(Graph&, size_t);

    template <typename C>
    auto emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto emplace(C&&...);

    template <typename C>
    auto silent_emplace(C&&);

    template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>* = nullptr>
    auto silent_emplace(C&&...);

    template <typename I, typename C>
    auto parallel_for(I, I, C&&, size_t = 0);

    template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>* = nullptr>
    auto parallel_for(T&, C&&, size_t = 0);

    template <typename I, typename T, typename B>
    auto reduce(I, I, T&, B&&);

    template <typename I, typename T>
    auto reduce_min(I, I, T&);
    
    template <typename I, typename T>
    auto reduce_max(I, I, T&);

    template <typename I, typename T, typename B, typename U>
    auto transform_reduce(I, I, T&, B&&, U&&);

    template <typename I, typename T, typename B, typename P, typename U>
    auto transform_reduce(I, I, T&, B&&, P&&, U&&);
    
    auto placeholder();
    
    void precede(Task, Task);
    void linearize(std::vector<Task>&);
    void linearize(std::initializer_list<Task>);
    void broadcast(Task, std::vector<Task>&);
    void broadcast(Task, std::initializer_list<Task>);
    void gather(std::vector<Task>&, Task);
    void gather(std::initializer_list<Task>, Task);  

    size_t size() const;

    bool empty() const;

  protected:

    Graph& _graph;

    size_t _partition_factor;

    template <typename L>
    void _linearize(L&);
};

inline FlowBuilder::FlowBuilder(Graph& graph, size_t f) : 
  _graph {graph}, 
  _partition_factor {f} {
}    

// Procedure: size
inline size_t FlowBuilder::size() const {
  return std::distance(_graph.begin(), _graph.end());
}

// Function: empty
inline bool FlowBuilder::empty() const {
  return _graph.empty();
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
inline void FlowBuilder::broadcast(
  Task from, std::initializer_list<Task> keys
) {
  from.broadcast(keys);
}

// Function: gather
inline void FlowBuilder::gather(
  std::vector<Task>& keys, Task to
) {
  to.gather(keys);
}

// Function: gather
inline void FlowBuilder::gather(
  std::initializer_list<Task> keys, Task to
) {
  to.gather(keys);
}

// Function: placeholder
inline auto FlowBuilder::placeholder() {
  auto& node = _graph.emplace_front();
  return Task(node);
}

// Function: silent_emplace
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
inline auto FlowBuilder::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}


// Function: parallel_for    
template <typename I, typename C>
inline auto FlowBuilder::parallel_for(I beg, I end, C&& c, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, _partition_factor);
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
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
inline auto FlowBuilder::parallel_for(T& t, C&& c, size_t group) {
  return parallel_for(t.begin(), t.end(), std::forward<C>(c), group);
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename I, typename T>
inline auto FlowBuilder::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename I, typename T>
inline auto FlowBuilder::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename I, typename T, typename B, typename U>
inline auto FlowBuilder::transform_reduce(
  I beg, I end, T& result, B&& bop, U&& uop
) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _partition_factor);
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
inline auto FlowBuilder::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _partition_factor);
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


// Procedure: _linearize
template <typename L>
inline void FlowBuilder::_linearize(L& keys) {
  (void) std::adjacent_find(
    keys.begin(), keys.end(), 
    [] (auto& from, auto& to) {
      from._node->precede(*(to._node));
      return false;
    }
  );
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
inline auto FlowBuilder::reduce(I beg, I end, T& result, B&& op) {
  
  using category = typename std::iterator_traits<I>::iterator_category;
  
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _partition_factor);
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
inline SubflowBuilder::SubflowBuilder(Args&&... args) :
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
inline auto FlowBuilder::emplace(C&& c) {
    
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
inline auto FlowBuilder::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: silent_emplace
template <typename C>
inline auto FlowBuilder::silent_emplace(C&& c) {
  // subflow task
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
  // regular task
  else if constexpr(std::is_invocable_v<C>) {
    auto& n = _graph.emplace_front(std::forward<C>(c));
    return Task(n);
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

// ============================================================================
// Taskflow Definition
// ============================================================================

// Class: Taskflow
class Taskflow : public FlowBuilder {
  
  // Closure
  struct Closure {
  
    Closure() = default;
    Closure(const Closure&) = delete;
    Closure(Closure&&);
    Closure(Taskflow&, Node&);

    Closure& operator = (Closure&&);
    Closure& operator = (const Closure&) = delete;
    
    void operator ()() const;

    Taskflow* taskflow {nullptr};
    Node*     node     {nullptr};
  };

  public:

  using StaticWork  = typename Node::StaticWork;
  using DynamicWork = typename Node::DynamicWork;
 
    explicit Taskflow();
    explicit Taskflow(unsigned);

    ~Taskflow();
    
    std::shared_future<void> dispatch();

    void silent_dispatch();
    void wait_for_all();
    void wait_for_topologies();

    size_t num_nodes() const;
    size_t num_workers() const;
    size_t num_topologies() const;

    std::string dump() const;
    std::string dump_topologies() const;

  private:

    //SimpleThreadpool<Closure> _executor;
    //ProactiveThreadpool<Closure> _executor;
    SpeculativeThreadpool<Closure> _executor;
    //PrivatizedThreadpool<Closure> _executor;

    Graph _graph;

    std::forward_list<Topology> _topologies;

    void _schedule(Node&);
};

// Constructor    
inline Taskflow::Closure::Closure(Closure&& rhs) : 
  taskflow {rhs.taskflow}, node {rhs.node} { 
  rhs.taskflow = nullptr;
  rhs.node     = nullptr;
}

// Constructor
inline Taskflow::Closure::Closure(Taskflow& t, Node& n) : 
  taskflow{&t}, node {&n} {
}

// Move assignment
inline Taskflow::Closure& Taskflow::Closure::operator = (Closure&& rhs) {
  taskflow = rhs.taskflow;
  node     = rhs.node;
  rhs.taskflow = nullptr;
  rhs.node     = nullptr;
  return *this;
}

// Operator ()
inline void Taskflow::Closure::operator () () const {
  
  assert(taskflow && node);

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // regular node type
  // The default node work type. We only need to execute the callback if any.
  if(auto index=node->_work.index(); index == 0) {
    if(auto &f = std::get<StaticWork>(node->_work); f != nullptr){
      std::invoke(f);
    }
  }
  // subflow node type 
  // The first time we enter into the subflow context, 
  // "subnodes" must be empty.
  // After executing the user's callback on subflow, 
  // there will be at least one node node used as "super source". 
  // The second time we enter this context there is no need
  // to re-execute the work.
  else {
    assert(std::holds_alternative<DynamicWork>(node->_work));
    
    SubflowBuilder fb(node->_subgraph, taskflow->num_workers());

    bool empty_graph = node->_subgraph.empty();

    std::invoke(std::get<DynamicWork>(node->_work), fb);
    
    // Need to create a subflow
    if(empty_graph) {

      auto& S = node->_subgraph.emplace_front([](){});

      S._topology = node->_topology;

      for(auto i = std::next(node->_subgraph.begin()); i != node->_subgraph.end(); ++i) {

        i->_topology = node->_topology;

        if(i->num_successors() == 0) {
          i->precede(fb.detached() ? node->_topology->_target : *node);
        }

        if(i->num_dependents() == 0) {
          S.precede(*i);
        }
      }
      
      // this is for the case where subflow graph might be empty
      if(!fb.detached()) {
        S.precede(*node);
      }

      taskflow->_schedule(S);

      if(!fb.detached()) {
        return;
      }
    }
  }
  
  // At this point, the node/node storage might be destructed.
  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_dependents) == 0) {
      taskflow->_schedule(*(node->_successors[i]));
    }
  }
}

// Constructor
inline Taskflow::Taskflow() : 
  FlowBuilder {_graph, std::thread::hardware_concurrency()},
  _executor   {std::thread::hardware_concurrency()} {
}

// Constructor
inline Taskflow::Taskflow(unsigned N) : 
  FlowBuilder {_graph, std::thread::hardware_concurrency()},
  _executor   {N} {
}

// Destructor
inline Taskflow::~Taskflow() {
  wait_for_topologies();
}

// Function: num_nodes
inline size_t Taskflow::num_nodes() const {
  //return _nodes.size();
  return std::distance(_graph.begin(), _graph.end());
}

// Function: num_workers
inline size_t Taskflow::num_workers() const {
  return _executor.num_workers();
}

// Function: num_topologies
inline size_t Taskflow::num_topologies() const {
  return std::distance(_topologies.begin(), _topologies.end());
}

// Procedure: silent_dispatch 
inline void Taskflow::silent_dispatch() {

  if(_graph.empty()) return;

  auto& topology = _topologies.emplace_front(std::move(_graph));

  // Start the taskflow
  _schedule(topology._source);
}

// Procedure: dispatch 
inline std::shared_future<void> Taskflow::dispatch() {

  if(_graph.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_graph));

  // Start the taskflow
  _schedule(topology._source);
  
  return topology._future;
}

// Procedure: wait_for_all
inline void Taskflow::wait_for_all() {
  if(!_graph.empty()) {
    silent_dispatch();
  }
  wait_for_topologies();
}

// Procedure: wait_for_topologies
inline void Taskflow::wait_for_topologies() {
  for(auto& t: _topologies){
    t._future.get();
  }
  _topologies.clear();
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
inline void Taskflow::_schedule(Node& node) {
  _executor.emplace(*this, node);
}

// Function: dump_topology
inline std::string Taskflow::dump_topologies() const {
  
  std::ostringstream os;

  for(const auto& tpg : _topologies) {
    tpg._dump(os);
  }
  
  return os.str();
}

// Function: dump
// Dumps the taskflow in graphviz. The result can be viewed at http://www.webgraphviz.com/.
inline std::string Taskflow::dump() const {

  std::ostringstream os;

  os << "digraph Taskflow {\n";
  
  for(const auto& node : _graph) {
    node._dump(os);
  }

  os << "}\n";
  
  return os.str();
}

};  // end of namespace tf. ---------------------------------------------------





