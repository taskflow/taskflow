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
// version
#define TASKFLOW_VERSION_MAJOR 2
#define TASKFLOW_VERSION_MINOR 2
#define TASKFLOW_VERSION_PATCH 0
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

/*// Class: ObjectPool
template <typename T>
class ObjectPool {
  
  struct Deleter {
    Deleter(std::vector<T*>&);
    void operator()(T*);
    std::vector<T*>& recycle;
  };
  
  public:
  
  using HandleType = std::unique_ptr<T, Deleter>;
    
    template <typename... ArgsT>
    auto get(ArgsT&&...);

  private:
  
    std::forward_list<T> _pool;
    std::vector<T*> _recycle; 
};

// Constructor
template <typename T>
ObjectPool<T>::Deleter::Deleter(std::vector<T*>& r) : recycle(r) {
}

// Operator
template <typename T>
void ObjectPool<T>::Deleter::operator()(T* item) {
  if(item != nullptr) {
    item->~T();
    recycle.push_back(item);
  }
}

// Constructor
template <typename T>
template <typename... ArgsT>
auto ObjectPool<T>::get(ArgsT&&... args) {
  // Pool is full
  if(_recycle.empty()) {
    T& item = _pool.emplace_front(std::forward<ArgsT>(args)...);
    return HandleType(&item, Deleter(_recycle));
  }
  // Get item from the recycle box
  else {
    auto item = _recycle.back(); 
    _recycle.pop_back();
    new (item) T(std::forward<ArgsT>(args)...);
    return HandleType(item, Deleter(_recycle));
  }
}*/


// Namespace of taskflow. -----------------------------------------------------
namespace tf {

// Procedure: throw_re
template <typename... ArgsT>
inline void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
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

// Struct: MoC
template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }
  
  mutable T object; 
};

// Forward declaration
template <template<typename...> class FuncType>
class BasicNode;

template <typename NodeType>
class BasicTopology;

template <typename NodeType>
class BasicTask;

template <typename NodeType>
class BasicFlowBuilder;

template <typename NodeType>
class BasicSubflowBuilder;

template <typename Traits>
class BasicTaskflow;

// ----------------------------------------------------------------------------

// Class: BasicNode
template <template<typename...> class FuncType>
class BasicNode {

  template <typename U> friend class BasicTask;
  template <typename T> friend class BasicTaskflow;
  template <typename S> friend class BasicTopology;

  using WorkType     = FuncType<void()>;
  using SubworkType  = FuncType<void(BasicSubflowBuilder<BasicNode>&)>;
  using TopologyType = BasicTopology<BasicNode>;

  public:

    BasicNode() = default;

    template <typename C>
    BasicNode(C&&);

    const std::string& name() const;
    
    void precede(BasicNode&);

    size_t num_successors() const;
    size_t num_dependents() const;

    std::string dump() const;

  private:
    
    std::string _name;
    std::variant<WorkType, SubworkType> _work;
    std::vector<BasicNode*> _successors;
    std::atomic<int> _dependents {0};
    std::forward_list<BasicNode> _children;
    TopologyType* _topology {nullptr};

    void _dump(std::ostream&) const;
};

// Constructor
template <template<typename...> class FuncType>
template <typename C>
BasicNode<FuncType>::BasicNode(C&& c) : _work {std::forward<C>(c)} {
}

// Procedure:
template <template<typename...> class FuncType>
void BasicNode<FuncType>::precede(BasicNode& v) {
  _successors.push_back(&v);
  ++v._dependents;
}

// Function: num_successors
template <template<typename...> class FuncType>
size_t BasicNode<FuncType>::num_successors() const {
  return _successors.size();
}

// Function: dependents
template <template<typename...> class FuncType>
size_t BasicNode<FuncType>::num_dependents() const {
  return _dependents.load();
}

// Function: name
template <template<typename...> class FuncType>
const std::string& BasicNode<FuncType>::name() const {
  return _name;
}

// Function: dump
template <template<typename...> class FuncType>
std::string BasicNode<FuncType>::dump() const {
  std::ostringstream os;  
  _dump(os);
  return os.str();
}

// Function: _dump
template <template<typename...> class FuncType>
void BasicNode<FuncType>::_dump(std::ostream& os) const {
  
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
  
  if(!_children.empty()) {

    os << "subgraph cluster_";
    if(_name.empty()) os << this;
    else os << _name;
    os << " {\n";

    os << "label = \"Subflow_";
    if(_name.empty()) os << this;
    else os << _name;

    os << "\";\n" << "color=blue\n";

    for(const auto& n : _children) {
      n._dump(os);
    }
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------
  
// class: BasicTopology
template <typename NodeType>
class BasicTopology {
  
  template <typename T> friend class BasicTaskflow;

  public:

    BasicTopology(std::forward_list<NodeType>&&);

    std::string dump() const;

  private:

    std::forward_list<NodeType> _nodes;
    std::shared_future<void> _future;

    NodeType _source;
    NodeType _target;

    void _dump(std::ostream&) const;
};

// Constructor
template <typename NodeType>
BasicTopology<NodeType>::BasicTopology(std::forward_list<NodeType>&& t) : 
  _nodes(std::move(t)) {

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
  for(auto& node : _nodes) {

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
template <typename NodeType>
void BasicTopology<NodeType>::_dump(std::ostream& os) const {

  assert(_source._children.empty());
  assert(_target._children.empty());
  
  os << "digraph Topology {\n"
     << _source.dump() 
     << _target.dump();

  for(const auto& node : _nodes) {
    os << node.dump();
  }

  os << "}\n";
}
  
// Function: dump
template <typename NodeType>
std::string BasicTopology<NodeType>::dump() const { 
  std::ostringstream os;
  _dump(os);
  return os.str();
}

// ----------------------------------------------------------------------------

// Class: BasicTask
template <typename NodeType>
class BasicTask {

  template <typename U> friend class BasicFlowBuilder;
  template <typename T> friend class BasicTaskflow;

  public:
    
    BasicTask() = default;
    BasicTask(NodeType&);
    BasicTask(const BasicTask&);
    BasicTask(BasicTask&&);

    BasicTask& operator = (const BasicTask&);

    const std::string& name() const;

    size_t num_successors() const;
    size_t num_dependents() const;

    BasicTask& name(const std::string&);
    BasicTask& precede(BasicTask);
    BasicTask& broadcast(std::vector<BasicTask>&);
    BasicTask& broadcast(std::initializer_list<BasicTask>);
    BasicTask& gather(std::vector<BasicTask>&);
    BasicTask& gather(std::initializer_list<BasicTask>);

    template <typename C>
    BasicTask& work(C&&);
  
    template <typename... Bs>
    BasicTask& broadcast(Bs&&...);

    template <typename... Bs>
    BasicTask& gather(Bs&&...);

  private:

    NodeType* _node {nullptr};

    template<typename S>
    void _broadcast(S&);

    template<typename S>
    void _gather(S&);
};

// Constructor
template <typename NodeType>
BasicTask<NodeType>::BasicTask(NodeType& t) : _node {&t} {
}

// Constructor
template <typename NodeType>
BasicTask<NodeType>::BasicTask(const BasicTask& rhs) : _node {rhs._node} {
}

// Function: precede
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::precede(BasicTask tgt) {
  _node->precede(*(tgt._node));
  return *this;
}

// Function: broadcast
template <typename NodeType>
template <typename... Bs>
BasicTask<NodeType>& BasicTask<NodeType>::broadcast(Bs&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Procedure: _broadcast
template <typename NodeType>
template <typename S>
void BasicTask<NodeType>::_broadcast(S& tgts) {
  for(auto& to : tgts) {
    _node->precede(*(to._node));
  }
}
      
// Function: broadcast
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::broadcast(std::vector<BasicTask>& tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: broadcast
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::broadcast(
  std::initializer_list<BasicTask> tgts
) {
  _broadcast(tgts);
  return *this;
}

// Function: gather
template <typename NodeType>
template <typename... Bs>
BasicTask<NodeType>& BasicTask<NodeType>::gather(Bs&&... tgts) {
  (tgts.precede(*this), ...);
  return *this;
}

// Procedure: _gather
template <typename NodeType>
template <typename S>
void BasicTask<NodeType>::_gather(S& tgts) {
  for(auto& from : tgts) {
    from._node->precede(*_node);
  }
}

// Function: gather
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::gather(std::vector<BasicTask>& tgts) {
  _gather(tgts);
  return *this;
}

// Function: gather
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::gather(
  std::initializer_list<BasicTask> tgts
) {
  _gather(tgts);
  return *this;
}

// Operator =
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::operator = (const BasicTask& rhs) {
  _node = rhs._node;
  return *this;
}

// Constructor
template <typename NodeType>
BasicTask<NodeType>::BasicTask(BasicTask&& rhs) : _node{rhs._node} { 
  rhs._node = nullptr; 
}

// Function: work
template <typename NodeType>
template <typename C>
BasicTask<NodeType>& BasicTask<NodeType>::work(C&& c) {
  _node->_work = std::forward<C>(c);
  return *this;
}

// Function: name
template <typename NodeType>
BasicTask<NodeType>& BasicTask<NodeType>::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
template <typename NodeType>
const std::string& BasicTask<NodeType>::name() const {
  return _node->_name;
}

// Function: num_dependents
template <typename NodeType>
size_t BasicTask<NodeType>::num_dependents() const {
  return _node->_dependents;
}

// Function: num_successors
template <typename NodeType>
size_t BasicTask<NodeType>::num_successors() const {
  return _node->_successors.size();
}

// ----------------------------------------------------------------------------

// Class: BasicFlowBuilder
template <typename NodeType>
class BasicFlowBuilder {

  using TaskType = BasicTask<NodeType>;

  public:

    BasicFlowBuilder(std::forward_list<NodeType>&, size_t);

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
    
    void precede(TaskType, TaskType);
    void linearize(std::vector<TaskType>&);
    void linearize(std::initializer_list<TaskType>);
    void broadcast(TaskType, std::vector<TaskType>&);
    void broadcast(TaskType, std::initializer_list<TaskType>);
    void gather(std::vector<TaskType>&, TaskType);
    void gather(std::initializer_list<TaskType>, TaskType);  

    size_t num_nodes() const;

    bool empty() const;

  protected:

    std::forward_list<NodeType>& _nodes;
    size_t _num_workers;

    template <typename L>
    void _linearize(L&);
};

template <typename NodeType>    
BasicFlowBuilder<NodeType>::BasicFlowBuilder(
  std::forward_list<NodeType>& nodes, size_t num_workers
) : 
  _nodes       {nodes}, 
  _num_workers {num_workers} {
}    

// Procedure: num_nodes
template <typename NodeType>
size_t BasicFlowBuilder<NodeType>::num_nodes() const {
  return std::distance(_nodes.begin(), _nodes.end());
}

// Function: empty
template <typename NodeType>
bool BasicFlowBuilder<NodeType>::empty() const {
  return _nodes.empty();
}

// Procedure: precede
template <typename NodeType>
void BasicFlowBuilder<NodeType>::precede(TaskType from, TaskType to) {
  from._node->precede(*(to._node));
}

// Procedure: broadcast
template <typename NodeType>
void BasicFlowBuilder<NodeType>::broadcast(
  TaskType from, std::vector<TaskType>& keys
) {
  from.broadcast(keys);
}

// Procedure: broadcast
template <typename NodeType>
void BasicFlowBuilder<NodeType>::broadcast(
  TaskType from, std::initializer_list<TaskType> keys
) {
  from.broadcast(keys);
}

// Function: gather
template <typename NodeType>
void BasicFlowBuilder<NodeType>::gather(
  std::vector<TaskType>& keys, TaskType to
) {
  to.gather(keys);
}

// Function: gather
template <typename NodeType>
void BasicFlowBuilder<NodeType>::gather(
  std::initializer_list<TaskType> keys, TaskType to
) {
  to.gather(keys);
}

// Function: placeholder
template <typename NodeType>
auto BasicFlowBuilder<NodeType>::placeholder() {
  auto& node = _nodes.emplace_front();
  return TaskType(node);
}

// Function: emplace
template <typename NodeType>
template <typename C>
auto BasicFlowBuilder<NodeType>::emplace(C&& c) {
    
  // subflow task
  if constexpr(std::is_invocable_v<C, BasicSubflowBuilder<NodeType>&>) {

    using R = std::invoke_result_t<C, BasicSubflowBuilder<NodeType>&>;
    std::promise<R> p;
    auto fu = p.get_future();
  
    if constexpr(std::is_same_v<void, R>) {
      auto& node = _nodes.emplace_front([p=MoC(std::move(p)), c=std::forward<C>(c)]
      (BasicSubflowBuilder<NodeType>& fb) mutable {
        if(fb._nodes.empty()) {
          c(fb);
          if(fb.detached()) {
            p.get().set_value();
          }
        }
        else {
          p.get().set_value();
        }
      });
      return std::make_pair(TaskType(node), std::move(fu));
    }
    else {
      auto& node = _nodes.emplace_front(
      [p=MoC(std::move(p)), c=std::forward<C>(c), r=std::optional<R>()]
      (BasicSubflowBuilder<NodeType>& fb) mutable {
        if(fb._nodes.empty()) {
          r = c(fb);
          if(fb.detached()) {
            p.get().set_value(std::move(*r)); 
          }
        }
        else {
          assert(r);
          p.get().set_value(std::move(*r));
        }
      });
      return std::make_pair(TaskType(node), std::move(fu));
    }
  }
  // regular task
  else if constexpr(std::is_invocable_v<C>) {

    using R = std::invoke_result_t<C>;
    std::promise<R> p;
    auto fu = p.get_future();

    if constexpr(std::is_same_v<void, R>) {
      auto& node = _nodes.emplace_front(
        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
          c(); 
          p.get().set_value();
        }
      );
      return std::make_pair(TaskType(node), std::move(fu));
    }
    else {
      auto& node = _nodes.emplace_front(
        [p=MoC(std::move(p)), c=std::forward<C>(c)]() mutable {
          p.get().set_value(c());
        }
      );
      return std::make_pair(TaskType(node), std::move(fu));
    }
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

// Function: emplace
template <typename NodeType>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicFlowBuilder<NodeType>::emplace(C&&... cs) {
  return std::make_tuple(emplace(std::forward<C>(cs))...);
}

// Function: silent_emplace
template <typename NodeType>
template <typename C>
auto BasicFlowBuilder<NodeType>::silent_emplace(C&& c) {
  // subflow task
  if constexpr(std::is_invocable_v<C, BasicSubflowBuilder<NodeType>&>) {
    auto& n = _nodes.emplace_front(
    [c=std::forward<C>(c)] (BasicSubflowBuilder<NodeType>& fb) {
      // first time execution
      if(fb._nodes.empty()) {
        c(fb);
      }
    });
    return TaskType(n);
  }
  // regular task
  else if constexpr(std::is_invocable_v<C>) {
    auto& n = _nodes.emplace_front(std::forward<C>(c));
    return TaskType(n);
  }
  else {
    static_assert(dependent_false_v<C>, "invalid task work type");
  }
}

// Function: silent_emplace
template <typename NodeType>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicFlowBuilder<NodeType>::silent_emplace(C&&... cs) {
  return std::make_tuple(silent_emplace(std::forward<C>(cs))...);
}


// Function: parallel_for    
template <typename NodeType>
template <typename I, typename C>
auto BasicFlowBuilder<NodeType>::parallel_for(I beg, I end, C&& c, size_t g) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  if(g == 0) {
    auto d = std::distance(beg, end);
    auto w = std::max(size_t{1}, _num_workers);
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
template <typename NodeType>
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
auto BasicFlowBuilder<NodeType>::parallel_for(T& t, C&& c, size_t group) {
  return parallel_for(t.begin(), t.end(), std::forward<C>(c), group);
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename NodeType>
template <typename I, typename T>
auto BasicFlowBuilder<NodeType>::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename NodeType>
template <typename I, typename T>
auto BasicFlowBuilder<NodeType>::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename NodeType>
template <typename I, typename T, typename B, typename U>
auto BasicFlowBuilder<NodeType>::transform_reduce(
  I beg, I end, T& result, B&& bop, U&& uop
) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
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
template <typename NodeType>
template <typename I, typename T, typename B, typename P, typename U>
auto BasicFlowBuilder<NodeType>::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {

  using category = typename std::iterator_traits<I>::iterator_category;
  
  // Even partition
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
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
template <typename NodeType>
template <typename L>
void BasicFlowBuilder<NodeType>::_linearize(L& keys) {
  (void) std::adjacent_find(
    keys.begin(), keys.end(), 
    [] (auto& from, auto& to) {
      from._node->precede(*(to._node));
      return false;
    }
  );
}

// Procedure: linearize
template <typename NodeType>
void BasicFlowBuilder<NodeType>::linearize(std::vector<TaskType>& keys) {
  _linearize(keys); 
}

// Procedure: linearize
template <typename NodeType>
void BasicFlowBuilder<NodeType>::linearize(std::initializer_list<TaskType> keys) {
  _linearize(keys);
}

// Proceduer: reduce
template <typename NodeType>
template <typename I, typename T, typename B>
auto BasicFlowBuilder<NodeType>::reduce(I beg, I end, T& result, B&& op) {
  
  using category = typename std::iterator_traits<I>::iterator_category;
  
  size_t d = std::distance(beg, end);
  size_t w = std::max(size_t{1}, _num_workers);
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

// Class: BasicSubflowBuilder
template <typename NodeType>
class BasicSubflowBuilder : public BasicFlowBuilder<NodeType> {

  using BaseType = BasicFlowBuilder<NodeType>;

  public:
    
    template <typename... Args>
    BasicSubflowBuilder(Args&&...);

    void join();
    void detach();

    bool detached() const;
    bool joined() const;

  private:

    bool _detached {false};
};

// Constructor
template <typename NodeType>
template <typename... Args>
BasicSubflowBuilder<NodeType>::BasicSubflowBuilder(Args&&... args) :
  BaseType {std::forward<Args>(args)...} {
}

// Procedure: join
template <typename NodeType>
void BasicSubflowBuilder<NodeType>::join() {
  _detached = false;
}

// Procedure: detach
template <typename NodeType>
void BasicSubflowBuilder<NodeType>::detach() {
  _detached = true;
}

// Function: detached
template <typename NodeType>
bool BasicSubflowBuilder<NodeType>::detached() const {
  return _detached;
}

// Function: joined
template <typename NodeType>
bool BasicSubflowBuilder<NodeType>::joined() const {
  return !_detached;
}

// ----------------------------------------------------------------------------

// Class: BasicTaskflow
template <typename Traits>
class BasicTaskflow {
  
  public:

  using ThreadpoolType     = typename Traits::ThreadpoolType;
  using NodeType           = typename Traits::NodeType;
  using WorkType           = typename Traits::NodeType::WorkType;
  using SubworkType        = typename Traits::NodeType::SubworkType;
  using TaskType           = BasicTask<NodeType>;
  using FlowBuilderType    = BasicFlowBuilder<NodeType>;
  using SubflowBuilderType = BasicSubflowBuilder<NodeType>;
  using TopologyType       = BasicTopology<NodeType>;
 
    BasicTaskflow();
    BasicTaskflow(unsigned);
    ~BasicTaskflow();
    
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

    std::shared_future<void> dispatch();

    void precede(TaskType, TaskType);
    void linearize(std::vector<TaskType>&);
    void linearize(std::initializer_list<TaskType>);
    void broadcast(TaskType, std::vector<TaskType>&);
    void broadcast(TaskType, std::initializer_list<TaskType>);
    void gather(std::vector<TaskType>&, TaskType);
    void gather(std::initializer_list<TaskType>, TaskType);  
    void silent_dispatch();
    void wait_for_all();
    void wait_for_topologies();
    void num_workers(size_t);

    size_t num_nodes() const;
    size_t num_workers() const;
    size_t num_topologies() const;

    std::string dump() const;
    std::string dump_topologies() const;

  private:

    ThreadpoolType _threadpool;

    std::forward_list<NodeType> _nodes;
    std::forward_list<TopologyType> _topologies;

    void _schedule(NodeType&);
};

// Constructor
template <typename Traits>
BasicTaskflow<Traits>::BasicTaskflow() : 
  _threadpool{std::thread::hardware_concurrency()} {
}

// Constructor
template <typename Traits>
BasicTaskflow<Traits>::BasicTaskflow(unsigned N) : _threadpool{N} {
}

// Destructor
template <typename Traits>
BasicTaskflow<Traits>::~BasicTaskflow() {
  wait_for_topologies();
}

// Procedure: num_workers
template <typename Traits>
void BasicTaskflow<Traits>::num_workers(size_t W) {
  _threadpool.shutdown();
  _threadpool.spawn(W);
}

// Function: num_nodes
template <typename Traits>
size_t BasicTaskflow<Traits>::num_nodes() const {
  //return _nodes.size();
  return std::distance(_nodes.begin(), _nodes.end());
}

// Function: num_workers
template <typename Traits>
size_t BasicTaskflow<Traits>::num_workers() const {
  return _threadpool.num_workers();
}

// Function: num_topologies
template <typename Traits>
size_t BasicTaskflow<Traits>::num_topologies() const {
  return _topologies.size();
}

// Procedure: precede
template <typename Traits>
void BasicTaskflow<Traits>::precede(TaskType from, TaskType to) {
  from._node->precede(*(to._node));
}

// Procedure: linearize
template <typename Traits>
void BasicTaskflow<Traits>::linearize(std::vector<TaskType>& keys) {
  FlowBuilderType(_nodes, num_workers()).linearize(keys);
}

// Procedure: linearize
template <typename Traits>
void BasicTaskflow<Traits>::linearize(std::initializer_list<TaskType> keys) {
  FlowBuilderType(_nodes, num_workers()).linearize(keys);
}

// Procedure: broadcast
template <typename Traits>
void BasicTaskflow<Traits>::broadcast(TaskType from, std::vector<TaskType>& keys) {
  from.broadcast(keys);
}

// Procedure: broadcast
template <typename Traits>
void BasicTaskflow<Traits>::broadcast(TaskType from, std::initializer_list<TaskType> keys) {
  from.broadcast(keys);
}


// Function: gather
template <typename Traits>
void BasicTaskflow<Traits>::gather(std::vector<TaskType>& keys, TaskType to) {
  to.gather(keys);
}

// Function: gather
template <typename Traits>
void BasicTaskflow<Traits>::gather(std::initializer_list<TaskType> keys, TaskType to) {
  to.gather(keys);
}

// Procedure: silent_dispatch 
template <typename Traits>
void BasicTaskflow<Traits>::silent_dispatch() {

  if(_nodes.empty()) return;

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology._source);
}

// Procedure: dispatch 
template <typename Traits>
std::shared_future<void> BasicTaskflow<Traits>::dispatch() {

  if(_nodes.empty()) {
    return std::async(std::launch::deferred, [](){}).share();
  }

  auto& topology = _topologies.emplace_front(std::move(_nodes));

  // Start the taskflow
  _schedule(topology._source);
  
  return topology._future;
}

// Procedure: wait_for_all
template <typename Traits>
void BasicTaskflow<Traits>::wait_for_all() {
  if(!_nodes.empty()) {
    silent_dispatch();
  }
  wait_for_topologies();
}

// Procedure: wait_for_topologies
template <typename Traits>
void BasicTaskflow<Traits>::wait_for_topologies() {
  for(auto& t: _topologies){
    t._future.get();
  }
  _topologies.clear();
}

// Function: placeholder
template <typename Traits>
auto BasicTaskflow<Traits>::placeholder() {
  return FlowBuilderType(_nodes, num_workers()).placeholder();
}

// Function: silent_emplace
template <typename Traits>
template <typename C>
auto BasicTaskflow<Traits>::silent_emplace(C&& c) {
  return FlowBuilderType(_nodes, num_workers()).silent_emplace(std::forward<C>(c));
}

// Function: silent_emplace
template <typename Traits>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<Traits>::silent_emplace(C&&... cs) {
  return FlowBuilderType(_nodes, num_workers()).silent_emplace(std::forward<C>(cs)...);
}

// Function: emplace
template <typename Traits>
template <typename C>
auto BasicTaskflow<Traits>::emplace(C&& c) {
  return FlowBuilderType(_nodes, num_workers()).emplace(std::forward<C>(c));
}

// Function: emplace
template <typename Traits>
template <typename... C, std::enable_if_t<(sizeof...(C)>1), void>*>
auto BasicTaskflow<Traits>::emplace(C&&... cs) {
  return FlowBuilderType(_nodes, num_workers()).emplace(std::forward<C>(cs)...);
}

// Function: parallel_for    
template <typename Traits>
template <typename I, typename C>
auto BasicTaskflow<Traits>::parallel_for(I beg, I end, C&& c, size_t g) {
  return FlowBuilderType(_nodes, num_workers()).parallel_for(
    beg, end, std::forward<C>(c), g
  );
}

// Function: parallel_for
template <typename Traits>
template <typename T, typename C, std::enable_if_t<is_iterable_v<T>, void>*>
auto BasicTaskflow<Traits>::parallel_for(T& t, C&& c, size_t group) {
  return FlowBuilderType(_nodes, num_workers()).parallel_for(
    t, std::forward<C>(c), group
  );
}

// Function: reduce 
template <typename Traits>
template <typename I, typename T, typename B>
auto BasicTaskflow<Traits>::reduce(I beg, I end, T& result, B&& op) {
  return FlowBuilderType(_nodes, num_workers()).reduce(
    beg, end, result, std::forward<B>(op)
  );
}

// Function: reduce_min
// Find the minimum element over a range of items.
template <typename Traits>
template <typename I, typename T>
auto BasicTaskflow<Traits>::reduce_min(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::min(l, r);
  });
}

// Function: reduce_max
// Find the maximum element over a range of items.
template <typename Traits>
template <typename I, typename T>
auto BasicTaskflow<Traits>::reduce_max(I beg, I end, T& result) {
  return reduce(beg, end, result, [] (const auto& l, const auto& r) {
    return std::max(l, r);
  });
}

// Function: transform_reduce    
template <typename Traits>
template <typename I, typename T, typename B, typename U>
auto BasicTaskflow<Traits>::transform_reduce(
  I beg, I end, T& result, B&& bop, U&& uop
) {
  return FlowBuilderType(_nodes, num_workers()).transform_reduce(
    beg, end, result, std::forward<B>(bop), std::forward<U>(uop)
  );
}

// Function: transform_reduce    
template <typename Traits>
template <typename I, typename T, typename B, typename P, typename U>
auto BasicTaskflow<Traits>::transform_reduce(
  I beg, I end, T& result, B&& bop, P&& pop, U&& uop
) {
  return FlowBuilderType(_nodes, num_workers()).transform_reduce(
    beg, end, result, std::forward<B>(bop), std::forward<P>(pop), std::forward<U>(uop)
  );
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
template <typename Traits>
void BasicTaskflow<Traits>::_schedule(NodeType& node) {

  _threadpool.silent_async([this, &node](){

    // Here we need to fetch the num_successors first to avoid the invalid memory
    // access caused by topology clear.
    const auto num_successors = node.num_successors();
    
    // regular node type
    // The default node work type. We only need to execute the callback if any.
    if(auto index=node._work.index(); index == 0){
      if(auto &f = std::get<WorkType>(node._work); f != nullptr){
        std::invoke(f);
      }
    }
    // subflow node type 
    // The first time we enter into the subflow context, "subnodes" must be empty.
    // After executing the user's callback on subflow, there will be at least one
    // node node used as "super source". The second time we enter this context we 
    // don't have to reexecute the work again.
    else {
      assert(std::holds_alternative<SubworkType>(node._work));
      
      SubflowBuilderType fb(node._children, num_workers());

      bool empty_graph = node._children.empty();

      std::invoke(std::get<SubworkType>(node._work), fb);
      
      // Need to create a subflow
      if(empty_graph) {

        auto& S = node._children.emplace_front([](){});

        S._topology = node._topology;

        for(auto i = std::next(node._children.begin()); i != node._children.end(); ++i) {

          i->_topology = node._topology;

          if(i->num_successors() == 0) {
            i->precede(fb.detached() ? node._topology->_target : node);
          }

          if(i->num_dependents() == 0) {
            S.precede(*i);
          }
        }
        
        // this is for the case where subflow graph might be empty
        if(!fb.detached()) {
          S.precede(node);
        }

        _schedule(S);

        if(!fb.detached()) {
          return;
        }
      }
    }
    
    // At this point, the node/node storage might be destructed.
    for(size_t i=0; i<num_successors; ++i) {
      if(--(node._successors[i]->_dependents) == 0) {
        _schedule(*(node._successors[i]));
      }
    }
  });
}

// Function: dump_topology
template <typename Traits>
std::string BasicTaskflow<Traits>::dump_topologies() const {
  
  std::ostringstream os;

  for(const auto& tpg : _topologies) {
    tpg._dump(os);
  }
  
  return os.str();
}

// Function: dump
// Dumps the taskflow in graphviz. The result can be viewed at http://www.webgraphviz.com/.
template <typename Traits>
std::string BasicTaskflow<Traits>::dump() const {

  std::ostringstream os;

  os << "digraph Taskflow {\n";
  
  for(const auto& node : _nodes) {
    node._dump(os);
  }

  os << "}\n";
  
  return os.str();
}

//-----------------------------------------------------------------------------

// Taskflow traits
struct TaskflowTraits {
  using NodeType       = BasicNode<std::function>;
  using ThreadpoolType = SimpleThreadpool;
};

using Taskflow       = BasicTaskflow<TaskflowTraits>;
using Task           = typename Taskflow::TaskType;
using SubflowBuilder = typename Taskflow::SubflowBuilderType;

};  // end of namespace tf. ---------------------------------------------------

