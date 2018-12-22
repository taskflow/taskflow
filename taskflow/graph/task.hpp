#pragma once

#include "graph.hpp"

namespace tf {

// Class: Task
class Task {

  friend class FlowBuilder;

  template <template<typename...> typename E> 
  friend class BasicTaskflow;

  public:
    
    Task() = default;
    Task(const Task&);
    Task(Task&&);

    Task& operator = (const Task&);

    const std::string& name() const;

    size_t num_successors() const;
    size_t num_dependents() const;

    Task& name(const std::string&);

    template <typename C>
    Task& work(C&&);

    template <typename... Ts>
    Task& precede(Ts&&...);
    
    template <typename... Bs>
    Task& broadcast(Bs&&...);
    
    Task& broadcast(std::vector<Task>&);
    Task& broadcast(std::initializer_list<Task>);
  
    template <typename... Bs>
    Task& gather(Bs&&...);

    Task& gather(std::vector<Task>&);
    Task& gather(std::initializer_list<Task>);

  private:
    
    Task(Node&);

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

// Function: broadcast
template <typename... Bs>
Task& Task::broadcast(Bs&&... tgts) {
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
inline Task& Task::broadcast(std::initializer_list<Task> tgts) {
  _broadcast(tgts);
  return *this;
}

// Function: precede
template <typename... Ts>
Task& Task::precede(Ts&&... tgts) {
  (_node->precede(*(tgts._node)), ...);
  return *this;
}

// Function: gather
template <typename... Bs>
Task& Task::gather(Bs&&... tgts) {
  (tgts._node->precede(*_node), ...);
  return *this;
}

// Procedure: _gather
template <typename S>
void Task::_gather(S& tgts) {
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

};  // end of namespace tf. ---------------------------------------------------
