#pragma once

#include "framework.hpp"

namespace tf {

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow; 
  friend class Framework;

  public:

    Topology(Graph&&);

    template <typename C>
    Topology(Graph&&, C&&);

    Topology(Framework&, size_t);

    std::string dump() const;
    void dump(std::ostream&) const;

  private:

    std::variant<Graph, Framework*> _handle;

    std::promise <void> _promise;
    size_t _repeat;

    std::shared_future<void> _future;

    std::vector<Node*> _sources;

    Node _target;
};


// Constructor
inline Topology::Topology(Framework& f, size_t repeat): _handle(&f), _repeat(repeat) {
  _future = _promise.get_future().share();
}


// TODO: remove duplicate code in the two constructors

// Constructor
inline Topology::Topology(Graph&& t) : 
  _handle(std::move(t)) {

  _target._topology = this;
  
  _future = _promise.get_future().share();

  _target._work = [this] () mutable { 
    this->_promise.set_value(); 
  };

  // Build the super source and super target.
  for(auto& node : std::get<Graph>(_handle)) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      node.precede(_target);
    }
  }
}


// Constructor
template <typename C>
inline Topology::Topology(Graph&& t, C&& c) : 
  _handle(std::move(t)) {

  _target._topology = this;
  
  _future = _promise.get_future().share();

  _target._work = [this, c{std::forward<C>(c)}] () mutable { 
    this->_promise.set_value();
    c();
  };

  // Build the super source and super target.
  for(auto& node : std::get<Graph>(_handle)) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      node.precede(_target);
    }
  }
}

// Procedure: dump
inline void Topology::dump(std::ostream& os) const {

  assert(!(_target._subgraph));
  
  os << "digraph Topology {\n"
     << _target.dump();

  std::visit(Functors{
    [&] (const Graph& graph) {
      for(const auto& node : graph) {
        node.dump(os);
      }
    },
    [&] (const Framework* framework) {
      for(const auto& node : framework->_graph) {
        node.dump(os);
      }
    }
  }, _handle);

  os << "}\n";
}

  
// Function: dump
inline std::string Topology::dump() const { 
  std::ostringstream os;
  dump(os);
  return os.str();
}

}  // end of namespace tf. ----------------------------------------------------
