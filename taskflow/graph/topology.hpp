#pragma once

#include "framework.hpp"

namespace tf {

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  template <template<typename...> typename E> 
  friend class BasicTaskflow; 
  friend class Framework;
  
  // TODO: make graph/framework handle uniform
  //struct GraphHandle {
  //  Graph graph;
  //};

  //struct FrameworkHandle {
  //  Framework* framework;
  //  std::function<bool()> predicate;
  //};

  public:

    Topology(Graph&&);

    template <typename C>
    Topology(Graph&&, C&&);
    
    template <typename P>
    Topology(Framework&, P&&);

    std::string dump() const;
    void dump(std::ostream&) const;

  private:

    std::variant<Graph, Framework*> _handle;

    std::promise <void> _promise;
    std::shared_future<void> _future;

    std::vector<Node*> _sources;
    std::atomic<int> _num_sinks {0};
    
    std::function<bool()> _predicate {nullptr};
    std::function<void()> _work {nullptr};
};


// Constructor
template <typename P>
inline Topology::Topology(Framework& f, P&& p): 
  _handle    {&f}, 
  _future    {_promise.get_future().share()},
  _predicate {std::forward<P>(p)} {
}


// TODO: remove duplicate code in the two constructors

// Constructor
inline Topology::Topology(Graph&& t) : 
  _handle {std::move(t)},
  _future {_promise.get_future().share()} {

  // Build the super source and super target.
  for(auto& node : std::get<Graph>(_handle)) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      _num_sinks ++;
    }
  }
}


// Constructor
template <typename C>
inline Topology::Topology(Graph&& t, C&& c) : 
  _handle {std::move(t)},
  _future {_promise.get_future().share()},
  _work   {std::forward<C>(c)} {

  // Build the super source and super target.
  for(auto& node : std::get<Graph>(_handle)) {

    node._topology = this;

    if(node.num_dependents() == 0) {
      _sources.push_back(&node);
    }

    if(node.num_successors() == 0) {
      _num_sinks ++;
    }
  }
}

// Procedure: dump
inline void Topology::dump(std::ostream& os) const {
  
  os << "digraph Topology {\n";

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
