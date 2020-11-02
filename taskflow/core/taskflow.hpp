#pragma once

#include <stack>

#include "flow_builder.hpp"
#include "topology.hpp"

/** 
@file core/taskflow.hpp
*/

namespace tf {

// ----------------------------------------------------------------------------

/**
@class Taskflow 

@brief main entry to create a task dependency graph

*/
class Taskflow : public FlowBuilder {

  friend class Topology;
  friend class Executor;
  friend class FlowBuilder;

  struct Dumper {
    std::stack<const Taskflow*> stack;
    std::unordered_set<const Taskflow*> visited;
  };

  public:

    /**
    @brief constructs a taskflow with a given name
    */
    Taskflow(const std::string& name);

    /**
    @brief constructs a taskflow
    */
    Taskflow();

    /**
    @brief destroy the taskflow (virtual call)
    */
    virtual ~Taskflow();
    
    /**
    @brief dumps the taskflow to a DOT format through an output stream
    
    @tparam T output stream type with insertion operator (<<) defined
    @param ostream an output stream target
    */
    template <typename T>
    void dump(T& ostream) const;
    
    /**
    @brief dumps the taskflow in DOT format to a std::string
    */
    std::string dump() const;
    
    /**
    @brief queries the number of tasks in the taskflow
    */
    size_t num_tasks() const;
    
    /**
    @brief queries the emptiness of the taskflow
    */
    bool empty() const;

    /**
    @brief sets the name of the taskflow
    
    @return @c *this
    */
    void name(const std::string&); 

    /**
    @brief queries the name of the taskflow
    */
    const std::string& name() const ;
    
    /**
    @brief clears the associated task dependency graph
    */
    void clear();

    /**
    @brief applies an visitor callable to each task in the taskflow
    */
    template <typename V>
    void for_each_task(V&& visitor) const;

  private:
 
    std::string _name;
   
    Graph _graph;

    std::mutex _mtx;

    std::list<Topology> _topologies;
    
    template <typename T>
    void _dump(T&, const Taskflow*) const;
    
    template <typename T>
    void _dump(T&, const Node*, Dumper&) const;
    
    template <typename T>
    void _dump(T&, const Graph&, Dumper&) const;

#ifdef TF_ENABLE_CUDA
    template <typename T>
    void _dump(T&, const cudaGraph&, const Node*) const;
#endif
};

// Constructor
inline Taskflow::Taskflow(const std::string& name) : 
  FlowBuilder {_graph},
  _name       {name} {
}

// Constructor
inline Taskflow::Taskflow() : FlowBuilder{_graph} {
}

// Destructor
inline Taskflow::~Taskflow() {
  assert(_topologies.empty());
}

// Procedure:
inline void Taskflow::clear() {
  _graph.clear();
}

// Function: num_tasks
inline size_t Taskflow::num_tasks() const {
  return _graph.size();
}

// Function: empty
inline bool Taskflow::empty() const {
  return _graph.empty();
}

// Function: name
inline void Taskflow::name(const std::string &name) {
  _name = name;
}

// Function: name
inline const std::string& Taskflow::name() const {
  return _name;
}

// Function: for_each_task
template <typename V>
void Taskflow::for_each_task(V&& visitor) const {
  for(size_t i=0; i<_graph._nodes.size(); ++i) {
    visitor(Task(_graph._nodes[i]));
  }
}

// Procedure: dump
inline std::string Taskflow::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Function: dump
template <typename T>
void Taskflow::dump(T& os) const {
  os << "digraph Taskflow {\n";
  _dump(os, this);
  os << "}\n";
}

// Procedure: _dump
template <typename T>
void Taskflow::_dump(T& os, const Taskflow* top) const {
  
  Dumper dumper;
  
  dumper.stack.push(top);
  dumper.visited.insert(top);

  while(!dumper.stack.empty()) {
    
    auto f = dumper.stack.top();
    dumper.stack.pop();
    
    os << "subgraph cluster_p" << f << " {\nlabel=\"Taskflow: ";
    if(f->_name.empty()) os << 'p' << f;
    else os << f->_name;
    os << "\";\n";
    _dump(os, f->_graph, dumper);
    os << "}\n";
  }
}

// Procedure: _dump
template <typename T>
void Taskflow::_dump(T& os, const Node* node, Dumper& dumper) const {

  os << 'p' << node << "[label=\"";
  if(node->_name.empty()) os << 'p' << node;
  else os << node->_name;
  os << "\" ";

  // shape for node
  switch(node->_handle.index()) {

    case Node::CONDITION_TASK:
      os << "shape=diamond color=black fillcolor=aquamarine style=filled";
    break;

#ifdef TF_ENABLE_CUDA
    case Node::CUDAFLOW_TASK:
      os << " style=\"filled\""
         << " color=\"black\" fillcolor=\"purple\""
         << " fontcolor=\"white\""
         << " shape=\"folder\"";
    break;
#endif

    default:
    break;
  }

  os << "];\n";
  
  for(size_t s=0; s<node->_successors.size(); ++s) {
    if(node->_handle.index() == Node::CONDITION_TASK) {
      // case edge is dashed
      os << 'p' << node << " -> p" << node->_successors[s] 
         << " [style=dashed label=\"" << s << "\"];\n";
    }
    else {
      os << 'p' << node << " -> p" << node->_successors[s] << ";\n";
    }
  }
  
  // subflow join node
  if(node->_parent && node->_successors.size() == 0) {
    os << 'p' << node << " -> p" << node->_parent << ";\n";
  }

  switch(node->_handle.index()) {

    case Node::DYNAMIC_TASK: {
      auto& sbg = std::get<Node::DynamicTask>(node->_handle).subgraph;
      if(!sbg.empty()) {
        os << "subgraph cluster_p" << node << " {\nlabel=\"Subflow: ";
        if(node->_name.empty()) os << 'p' << node;
        else os << node->_name;

        os << "\";\n" << "color=blue\n";
        _dump(os, sbg, dumper);
        os << "}\n";
      }
    }
    break;

#ifdef TF_ENABLE_CUDA
    case Node::CUDAFLOW_TASK: {
      _dump(os, std::get<Node::cudaFlowTask>(node->_handle).graph, node);
    }
    break;
#endif

    default:
    break;
  }
}

// Procedure: _dump
template <typename T>
void Taskflow::_dump(T& os, const Graph& graph, Dumper& dumper) const {
    
  for(const auto& n : graph._nodes) {

    // regular task
    if(n->_handle.index() != Node::MODULE_TASK) {
      _dump(os, n, dumper);
    }
    // module task
    else {

      auto module = std::get<Node::ModuleTask>(n->_handle).module;

      os << 'p' << n << "[shape=box3d, color=blue, label=\"";
      if(n->_name.empty()) os << n;
      else os << n->_name;
      os << " [Taskflow: ";
      if(module->_name.empty()) os << 'p' << module;
      else os << module->_name;
      os << "]\"];\n";

      if(dumper.visited.find(module) == dumper.visited.end()) {
        dumper.visited.insert(module);
        dumper.stack.push(module);
      }

      for(const auto s : n->_successors) {
        os << 'p' << n << "->" << 'p' << s << ";\n";
      }
    }
  }
}

#ifdef TF_ENABLE_CUDA

// Procedure: dump the graph to a DOT format
template <typename T>
void Taskflow::_dump(T& os, const cudaGraph& g, const Node* root) const {
  
  // recursive dump with stack
  std::stack<std::tuple<const cudaGraph*, const cudaNode*, int>> stack;
  stack.push(std::make_tuple(&g, nullptr, 1));

  int pl = 0;

  while(!stack.empty()) {

    auto [graph, parent, l] = stack.top();
    stack.pop();

    for(int i=0; i<pl-l+1; i++) {
      os << "}\n";
    }
  
    if(parent == nullptr) {
      if(root) {
        os << "subgraph cluster_p" << root << " {\nlabel=\"cudaFlow: ";
        if(root->name().empty()) os << 'p' << root;
        else os << root->name();
        os << "\";\n" << "color=\"purple\"\n";
      }
      else {
        os << "digraph cudaFlow {\n";
      }
    }
    else {
      os << "subgraph cluster_p" << parent << " {\nlabel=\"cudaSubflow: ";
      if(parent->_name.empty()) os << 'p' << parent;
      else os << parent->_name;
      os << "\";\n" << "color=\"purple\"\n";
    }

    for(auto& v : graph->_nodes) {
      
      os << 'p' << v << "[label=\"";
      if(v->_name.empty()) {
        os << 'p' << v << "\"";
      }
      else {
        os << v->_name << "\"";
      }
          
      switch(v->_handle.index()) {
        case cudaNode::CUDA_KERNEL_TASK:
          os << " style=\"filled\""
             << " color=\"white\" fillcolor=\"black\""
             << " fontcolor=\"white\""
             << " shape=\"box3d\"";
        break;

        case cudaNode::CUDA_SUBFLOW_TASK:
          stack.push(std::make_tuple(
            &std::get<cudaNode::Subflow>(v->_handle).graph, v, l+1)
          );
          os << " style=\"filled\""
             << " color=\"black\" fillcolor=\"purple\""
             << " fontcolor=\"white\""
             << " shape=\"folder\"";
        break;

        default:
        break;
      }
  
      os << "];\n";

      for(const auto s : v->_successors) {
        os << 'p' << v << " -> " << 'p' << s << ";\n";
      }
      
      if(v->_successors.size() == 0) {
        if(parent == nullptr) {
          if(root) {
            os << 'p' << v << " -> p" << root << ";\n";
          }
        }
        else {
          os << 'p' << v << " -> p" << parent << ";\n";
        }
      }
    }
    
    // set the previous level
    pl = l;
  }

  for(int i=0; i<pl; i++) {
    os << "}\n";
  }

}

#endif

}  // end of namespace tf. ---------------------------------------------------

