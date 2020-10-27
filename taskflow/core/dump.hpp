#pragma once

#include "graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// dump definitions for cudaGraph
// ----------------------------------------------------------------------------

#ifdef TF_ENABLE_CUDA

// Procedure: dump the graph to a DOT format
template <typename T>
void cudaGraph::dump(T& os, const Node* root) const {
  
  // recursive dump with stack
  std::stack<std::tuple<const cudaGraph*, const cudaNode*, int>> stack;
  stack.push(std::make_tuple(this, nullptr, 1));

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

        case cudaNode::CUDA_CHILDFLOW_TASK:
          stack.push(std::make_tuple(
            &std::get<cudaNode::Childflow>(v->_handle).graph, v, l+1)
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




}  // end of namespace tf -----------------------------------------------------





