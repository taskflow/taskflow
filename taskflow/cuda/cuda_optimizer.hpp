#pragma once

#include "cuda_graph.hpp"

/**
@file cuda_optimizer.hpp
@brief %cudaFlow capturing algorithms include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaFlowOptimizerBase
// ----------------------------------------------------------------------------

/**
@private

@brief class to provide helper common methods for optimization algorithms
*/
class cudaFlowOptimizerBase {

  protected:

    std::vector<cudaFlowNode*> _toposort(cudaFlowGraph&);
    std::vector<std::vector<cudaFlowNode*>> _levelize(cudaFlowGraph&);
};

// Function: _toposort
inline std::vector<cudaFlowNode*> cudaFlowOptimizerBase::_toposort(cudaFlowGraph& graph) {

  std::vector<cudaFlowNode*> res;
  std::queue<cudaFlowNode*> bfs;

  res.reserve(graph._nodes.size());

  // insert the first level of nodes into the queue
  for(auto& u : graph._nodes) {

    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
    hu->level = u->_dependents.size();

    if(hu->level == 0) {
      bfs.push(u.get());
    }
  }

  // levelize the graph using bfs
  while(!bfs.empty()) {

    auto u = bfs.front();
    bfs.pop();

    res.push_back(u);

    for(auto v : u->_successors) {
      auto hv = std::get_if<cudaFlowNode::Capture>(&v->_handle);
      if(--hv->level == 0) {
        bfs.push(v);
      }
    }
  }

  return res;
}

// Function: _levelize
inline std::vector<std::vector<cudaFlowNode*>>
cudaFlowOptimizerBase::_levelize(cudaFlowGraph& graph) {

  std::queue<cudaFlowNode*> bfs;

  size_t max_level = 0;

  // insert the first level of nodes into the queue
  for(auto& u : graph._nodes) {

    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
    hu->level = u->_dependents.size();

    if(hu->level == 0) {
      bfs.push(u.get());
    }
  }

  // levelize the graph using bfs
  while(!bfs.empty()) {

    auto u = bfs.front();
    bfs.pop();

    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);

    for(auto v : u->_successors) {
      auto hv = std::get_if<cudaFlowNode::Capture>(&v->_handle);
      if(--hv->level == 0) {
        hv->level = hu->level + 1;
        if(hv->level > max_level) {
          max_level = hv->level;
        }
        bfs.push(v);
      }
    }
  }

  // set level_graph and each node's idx
  std::vector<std::vector<cudaFlowNode*>> level_graph(max_level+1);
  for(auto& u : graph._nodes) {
    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
    hu->lid = level_graph[hu->level].size();
    level_graph[hu->level].emplace_back(u.get());

    //for(auto s : u->_successors) {
    //  assert(hu.level < std::get_if<cudaFlowNode::Capture>(&s->_handle)->level);
    //}
  }

  return level_graph;
}

// ----------------------------------------------------------------------------
// class definition: cudaFlowSequentialOptimizer
// ----------------------------------------------------------------------------

/**
@class cudaFlowSequentialOptimizer

@brief class to capture a CUDA graph using a sequential stream

A sequential capturing algorithm finds a topological order of
the described graph and captures dependent GPU tasks using a single stream.
All GPU tasks run sequentially without breaking inter dependencies.
*/
class cudaFlowSequentialOptimizer : public cudaFlowOptimizerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief constructs a sequential optimizer
    */
    cudaFlowSequentialOptimizer() = default;

  private:

    cudaGraph_t _optimize(cudaFlowGraph& graph);
};

inline cudaGraph_t cudaFlowSequentialOptimizer::_optimize(cudaFlowGraph& graph) {

  // acquire per-thread stream and turn it into capture mode
  // we must use ThreadLocal mode to avoid clashing with CUDA global states
  
  cudaStream stream;

  stream.begin_capture(cudaStreamCaptureModeThreadLocal);

  auto ordered = _toposort(graph);
  for(auto node : ordered) {
    std::get_if<cudaFlowNode::Capture>(&node->_handle)->work(stream);
  }
  
  return stream.end_capture();
}

// ----------------------------------------------------------------------------
// class definition: cudaFlowLinearOptimizer
// ----------------------------------------------------------------------------

/**
@class cudaFlowLinearOptimizer

@brief class to capture a linear CUDA graph using a sequential stream

A linear capturing algorithm is a special case of tf::cudaFlowSequentialOptimizer
and assumes the input task graph to be a single linear chain of tasks
(i.e., a straight line).
This assumption allows faster optimization during the capturing process.
If the input task graph is not a linear chain, the behavior is undefined.
*/
class cudaFlowLinearOptimizer : public cudaFlowOptimizerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief constructs a linear optimizer
    */
    cudaFlowLinearOptimizer() = default;

  private:

    cudaGraph_t _optimize(cudaFlowGraph& graph);
};

inline cudaGraph_t cudaFlowLinearOptimizer::_optimize(cudaFlowGraph& graph) {

  // acquire per-thread stream and turn it into capture mode
  // we must use ThreadLocal mode to avoid clashing with CUDA global states
  cudaStream stream;

  stream.begin_capture(cudaStreamCaptureModeThreadLocal);

  // find the source node
  cudaFlowNode* src {nullptr};
  for(auto& u : graph._nodes) {
    if(u->_dependents.size() == 0) {
      src = u.get();
      while(src) {
        std::get_if<cudaFlowNode::Capture>(&src->_handle)->work(stream);
        src = src->_successors.empty() ? nullptr : src->_successors[0];
      }
      break;
    }
    // ideally, there should be only one source
  }

  return stream.end_capture();
}

// ----------------------------------------------------------------------------
// class definition: cudaFlowRoundRobinOptimizer
// ----------------------------------------------------------------------------

/**
@class cudaFlowRoundRobinOptimizer

@brief class to capture a CUDA graph using a round-robin algorithm

A round-robin capturing algorithm levelizes the user-described graph
and assign streams to nodes in a round-robin order level by level.
The algorithm is based on the following paper published in Euro-Par 2021:
  + Dian-Lun Lin and Tsung-Wei Huang, &quot;Efficient GPU Computation using %Task Graph Parallelism,&quot; <i>European Conference on Parallel and Distributed Computing (Euro-Par)</i>, 2021

The round-robin optimization algorithm is best suited for large %cudaFlow graphs
that compose hundreds of or thousands of GPU operations
(e.g., kernels and memory copies) with many of them being able to run in parallel.
You can configure the number of streams to the optimizer to adjust the
maximum kernel currency in the captured CUDA graph.
*/
class cudaFlowRoundRobinOptimizer : public cudaFlowOptimizerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief constructs a round-robin optimizer with 4 streams by default
     */
    cudaFlowRoundRobinOptimizer() = default;

    /**
    @brief constructs a round-robin optimizer with the given number of streams
     */
    explicit cudaFlowRoundRobinOptimizer(size_t num_streams);
    
    /**
    @brief queries the number of streams used by the optimizer
     */
    size_t num_streams() const;

    /**
    @brief sets the number of streams used by the optimizer
     */
    void num_streams(size_t n);

  private:

    size_t _num_streams {4};

    cudaGraph_t _optimize(cudaFlowGraph& graph);

    void _reset(std::vector<std::vector<cudaFlowNode*>>& graph);

};

// Constructor
inline cudaFlowRoundRobinOptimizer::cudaFlowRoundRobinOptimizer(size_t num_streams) :
  _num_streams {num_streams} {

  if(num_streams == 0) {
    TF_THROW("number of streams must be at least one");
  }
}

// Function: num_streams
inline size_t cudaFlowRoundRobinOptimizer::num_streams() const {
  return _num_streams;
}

// Procedure: num_streams
inline void cudaFlowRoundRobinOptimizer::num_streams(size_t n) {
  if(n == 0) {
    TF_THROW("number of streams must be at least one");
  }
  _num_streams = n;
}

inline void cudaFlowRoundRobinOptimizer::_reset(
  std::vector<std::vector<cudaFlowNode*>>& graph
) {
  //level == global id
  //idx == stream id we want to skip
  size_t id{0};
  for(auto& each_level: graph) {
    for(auto& node: each_level) {
      auto hn = std::get_if<cudaFlowNode::Capture>(&node->_handle);
      hn->level = id++;
      hn->idx = _num_streams;
      hn->event = nullptr;
    }
  }
}

// Function: _optimize
inline cudaGraph_t cudaFlowRoundRobinOptimizer::_optimize(cudaFlowGraph& graph) {

  // levelize the graph
  auto levelized = _levelize(graph);

  // initialize the data structure
  _reset(levelized);

  // begin to capture
  std::vector<cudaStream> streams(_num_streams);

  streams[0].begin_capture(cudaStreamCaptureModeThreadLocal);
  
  // reserve space for scoped events
  std::vector<cudaEvent> events;
  events.reserve((_num_streams >> 1) + levelized.size());

  // fork
  cudaEvent_t fork_event = events.emplace_back();
  streams[0].record(fork_event);

  for(size_t i = 1; i < streams.size(); ++i) {
    streams[i].wait(fork_event);
  }

  // assign streams to levelized nodes in a round-robin manner
  for(auto& each_level: levelized) {
    for(auto& node: each_level) {
      auto hn = std::get_if<cudaFlowNode::Capture>(&node->_handle);
      size_t sid = hn->lid % _num_streams;

      //wait events
      cudaFlowNode* wait_node{nullptr};
      for(auto& pn: node->_dependents) {
        auto phn = std::get_if<cudaFlowNode::Capture>(&pn->_handle);
        size_t psid = phn->lid % _num_streams;

        //level == global id
        //idx == stream id we want to skip
        if(psid == hn->idx) {
          if(wait_node == nullptr ||
             std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->level < phn->level) {
            wait_node = pn;
          }
        }
        else if(psid != sid) {
          streams[sid].wait(phn->event);
        }
      }

      if(wait_node != nullptr) {
        assert(std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->event); 
        streams[sid].wait(std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->event);
      }

      //capture
      hn->work(streams[sid]);

      //create/record stream
      for(auto& sn: node->_successors) {
        auto shn = std::get_if<cudaFlowNode::Capture>(&sn->_handle);
        size_t ssid = shn->lid % _num_streams;
        if(ssid != sid) {
          if(!hn->event) {
            hn->event = events.emplace_back();
            streams[sid].record(hn->event);
          }
          //idx == stream id we want to skip
          shn->idx = sid;
        }
      }
    }
  }

  // join
  for(size_t i=1; i<_num_streams; ++i) {
    cudaEvent_t join_event = events.emplace_back();
    streams[i].record(join_event);
    streams[0].wait(join_event);
  }

  return streams[0].end_capture();
}


}  // end of namespace tf -----------------------------------------------------

