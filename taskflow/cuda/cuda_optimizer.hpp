#pragma once

#include "cuda_graph.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaGraphOpt
// ----------------------------------------------------------------------------

class cudaGraphOpt {

  friend class SequentialOptimizer;
  friend class RoundRobinOptimizer;
  friend class GreedyOptimizer;

  private:

    cudaGraphOpt(cudaGraph& graph);

    cudaGraph& _graph;
    int _max_level{-1};
    std::vector<std::vector<cudaNode*>> _level_graph;
};

inline cudaGraphOpt::cudaGraphOpt(cudaGraph& graph): _graph{graph} {

  std::queue<cudaNode*> bfs;
  auto& nodes = _graph._nodes;

  for(auto node: nodes) {
    node->_unset_state(cudaNode::STATE_VISITED);
    //TODO: delete cudaEvent
  }

  //set level for each node
  for(auto node: nodes) {
    
    auto& cur_node_level = std::get<cudaNode::Capture>(node->_handle).level;

    if(!(node->_has_state(cudaNode::STATE_VISITED))) {
      node->_set_state(cudaNode::STATE_VISITED);
      bfs.push(node);
      cur_node_level = 0;
    }

    while(!bfs.empty()) {
      auto u = bfs.front();
      bfs.pop();
      for(auto s: u->_successors) {

        auto& suc_node_level = std::get<cudaNode::Capture>(s->_handle).level;

        suc_node_level = cur_node_level + 1;

        if(!(s->_has_state(cudaNode::STATE_VISITED))) {
          s->_set_state(cudaNode::STATE_VISITED);
          bfs.push(s);
        }
      }
    }

    _max_level = std::max(_max_level, cur_node_level);
  } 

  //set level_graph and each node's idx
  _level_graph.resize(_max_level + 1);
  for(auto node: nodes) {
    auto& cur_node = std::get<cudaNode::Capture>(node->_handle);
      
    cur_node.idx = _level_graph[cur_node.level].size();
    _level_graph[cur_node.level].emplace_back(node);
  }

}

// ----------------------------------------------------------------------------
// class definition: SequentialOptimizer
// ----------------------------------------------------------------------------

class SequentialOptimizer {

  friend cudaFlowCapturer;
  
  private:

    cudaGraph_t _optimize(cudaGraph* graph);
};

inline cudaGraph_t SequentialOptimizer::_optimize(cudaGraph* graph) {
  // acquire per-thread stream and turn it into capture mode
  // we must use ThreadLocal mode to avoid clashing with CUDA global states
  cudaScopedPerThreadStream stream;

  cudaGraph_t native_g;

  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );

  auto ordered = graph->_toposort();
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  TF_CHECK_CUDA(
    cudaStreamEndCapture(stream, &native_g), "failed to end capture"
  );

  return native_g;
}

// ----------------------------------------------------------------------------
// class definition: RoundRobinOptimizer
// ----------------------------------------------------------------------------

class RoundRobinOptimizer {

  friend cudaFlowCapturer;

  public:

    RoundRobinOptimizer(size_t num_streams) : _num_streams{num_streams} {}

  private:

    size_t _num_streams;

    cudaGraph_t _optimize(cudaGraph* graph);
};

//class GreedyOptimizer {

  //friend cudaFlowCapturer;
  
  //private:

    //size_t _num_streams;

    //GreedyOptimizer(size_t num_streams): _num_streams{num_streams} {}

    //cudaGraph_t _optimize(cudaGraph* graph);
//};


inline cudaGraph_t RoundRobinOptimizer::_optimize(cudaGraph* graph) {

  std::vector<cudaScopedPerThreadStream> streams(_num_streams);

  cudaGraph_t native_g;

  cudaGraphOpt g(*graph);

  TF_CHECK_CUDA(
    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );

  //fork
  cudaEvent_t fork_event;
  TF_CHECK_CUDA(cudaEventCreate(&fork_event), "failed to create event");
  TF_CHECK_CUDA(cudaEventRecord(fork_event, streams[0]), "faid to record event");

  for(size_t i = 1; i < streams.size(); ++i) {
    TF_CHECK_CUDA(cudaStreamWaitEvent(streams[i], fork_event, 0), "failed to wait event");
  }

  //Round-Robin
  for(auto& each_level_nodes: g._level_graph) {
    for(size_t i = 0; i < each_level_nodes.size(); ++i) {

      auto& node = each_level_nodes[i];
      auto& succs = node->_successors;
      auto& preds = node->_predecessors;

      size_t stream_id = i % _num_streams;

      //wait event in previous level
      for(size_t p = 0; p < preds.size(); ++p) {
        if((std::get<cudaNode::Capture>(preds[p]->_handle).idx % _num_streams) != stream_id) {
          TF_CHECK_CUDA(cudaStreamWaitEvent(streams[stream_id], std::get<cudaNode::Capture>(preds[p]->_handle).event, 0), "failed to wait event");
        }
      }

      std::get<cudaNode::Capture>(node->_handle).work(streams[stream_id]);  

      //create event if there is a node in the next level executed in different stream
      for(size_t k = 0; k < succs.size(); ++k) {
        if((std::get<cudaNode::Capture>(succs[k]->_handle).idx % _num_streams) != stream_id) {
          TF_CHECK_CUDA(cudaEventCreate(&(std::get<cudaNode::Capture>(node->_handle).event)), "failed to create event");
          TF_CHECK_CUDA(cudaEventRecord(std::get<cudaNode::Capture>(node->_handle).event, streams[stream_id]), "failed to record event");
          break;
        }
      }
    }
      
  }

  //join
  std::vector<cudaEvent_t> join_events{_num_streams - 1};
  for(size_t i = 0; i < join_events.size(); ++i) {
    TF_CHECK_CUDA(cudaEventCreate(&join_events[i]), "failed to create event");
    TF_CHECK_CUDA(cudaEventRecord(join_events[i], streams[i+1]), "failed to record event");
    TF_CHECK_CUDA(cudaStreamWaitEvent(streams[0], join_events[i], 0), "failed to wait event");
  }

  TF_CHECK_CUDA(
    cudaStreamEndCapture(streams[0], &native_g), "failed to end capture"
  );

  return native_g;
}

//inline cudaGraph_t GreedyOptimizer::_optimize(cudaGraph* graph) {
//}

}  // end of namespace tf -----------------------------------------------------------------------
