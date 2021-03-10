#pragma once

#include "cuda_graph.hpp"

/** 
@file cuda_optimizer.hpp
@brief %cudaFlow capturing algorithms include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// cudaCapturingBase
// ----------------------------------------------------------------------------

/**
@private

@brief class to provide helper common methods for optimization algorithms
*/
class cudaCapturingBase {

  protected:

    std::vector<cudaNode*> _toposort(cudaGraph&);
    std::vector<std::vector<cudaNode*>> _levelize(cudaGraph&);
};

// Function: _toposort
inline std::vector<cudaNode*> cudaCapturingBase::_toposort(cudaGraph& graph) {
  
  std::vector<cudaNode*> res;
  std::queue<cudaNode*> bfs;

  res.reserve(graph._nodes.size());

  // insert the first level of nodes into the queue
  for(auto& u : graph._nodes) {

    auto& hu = std::get<cudaNode::Capture>(u->_handle);
    hu.level = u->_dependents.size();

    if(hu.level == 0) {
      bfs.push(u.get());
    }
  }
  
  // levelize the graph using bfs
  while(!bfs.empty()) {

    auto u = bfs.front();
    bfs.pop();

    res.push_back(u);
    
    auto& hu = std::get<cudaNode::Capture>(u->_handle);

    for(auto v : u->_successors) {
      auto& hv = std::get<cudaNode::Capture>(v->_handle);
      if(--hv.level == 0) {
        bfs.push(v);
      }
    }
  }

  return res;
}

// Function: _levelize
inline std::vector<std::vector<cudaNode*>> 
cudaCapturingBase::_levelize(cudaGraph& graph) {

  std::queue<cudaNode*> bfs;

  size_t max_level = 0;
  
  // insert the first level of nodes into the queue
  for(auto& u : graph._nodes) {

    auto& hu = std::get<cudaNode::Capture>(u->_handle);
    hu.level = u->_dependents.size();

    if(hu.level == 0) {
      bfs.push(u.get());
    }
  }
  
  // levelize the graph using bfs
  while(!bfs.empty()) {

    auto u = bfs.front();
    bfs.pop();
    
    auto& hu = std::get<cudaNode::Capture>(u->_handle);

    for(auto v : u->_successors) {
      auto& hv = std::get<cudaNode::Capture>(v->_handle);
      if(--hv.level == 0) {
        hv.level = hu.level + 1;
        if(hv.level > max_level) {
          max_level = hv.level;
        }
        bfs.push(v);
      }
    }
  }

  // set level_graph and each node's idx
  std::vector<std::vector<cudaNode*>> level_graph(max_level+1);
  for(auto& u : graph._nodes) {
    auto& hu = std::get<cudaNode::Capture>(u->_handle);
    hu.lid = level_graph[hu.level].size();
    level_graph[hu.level].emplace_back(u.get());
    
    //for(auto s : u->_successors) {
    //  assert(hu.level < std::get<cudaNode::Capture>(s->_handle).level);
    //}
  }
  
  return level_graph;
}

// ----------------------------------------------------------------------------
// class definition: cudaSequentialCapturing
// ----------------------------------------------------------------------------

/**
@class cudaSequentialCapturing

@brief class to capture the described graph into a native cudaGraph
       using a single stream

A sequential capturing algorithm finds a topological order of 
the described graph and captures dependent GPU tasks using a single stream.
All GPU tasks run sequentially without breaking inter dependencies.
*/
class cudaSequentialCapturing : public cudaCapturingBase {

  friend class cudaFlowCapturer;

  public:
    
    /**
    @brief constructs a sequential optimizer
    */
    cudaSequentialCapturing() = default;
  
  private:

    cudaGraph_t _optimize(cudaGraph& graph);
};

inline cudaGraph_t cudaSequentialCapturing::_optimize(cudaGraph& graph) {
  // acquire per-thread stream and turn it into capture mode
  // we must use ThreadLocal mode to avoid clashing with CUDA global states
  cudaScopedPerThreadStream stream;

  cudaGraph_t native_g;

  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );

  auto ordered = _toposort(graph);
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  TF_CHECK_CUDA(
    cudaStreamEndCapture(stream, &native_g), "failed to end capture"
  );

  return native_g;
}

// ----------------------------------------------------------------------------
// class definition: cudaRoundRobinCapturing
// ----------------------------------------------------------------------------

/**
@class cudaRoundRobinCapturing

@brief class to capture the described graph into a native cudaGraph
       using a greedy round-robin algorithm on a fixed number of streams

A round-robin capturing algorithm levelizes the user-described graph
and assign streams to nodes in a round-robin order level by level.
*/
class cudaRoundRobinCapturing : public cudaCapturingBase {

  friend class cudaFlowCapturer;

  public:
    
    /**
    @brief constructs a round-robin optimizer with 4 streams by default
     */
    cudaRoundRobinCapturing();
    
    /**
    @brief constructs a round-robin optimizer with the given number of streams
     */
    cudaRoundRobinCapturing(size_t num_streams);
    
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

    cudaGraph_t _optimize(cudaGraph& graph);

    void _reset(std::vector<std::vector<cudaNode*>>& graph);

};

// Constructor
inline cudaRoundRobinCapturing::cudaRoundRobinCapturing(size_t num_streams) :
  _num_streams {num_streams} {

  if(num_streams == 0) {
    TF_THROW("number of streams must be at least one");
  }
}

// Function: num_streams
inline size_t cudaRoundRobinCapturing::num_streams() const {
  return _num_streams;
}

// Procedure: num_streams
inline void cudaRoundRobinCapturing::num_streams(size_t n) {
  if(n == 0) {
    TF_THROW("number of streams must be at least one");
  }
  _num_streams = n;
}

inline void cudaRoundRobinCapturing::_reset(std::vector<std::vector<cudaNode*>>& graph) {
  //level == global id 
  //idx == stream id we want to skip
  size_t id{0};
  for(auto& each_level: graph) {
    for(auto& node: each_level) {
      auto& hn = std::get<cudaNode::Capture>(node->_handle);
      hn.level = id++;
      hn.idx = _num_streams;
      hn.event = nullptr;
    }
  }
}

// Function: _optimize 
inline cudaGraph_t cudaRoundRobinCapturing::_optimize(cudaGraph& graph) {
  
  // levelize the graph
  auto levelized = _levelize(graph);
  
  // initialize the data structure
  _reset(levelized);

  // begin to capture
  std::vector<cudaScopedPerThreadStream> streams(_num_streams);

  TF_CHECK_CUDA(
    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );
  
  // reserve space for scoped events
  std::vector<cudaScopedPerThreadEvent> events;
  events.reserve((_num_streams >> 1) + levelized.size());
  
  // fork
  cudaEvent_t fork_event = events.emplace_back();
  TF_CHECK_CUDA(
    cudaEventRecord(fork_event, streams[0]), "faid to record fork"
  );

  for(size_t i = 1; i < streams.size(); ++i) {
    TF_CHECK_CUDA(
      cudaStreamWaitEvent(streams[i], fork_event, 0), "failed to wait on fork"
    );
  }

  // assign streams to levelized nodes in a round-robin manner
  for(auto& each_level: levelized) {
    for(auto& node: each_level) {
      auto& hn = std::get<cudaNode::Capture>(node->_handle);
      size_t sid = hn.lid % _num_streams;

      //wait events
      cudaNode* wait_node{nullptr};
      for(auto& pn: node->_dependents) {
        auto& phn = std::get<cudaNode::Capture>(pn->_handle);
        size_t psid = phn.lid % _num_streams;

        //level == global id 
        //idx == stream id we want to skip
        if(psid == hn.idx) {
          if(wait_node == nullptr || std::get<cudaNode::Capture>(wait_node->_handle).level < phn.level) {
            wait_node = pn;
          }
        }
        else if(psid != sid) {
          TF_CHECK_CUDA(
            cudaStreamWaitEvent(streams[sid], phn.event, 0), 
            "failed to wait on node's stream"
          );
        }
      }

      if(wait_node != nullptr) {
        assert(std::get<cudaNode::Capture>(wait_node->_handle).event); 
        TF_CHECK_CUDA(
          cudaStreamWaitEvent(
            streams[sid], 
            std::get<cudaNode::Capture>(wait_node->_handle).event, 
            0
          ), "failed to wait on node's stream"
        );
      }

      //capture
      hn.work(streams[sid]);

      //create/record stream
      for(auto& sn: node->_successors) {
        auto& shn = std::get<cudaNode::Capture>(sn->_handle);
        size_t ssid = shn.lid % _num_streams;
        if(ssid != sid) {
          if(!hn.event) {
            hn.event = events.emplace_back();
            TF_CHECK_CUDA(
              cudaEventRecord(hn.event, streams[sid]), "failed to record node's stream"
            );
          }
          //idx == stream id we want to skip
          shn.idx = sid;
        }
      }
    }
  }

  // join
  for(size_t i=1; i<_num_streams; ++i) {
    cudaEvent_t join_event = events.emplace_back();
    TF_CHECK_CUDA(
      cudaEventRecord(join_event, streams[i]), "failed to record join"
    );
    TF_CHECK_CUDA(
      cudaStreamWaitEvent(streams[0], join_event), "failed to wait on join"
    );
  }

  cudaGraph_t native_g;

  TF_CHECK_CUDA(
    cudaStreamEndCapture(streams[0], &native_g), "failed to end capture"
  );

  //tf::cuda_dump_graph(std::cout, native_g);
  //std::cout << '\n';

  return native_g;
}


/*class cudaGreedyCapturing: public cudaCapturingBase {
  
  friend class cudaFlowCapturer;

  public:

    cudaGreedyCapturing();

    cudaGreedyCapturing(size_t num_stream);

  private:

    size_t _num_streams{4};
  
    cudaGraph_t _optimize(cudaGraph& graph);
};

inline cudaGreedyCapturing::cudaGreedyCapturing(size_t num_streams): 
  _num_streams {num_streams} {

  if(num_streams == 0) {
    TF_THROW("number of streams must be at least one");
  }
}

inline cudaGraph_t cudaGreedyCapturing::_optimize(cudaGraph& graph) {
  // levelize the graph
  auto level_graph = _levelize(graph);
  
  // begin to capture
  std::vector<cudaScopedPerThreadStream> streams(_num_streams);

  TF_CHECK_CUDA(
    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );
  
  // reserve space for scoped events
  std::vector<cudaScopedPerThreadEvent> events;
  events.reserve((_num_streams >> 1) + level_graph.size());
  
  // fork
  cudaEvent_t fork_event = events.emplace_back();
  TF_CHECK_CUDA(
    cudaEventRecord(fork_event, streams[0]), "faid to record fork"
  );

  for(size_t i = 1; i < streams.size(); ++i) {
    TF_CHECK_CUDA(
      cudaStreamWaitEvent(streams[i], fork_event, 0), "failed to wait on fork"
    );
  }

  //assign sid to each node
  std::vector<cudaNode*> assign(streams.size());
  std::vector<cudaNode*> prev_assign(streams.size());
  std::queue<cudaNode*> remains;;
  size_t counter{0};

  //first level (we don't have any predecessors in the firset level)
  for(auto& node : level_graph[0]) {
    auto& hn = std::get<cudaNode::Capture>(node->_handle);

    hn.work(streams[counter]);
    hn.sid = counter;
    prev_assign[hn.sid] = node;
    if(++counter == streams.size()) {
      counter = 0;
    }
  }

  //other levels
  for(size_t l = 1; l < level_graph.size(); ++l) {
    for(size_t n = 0; n < level_graph[l].size(); ++n) {

      auto& node = level_graph[l][n];
      auto& hn = std::get<cudaNode::Capture>(node->_handle);
      auto& preds = node->_dependents;

      //1. try to assign the same stream as one of parent nodes'
      //maybe assigned by 2 in advance
      if(hn.sid == -1) {
        for(auto& pn: preds) {
          auto& phn = std::get<cudaNode::Capture>(pn->_handle);
          //TODO:ignore cross-level? (currently no, since we still need to create events for cross-level dependencies)
          //TODO::we may not choose the most suitable stream if the node has multiple parents
          if(assign[phn.sid] == nullptr) {
            hn.sid = phn.sid;
            assign[hn.sid] = node;
            //sgraph[hn.sid].emplace_back(node);

            ++counter;
            break;
          }
        }
      }

      if(hn.sid == -1) {
        //2. if 1. failed
        //try to find idle stream
        //however, the idle stream may be 'booked' by incoming nodes
        //we then skip such case to avoid the cost of creating event
        //in other words, we let future nodes use 1. as much as possible
        for(size_t s = 0; s < streams.size(); ++s) {
          if(assign[s] == nullptr) {
            bool is_assigned{false};

            if(prev_assign[s] != nullptr) {
              auto& pan = prev_assign[s];
              auto& pahn = std::get<cudaNode::Capture>(pan->_handle);

              for(auto& sn: pan->_successors) {
                auto& shn = std::get<cudaNode::Capture>(sn->_handle);
                if(shn.level - pahn.level == 1 && shn.sid == -1) {
                  shn.sid = s;
                  assign[s] = sn;
                  ++counter;
                  is_assigned = true;
                  break;
                }
              }
            }

            if(!is_assigned) {
              hn.sid = s;
              assign[s] = node;
              ++counter;
            }

            break;
          }
        }
      }

      //3. if 1. and 2. failed
      //emplace_back to remains
      if(hn.sid == -1) {
        remains.push(node);
      }

      if(counter == streams.size()) {
        //insert remaining nodes into remains
        for(size_t i = n + 1; i < level_graph[l].size(); ++i) {
          auto& ln = level_graph[l][i];
          auto& lhn = std::get<cudaNode::Capture>(ln->_handle);
          if(lhn.sid == -1) {
            remains.push(ln);
          }
        } 

        prev_assign = std::move(assign);
        assign.resize(streams.size());
        counter = 0;

        break;
      }
    }


    while(!remains.empty()) {
      //r1. same as 1
      auto& node = remains.front();
      auto& hn = std::get<cudaNode::Capture>(node->_handle);
      auto& preds = node->_dependents;
      remains.pop();

      for(auto& pn: preds) {
        auto& phn = std::get<cudaNode::Capture>(pn->_handle);
        //TODO: ignore cross-level?
        if(assign[phn.sid] == nullptr) {
          hn.sid = phn.sid;
          assign[hn.sid] = node;
          prev_assign[hn.sid] = node;
          //sgraph[hn.sid].emplace_back(node);
          ++counter;
        }
      }

      //if r1 failed
      //r2.
      if(hn.sid == -1) {
        for(size_t s = 0; s < streams.size(); ++s) {
          if(assign[s] == nullptr) {
            hn.sid = s;
            assign[hn.sid] = node;
            prev_assign[hn.sid] = node;
            //sgraph[hn.sid].emplace_back(node);
            ++counter;
          }
        }
      }

      if(counter == streams.size()) {
        assign.clear();
        assign.resize(streams.size());
        counter = 0;
      }
    }

    //reset
    counter = 0;
    assign.clear();
    assign.resize(streams.size());
  }

  //add events and work
  for(auto& level_nodes: level_graph) {
    for(auto& node: level_nodes) {
      auto& hn = std::get<cudaNode::Capture>(node->_handle);

      //wait
      for(auto& pn: node->_dependents) {
        auto& phn = std::get<cudaNode::Capture>(pn->_handle);
        if(phn.sid != hn.sid) {
          TF_CHECK_CUDA(
            cudaStreamWaitEvent(streams[hn.sid], phn.event), "failed to wait on the node"
          );
        }
      }

      //capture
      hn.work(streams[hn.sid]);

      //create event
      for(auto& sn : node->_successors) {
        auto& shn = std::get<cudaNode::Capture>(sn->_handle);
        if(hn.sid != shn.sid && hn.event == nullptr) {
          hn.event = events.emplace_back();
          TF_CHECK_CUDA(
            cudaEventRecord(hn.event, streams[hn.sid]), "failed to record"
          );
        }
      }
    }
  }


  // join
  for(size_t i=1; i<_num_streams; ++i) {
    cudaEvent_t join_event = events.emplace_back();
    TF_CHECK_CUDA(
      cudaEventRecord(join_event, streams[i]), "failed to record join"
    );
    TF_CHECK_CUDA(
      cudaStreamWaitEvent(streams[0], join_event), "failed to wait on join"
    );
  }

  cudaGraph_t native_g;

  TF_CHECK_CUDA(
    cudaStreamEndCapture(streams[0], &native_g), "failed to end capture"
  );

  //tf::cuda_dump_graph(std::cout, native_g);

  return native_g;
} */

}  // end of namespace tf -----------------------------------------------------

