#pragma once

#include "./graph_base.hpp"
#include <taskflow/cuda/cudaflow.hpp>
#include <cassert>

template <typename OPT>
class GraphExecutor {

  public:

    GraphExecutor(Graph& graph, int dev_id = 0);

    template <typename... OPT_Args>
    void traversal(OPT_Args&&... args);

  private:

    int _dev_id;

    Graph& _g;

};

template <typename OPT>
GraphExecutor<OPT>::GraphExecutor(Graph& graph, int dev_id): _g{graph}, _dev_id{dev_id} {
  //TODO: why we cannot put cuda lambda function here?
}

template <typename OPT>
template <typename... OPT_Args>
void GraphExecutor<OPT>::traversal(OPT_Args&&... args) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  taskflow.emplace([this, args...]() {

    tf::cudaFlowCapturer cf;

    cf.make_optimizer<OPT>(args...);

    std::vector<std::vector<tf::cudaTask>> tasks;
    tasks.resize(_g.get_graph().size());

    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        bool* v = _g.at(l, i).visited;
        tasks[l][i] = cf.single_task([v] __device__ () {
          *v = true;
        });
      }
    }

    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

    tf::cudaStream stream;
    cf.run(stream);
    stream.synchronize();

  }).name("traverse");

  //auto check_t = taskflow.emplace([this](){
    //assert(_g.traversed());
  //});

  //trav_t.precede(check_t);

  executor.run(taskflow).wait();
}

