#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/for_each.hpp>

#include "./details/graph_executor.hpp"
#include "./details/tree.hpp"
#include "./details/random_DAG.hpp"
#include "./details/tree.hpp"
#include "./details/diamond.hpp"

// ----------------------------------------------------------------------------
// Graph traversal
// ----------------------------------------------------------------------------
template <typename GRAPH, typename OPT, typename... OPT_Args>
void traversal(OPT_Args&&... args) {
  for(int i = 0; i < 13; ++i) {
    Graph* g;
    if constexpr(std::is_same_v<GRAPH, Tree>) {
      g = new Tree(::rand() % 3 + 1, ::rand() % 5 + 1);
    }
    else if constexpr(std::is_same_v<GRAPH, RandomDAG>) {
      g = new RandomDAG(::rand() % 10 + 1, ::rand() % 10 + 1, ::rand() % 10 + 1);
    }
    else if constexpr(std::is_same_v<GRAPH, Diamond>) {
      g = new Diamond(::rand() % 10 + 1, ::rand() % 10 + 1);
    }
    GraphExecutor<OPT> executor(*g, 0); 
    executor.traversal(std::forward<OPT_Args>(args)...);

    REQUIRE(g->traversed());
    delete g;
  }

}

TEST_CASE("cudaFlowCapturer.tree.Sequential") {
  traversal<Tree, tf::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.1") {
  traversal<Tree, tf::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.2") {
  traversal<Tree, tf::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.3") {
  traversal<Tree, tf::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.tree.RoundRobin.4") {
  traversal<Tree, tf::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.tree.Greedy.1") {
//  traversal<Tree, tf::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.2") {
//  traversal<Tree, tf::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.3") {
//  traversal<Tree, tf::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.tree.Greedy.4") {
//  traversal<RandomDAG, tf::cudaGreedyCapturing>(4);
//}

TEST_CASE("cudaFlowCapturer.randomDAG.Sequential") {
  traversal<RandomDAG,tf::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.1") {
  traversal<RandomDAG, tf::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.2") {
  traversal<RandomDAG, tf::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.3") {
  traversal<RandomDAG, tf::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.randomDAG.RoundRobin.4") {
  traversal<RandomDAG, tf::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.1") {
//  traversal<RandomDAG, tf::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.2") {
//  traversal<RandomDAG, tf::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.3") {
//  traversal<RandomDAG, tf::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.randomDAG.Greedy.4") {
//  traversal<RandomDAG, tf::cudaGreedyCapturing>(4);
//}

TEST_CASE("cudaFlowCapturer.diamond.Sequential") {
  traversal<Diamond, tf::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.1") {
  traversal<Diamond, tf::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.2") {
  traversal<Diamond, tf::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.3") {
  traversal<Diamond, tf::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.diamond.RoundRobin.4") {
  traversal<Diamond, tf::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.diamond.Greedy.1") {
//  traversal<Diamond, tf::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.2") {
//  traversal<Diamond, tf::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.3") {
//  traversal<Diamond, tf::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.diamond.Greedy.4") {
//  traversal<Diamond, tf::cudaGreedyCapturing>(4);
//}

//------------------------------------------------------
// dependencies
//------------------------------------------------------

template <typename OPT, typename... OPT_Args>
void dependencies(OPT_Args ...args) {
  
  for(int t = 0; t < 17; ++t) {
    int num_partitions = ::rand() % 5 + 1;
    int num_iterations = ::rand() % 7 + 1;

    Diamond g(num_partitions, num_iterations);

    tf::cudaFlowCapturer cf;
    cf.make_optimizer<OPT>(std::forward<OPT_Args>(args)...);

    int* inputs{nullptr};
    REQUIRE(cudaMallocManaged(&inputs, num_partitions * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMemset(inputs, 0, num_partitions * sizeof(int)) == cudaSuccess);

    std::vector<std::vector<tf::cudaTask>> tasks;
    tasks.resize(g.get_size());

    for(size_t l = 0; l < g.get_size(); ++l) {
      tasks[l].resize((g.get_graph())[l].size());
      for(size_t i = 0; i < (g.get_graph())[l].size(); ++i) {
        
        if(l % 2 == 1) {
          tasks[l][i] = cf.single_task([inputs, i] __device__ () {
            inputs[i]++;
          });
        }
        else {
          tasks[l][i] = cf.on([=](cudaStream_t stream){
            cuda_for_each(
              tf::cudaDefaultExecutionPolicy(stream), inputs, inputs + num_partitions, 
              [] __device__ (int& v) { v*=2; }
            );
          });
        }
      }
    }

    for(size_t l = 0; l < g.get_size() - 1; ++l) {
      for(size_t i = 0; i < (g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: g.at(l, i).out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

    tf::cudaStream stream;
    cf.run(stream);
    stream.synchronize();
    
    int result = 2;
    for(int i = 1; i < num_iterations; ++i) {
      result = result * 2 + 2;
    }

    for(int i = 0; i < num_partitions; ++i) {
      REQUIRE(inputs[i] == result);
    }

    REQUIRE(cudaFree(inputs) == cudaSuccess);
  }
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.Sequential") {
  dependencies<tf::cudaFlowSequentialOptimizer>();
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.1") {
  dependencies<tf::cudaFlowRoundRobinOptimizer>(1);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.2") {
  dependencies<tf::cudaFlowRoundRobinOptimizer>(2);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.3") {
  dependencies<tf::cudaFlowRoundRobinOptimizer>(3);
}

TEST_CASE("cudaFlowCapturer.dependencies.diamond.RoundRobin.4") {
  dependencies<tf::cudaFlowRoundRobinOptimizer>(4);
}

//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.1") {
//  dependencies<tf::cudaGreedyCapturing>(1);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.2") {
//  dependencies<tf::cudaGreedyCapturing>(2);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.3") {
//  dependencies<tf::cudaGreedyCapturing>(3);
//}
//
//TEST_CASE("cudaFlowCapturer.dependencies.diamond.Greedy.4") {
//  dependencies<tf::cudaGreedyCapturing>(4);
//}
