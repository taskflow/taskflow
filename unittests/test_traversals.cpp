#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Graph generation
// --------------------------------------------------------

struct Node {

  std::string name;
  size_t idx   {0};
  size_t level {0};
  bool visited {false};

  std::atomic<size_t> dependents {0};
  std::vector<Node*> successors;

  void precede(Node& n) {
    successors.emplace_back(&n);
    n.dependents ++;
  }
};

std::unique_ptr<Node[]> make_dag(size_t num_nodes, size_t max_degree) {

  std::unique_ptr<Node[]> nodes(new Node[num_nodes]);

  // Make sure nodes are in clean state
  for(size_t i=0; i<num_nodes; i++) {
    nodes[i].idx = i;
    nodes[i].name = std::to_string(i);
    REQUIRE(!nodes[i].visited);
    REQUIRE(nodes[i].successors.empty());
    REQUIRE(nodes[i].dependents == 0);
  }

  // Create a DAG
  for(size_t i=0; i<num_nodes; i++) {
    size_t degree {0};
    for(size_t j=i+1; j<num_nodes && degree < max_degree; j++) {
      if(j%2 == 1) {
        nodes[i].precede(nodes[j]);
        degree ++;
      }
    }
  }

  return nodes;
}

std::unique_ptr<Node[]> make_chain(size_t num_nodes) {

  std::unique_ptr<Node[]> nodes(new Node[num_nodes]);

  // Make sure nodes are in clean state
  for(size_t i=0; i<num_nodes; i++) {
    nodes[i].idx = i;
    nodes[i].name = std::to_string(i);
    REQUIRE(!nodes[i].visited);
    REQUIRE(nodes[i].successors.empty());
    REQUIRE(nodes[i].dependents == 0);
  }

  // Create a DAG
  for(size_t i=1; i<num_nodes; i++) {
    nodes[i-1].precede(nodes[i]);
  }

  return nodes;
}

// --------------------------------------------------------
// Testcase: StaticTraversal
// --------------------------------------------------------
TEST_CASE("StaticTraversal" * doctest::timeout(300)) {

  size_t max_degree = 4;
  size_t num_nodes = 1000;

  for(unsigned w=1; w<=4; w++) {

    auto nodes = make_dag(num_nodes, max_degree);

    tf::Taskflow tf;
    tf::Executor executor(w);

    std::atomic<size_t> level(0);
    std::vector<tf::Task> tasks;

    for(size_t i=0; i<num_nodes; ++i) {
      auto task = tf.emplace([&level, v=&(nodes[i])](){
        v->level = ++level;
        v->visited = true;
        for(size_t j=0; j<v->successors.size(); ++j) {
          v->successors[j]->dependents.fetch_sub(1);
        }
      }).name(nodes[i].name);

      tasks.push_back(task);
    }

    for(size_t i=0; i<num_nodes; ++i) {
      for(size_t j=0; j<nodes[i].successors.size(); ++j) {
        tasks[i].precede(tasks[nodes[i].successors[j]->idx]);
      }
    }

    executor.run(tf).wait();  // block until finished

    for(size_t i=0; i<num_nodes; i++) {
      REQUIRE(nodes[i].visited);
      REQUIRE(nodes[i].dependents == 0);
      for(size_t j=0; j<nodes[i].successors.size(); ++j) {
        REQUIRE(nodes[i].level < nodes[i].successors[j]->level);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: DynamicTraversal
// --------------------------------------------------------
TEST_CASE("DynamicTraversal" * doctest::timeout(300)) {

  std::atomic<size_t> level;

  std::function<void(Node*, tf::Subflow&)> traverse;

  traverse = [&] (Node* n, tf::Subflow& subflow) {
    REQUIRE(!n->visited);
    n->visited = true;
    size_t S = n->successors.size();
    for(size_t i=0; i<S; i++) {
      if(n->successors[i]->dependents.fetch_sub(1) == 1) {
        n->successors[i]->level = ++level;
        subflow.emplace([s=n->successors[i], &traverse](tf::Subflow &subflow2){
          traverse(s, subflow2);
        });
      }
    }
  };

  size_t max_degree = 4;
  size_t num_nodes = 1000;

  for(unsigned w=1; w<=4; w++) {

    auto nodes = make_dag(num_nodes, max_degree);

    std::vector<Node*> src;
    for(size_t i=0; i<num_nodes; i++) {
      if(nodes[i].dependents == 0) {
        src.emplace_back(&(nodes[i]));
      }
    }

    level = 0;

    tf::Taskflow tf;
    tf::Executor executor(w);

    for(size_t i=0; i<src.size(); i++) {
      tf.emplace([s=src[i], &traverse](tf::Subflow& subflow){
        traverse(s, subflow);
      });
    }

    executor.run(tf).wait();  // block until finished

    for(size_t i=0; i<num_nodes; i++) {
      REQUIRE(nodes[i].visited);
      REQUIRE(nodes[i].dependents == 0);
      for(size_t j=0; j<nodes[i].successors.size(); ++j) {
        REQUIRE(nodes[i].level < nodes[i].successors[j]->level);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: RecursiveTraversal
// --------------------------------------------------------
//TEST_CASE("RecursiveTraversal" * doctest::timeout(300)) {
//
//  std::atomic<size_t> level;
//
//  std::function<void(Node*, tf::Subflow&)> traverse;
//
//  traverse = [&] (Node* n, tf::Subflow& subflow) {
//    REQUIRE(!n->visited);
//    n->visited = true;
//    size_t S = n->successors.size();
//    for(size_t i=0; i<S; i++) {
//      if(n->successors[i]->dependents.fetch_sub(1) == 1) {
//        n->successors[i]->level = ++level;
//        subflow.emplace([s=n->successors[i], &traverse](tf::Subflow &subflow){
//          traverse(s, subflow);
//        });
//      }
//    }
//  };
//
//  size_t num_nodes = 1000;
//
//  for(unsigned w=1; w<=4; w++) {
//
//    auto nodes = make_chain(num_nodes);
//
//    std::vector<Node*> src;
//    for(size_t i=0; i<num_nodes; i++) {
//      if(nodes[i].dependents == 0) {
//        src.emplace_back(&(nodes[i]));
//      }
//    }
//
//    level = 0;
//
//    tf::Taskflow tf;
//    tf::Executor executor(w);
//
//    for(size_t i=0; i<src.size(); i++) {
//      tf.emplace([s=src[i], &traverse](tf::Subflow& subflow){
//        traverse(s, subflow);
//      });
//    }
//
//    executor.run(tf).wait();  // block until finished
//
//    for(size_t i=0; i<num_nodes; i++) {
//      REQUIRE(nodes[i].visited);
//      REQUIRE(nodes[i].dependents == 0);
//      for(size_t j=0; j<nodes[i].successors.size(); ++j) {
//        REQUIRE(nodes[i].level < nodes[i].successors[j]->level);
//      }
//    }
//  }
//}

// --------------------------------------------------------
// Testcase: ParallelTraversal
// --------------------------------------------------------

/*void parallel_traversal(unsigned num_threads) {

  tf::Executor executor(num_threads);

  std::vector<std::thread> threads;

  for(unsigned t=0; t<num_threads; ++t) {

    threads.emplace_back([&](){

      std::atomic<size_t> level {0};

      size_t max_degree = 4;
      size_t num_nodes = 1000;

      auto nodes = make_dag(num_nodes, max_degree);

      std::vector<Node*> src;
      for(size_t i=0; i<num_nodes; i++) {
        if(nodes[i].dependents == 0) {
          src.emplace_back(&(nodes[i]));
        }
      }

      std::function<void(Node*, tf::Subflow&)> traverse;

      traverse = [&] (Node* n, tf::Subflow& subflow) {
        REQUIRE(!n->visited);
        n->visited = true;
        size_t S = n->successors.size();
        for(size_t i=0; i<S; i++) {
          if(n->successors[i]->dependents.fetch_sub(1) == 1) {
            n->successors[i]->level = ++level;
            subflow.emplace([s=n->successors[i], &traverse](tf::Subflow &subflow){
              traverse(s, subflow);
            });
          }
        }
      };

      tf::Taskflow tf;

      for(size_t i=0; i<src.size(); i++) {
        tf.emplace([s=src[i], &traverse](tf::Subflow& subflow){
          traverse(s, subflow);
        });
      }

      executor.run(tf).wait();  // block until finished

      for(size_t i=0; i<num_nodes; i++) {
        REQUIRE(nodes[i].visited);
        REQUIRE(nodes[i].dependents == 0);
        for(size_t j=0; j<nodes[i].successors.size(); ++j) {
          REQUIRE(nodes[i].level < nodes[i].successors[j]->level);
        }
      }
    });
  }

  for(auto& thread : threads) thread.join();
}

TEST_CASE("ParallelTraversal.1" * doctest::timeout(300)) {
  parallel_traversal(1);
}

TEST_CASE("ParallelTraversal.2" * doctest::timeout(300)) {
  parallel_traversal(2);
}

TEST_CASE("ParallelTraversal.3" * doctest::timeout(300)) {
  parallel_traversal(3);
}

TEST_CASE("ParallelTraversal.4" * doctest::timeout(300)) {
  parallel_traversal(4);
}

TEST_CASE("ParallelTraversal.5" * doctest::timeout(300)) {
  parallel_traversal(5);
}

TEST_CASE("ParallelTraversal.6" * doctest::timeout(300)) {
  parallel_traversal(6);
}

TEST_CASE("ParallelTraversal.7" * doctest::timeout(300)) {
  parallel_traversal(7);
}

TEST_CASE("ParallelTraversal.8" * doctest::timeout(300)) {
  parallel_traversal(8);
}
*/


