// 2019/01/03 - created by Chun-Xun Lin
//   - TODO:
//     1. refactored the code (80-100 cols)
//     2. parameterized the arguments

#include <taskflow/taskflow.hpp>  
#include <random>
#include <chrono>
#include <cstring>

struct Node {

  size_t level {0};
  bool visited {false};

  std::atomic<size_t> dependents {0};
  std::vector<Node*> successors;

  void precede(Node& n) {
    successors.emplace_back(&n);
    n.dependents ++;
  }
};

// Verify all nodes are visited
void validate(Node nodes[], size_t num_nodes) {
  for(size_t i=0; i<num_nodes; i++) {
    assert(nodes[i].visited);
    assert(nodes[i].dependents == 0);
  }
}

// Reset nodes' states
void reset(Node nodes[], size_t num_nodes) {
  for(size_t i=0; i<num_nodes; i++) {
    nodes[i].visited = false;
    for(auto &n: nodes[i].successors) {
      n->dependents ++;
    }
  }
}

void traverse(Node* n, tf::SubflowBuilder& subflow) {
  assert(!n->visited);
  n->visited = true;
  for(size_t i=0; i<n->successors.size(); i++) {
    if(--(n->successors[i]->dependents) == 0) {
      n->successors[i]->level = n->level + 1;
      subflow.silent_emplace([s=n->successors[i]](tf::SubflowBuilder &subflow){ traverse(s, subflow); });
    }
  }
}

void sequential_traversal(std::vector<Node*>& src, Node nodes[], size_t num_nodes) {
  std::cout << "Profiling seq with repeat 100 times ...\n";
  auto start = std::chrono::system_clock::now();
  std::vector<Node*> active;
  for(auto repeat=100; repeat > 0; repeat --) {
    active = src;
    while(!active.empty()) {
      auto n = active.back();
      assert(!n->visited);
      n->visited = true;
      active.pop_back();
      for(auto& s: n->successors) {
        if(--s->dependents == 0) {
          s->level = n->level + 1;
          active.emplace_back(s);
        }
      }
    }
    validate(nodes, num_nodes);
    reset(nodes, num_nodes);
  }
  auto end = std::chrono::system_clock::now();
  std::cout << "Seq runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n'; 
}

void tf_traversal(std::vector<Node*>& src, Node nodes[], size_t num_nodes) {
  std::cout << "Profiling Taskflow with repeat 100 times ...\n";
  auto start = std::chrono::system_clock::now();

  tf::Taskflow tf(4);
  tf::Framework framework;
  for(size_t i=0; i<src.size(); i++) {
    framework.silent_emplace([i=i, &src](auto& subflow){ traverse(src[i], subflow); });
  }
  tf.silent_run_n(framework, 100, [&, iteration=0]() mutable {
    validate(nodes, num_nodes);
    reset(nodes, num_nodes);
  });
  tf.wait_for_all();  // block until finished
  
  auto end = std::chrono::system_clock::now();
  std::cout << "Tf runtime: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
            << '\n'; 
}

int main(int argc, char* argv[]){

  enum class Mode {TF, SEQ};

  Mode mode {Mode::TF};
  bool fully_connected {false};

  for(int i=0; i<argc; i++) {
    if(::strcmp(argv[i], "full") == 0) {
      fully_connected = true;
    }
    if(::strcmp(argv[i], "tf") == 0) {
      mode = Mode::TF;
    }
    if(::strcmp(argv[i], "seq") == 0) {
      mode = Mode::SEQ;
    }
  }

  size_t max_degree {4};
  size_t num_nodes {100000};

  // Shrink the size of graph if fully-connnected enabled
  if(fully_connected) {
    num_nodes /= 100;
  }

  Node* nodes = new Node[num_nodes];

  // Make sure nodes are in clean state
  for(size_t i=0; i<num_nodes; i++) {
    assert(!nodes[i].visited);
    assert(nodes[i].successors.empty());
    assert(nodes[i].dependents == 0);
  }

  // Create a DAG
  for(size_t i=0; i<num_nodes; i++) {
    size_t degree {0};
    for(size_t j=i+1; j<num_nodes && degree < max_degree; j++) {
      if(fully_connected || rand()%2 == 1) {
        nodes[i].precede(nodes[j]);
        if(!fully_connected) {
          degree ++;
        }
      }
    }
  }

  // Find source nodes
  std::vector<Node*> src;
  for(size_t i=0; i<num_nodes; i++) {
    if(!fully_connected) {
      assert(nodes[i].successors.size() <= max_degree);
    }
    if(nodes[i].dependents == 0) { 
      src.emplace_back(&nodes[i]);
    }
  }

  switch(mode) {
    case Mode::TF:  tf_traversal(src, nodes, num_nodes); break;
    case Mode::SEQ: sequential_traversal(src, nodes, num_nodes); break;
  };
  //validate();

  delete[] nodes;

  return 0;
}

