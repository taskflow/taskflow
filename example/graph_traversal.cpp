// A simple example to capture the following task dependencies.
//
// TaskA---->TaskB---->TaskD
// TaskA---->TaskC---->TaskD

#include <taskflow/taskflow.hpp>  // the only include you need
#include <random>
#include <chrono>

struct Node {
  size_t level {0};
  bool visited {false};

  std::atomic<size_t> dependents {0};
  std::vector<Node*> successors;

  void precede(auto& n) {
    successors.emplace_back(&n);
    n.dependents ++;
  }
};

void sequential_traversal(std::vector<Node*>& src) {
  auto start = std::chrono::system_clock::now();
  while(!src.empty()) {
    auto n = src.back();
    assert(!n->visited);
    n->visited = true;
    src.pop_back();
    for(auto& s: n->successors) {
      if(--s->dependents == 0) {
        s->level = n->level + 1;
        src.emplace_back(s);
      }
    }
  }
  auto end = std::chrono::system_clock::now();
  std::cout << "Seq runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n'; 
}

void traverse(Node* n, tf::SubflowBuilder& subflow) {
  assert(!n->visited);
  n->visited = true;
  for(size_t i=0; i<n->successors.size(); i++) {
    if(--(n->successors[i]->dependents) == 0) {
      subflow.silent_emplace([s=n->successors[i]](auto &subflow){ traverse(s, subflow); });
    }
  }
}



int main(){

  std::srand(1);
  constexpr size_t max_degree {4};
  constexpr size_t num_nodes {10000000};

  auto nodes = new Node[num_nodes];

  // A lambda to verify all nodes are visited
  auto validate = [&nodes](){
    for(size_t i=0; i<num_nodes; i++) {
      assert(nodes[i].visited);
      assert(nodes[i].dependents == 0);
    }
  };

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
      if(rand()%2 == 1) {
        nodes[i].precede(nodes[j]);
        degree ++;
      }
    }
  }

  // Find source nodes
  std::vector<Node*> src;
  for(size_t i=0; i<num_nodes; i++) {
    assert(nodes[i].successors.size() <= 4);
    if(nodes[i].dependents == 0) { 
      src.emplace_back(&nodes[i]);
    }
  }
  //std::cout << "# of source = " << src.size() << '\n';

  //sequential_traversal(src);
  //validate();
  //return 0;


  tf::Taskflow tf(4);
  for(size_t i=0; i<src.size(); i++) {
    tf.silent_emplace([i=i, &src](auto& subflow){ traverse(src[i], subflow); });
  }
  
  auto start = std::chrono::system_clock::now();
  tf.wait_for_all();  // block until finished
  auto end = std::chrono::system_clock::now();
  validate();
  std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n'; 

  delete[] nodes;

  return 0;
}

