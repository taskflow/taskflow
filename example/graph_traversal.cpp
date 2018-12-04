#include <taskflow/taskflow.hpp>  
#include <random>
#include <chrono>
#include <cstring>

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


void traverse(Node* n, tf::SubflowBuilder& subflow) {
  assert(!n->visited);
  n->visited = true;
  for(size_t i=0; i<n->successors.size(); i++) {
    if(--(n->successors[i]->dependents) == 0) {
      n->successors[i]->level = n->level + 1;
      subflow.silent_emplace([s=n->successors[i]](auto &subflow){ traverse(s, subflow); });
    }
  }
}



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

void tf_traversal(std::vector<Node*>& src) {
  auto start = std::chrono::system_clock::now();

  tf::Taskflow tf(4);
  for(size_t i=0; i<src.size(); i++) {
    tf.silent_emplace([i=i, &src](auto& subflow){ traverse(src[i], subflow); });
  }
  tf.wait_for_all();  // block until finished

  auto end = std::chrono::system_clock::now();
  std::cout << "Tf runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n'; 
}



int main(int argc, char* argv[]){

  enum class Mode {TF, SEQ};

  Mode mode {Mode::TF};
  bool fully_connected {false};

  for(auto i=0; i<argc; i++) {
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

  std::srand(1);
  constexpr size_t max_degree {4};
  constexpr size_t num_nodes {500000};

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
    case Mode::TF:  tf_traversal(src); break;
    case Mode::SEQ: sequential_traversal(src); break;
  };
  validate();

  delete[] nodes;

  return 0;
}

