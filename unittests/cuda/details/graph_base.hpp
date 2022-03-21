#pragma once

#include <taskflow/cuda/cudaflow.hpp>

//-------------------------------------------------------------------------
//Node
//-------------------------------------------------------------------------

struct Node {

  Node(
    size_t level, size_t idx,
    std::vector<size_t>& out_nodes
  );

  inline void mark() { *visited = true; }

  inline void unmark() { *visited = false; }

  inline bool is_visited() { return *visited; }

  inline void print_node(std::ostream& os);

  size_t level;
  size_t idx;
  bool* visited{nullptr}; //allocated by cudaMallocManaged

  std::vector<size_t> out_nodes;
};

Node::Node(
  size_t level, size_t idx,
  std::vector<size_t>& out_nodes
)
: level{level}, idx{idx},
  out_nodes{std::move(out_nodes)}
{
}

void Node::print_node(std::ostream& os) {
  os << "id: " << idx << " out_nodes: ";
  for(auto&& node: out_nodes) {
    os << node << ' ';
  }
  os << "\nStatus: " << *visited;
  os << '\n';
}

//-------------------------------------------------------------------------
// Basic Graph
//-------------------------------------------------------------------------

class Graph {

  public:

    Graph(int level): _level{level} { _graph.reserve(_level); }

    virtual ~Graph() = default;

    bool traversed();

    void print_graph(std::ostream& os);

    Node& at(int level, int idx) { return _graph[level][idx]; }

    const std::vector<std::vector<Node>>& get_graph() { return _graph; };

    size_t get_size() { return _graph.size(); };

    void allocate_nodes();

    void free_nodes();

  protected:

    std::vector<std::vector<Node>> _graph;

    bool* _visited_start{nullptr};

    int _level;

    size_t _num_nodes{0};
};

bool Graph::traversed() {
  for(auto&& nodes: _graph) {
    for(auto&& node: nodes) {
      if(!node.is_visited()) {
        return false;
      }
    }
  }
  return true;
}

void Graph::print_graph(std::ostream& os) {
  size_t l{0};
  for(auto&& nodes: _graph) {
    os << "-----------Level: " << l++ << "-------------\n";
    for(auto&& node: nodes) {
      node.print_node(os);
    }
  }
}

void Graph::allocate_nodes() {

  cudaMallocManaged(&_visited_start, sizeof(bool) * _num_nodes);
  std::memset(_visited_start, 0, sizeof(bool) * _num_nodes);

  bool* v = _visited_start;

  for(int l = 0; l < _level; ++l) {
    for(size_t i = 0; i < _graph[l].size(); ++i) {
      _graph[l][i].visited =  v++;
    }
  }
}

void Graph::free_nodes() {
  cudaFree(_visited_start);
}

