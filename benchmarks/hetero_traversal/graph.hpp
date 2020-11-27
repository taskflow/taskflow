#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>
#include <fstream>

struct pair_hash {
  template <typename T1, typename T2>
  size_t operator ()(const std::pair<T1, T2>& pair) const {
    auto h1 = std::hash<T1>()(pair.first);
    auto h2 = std::hash<T2>()(pair.second);
    return h1^h2;
  }
};

struct Graph {

  struct Node {
    int v, g;
  };

  struct Edge {
    int u, v;
  };

  int num_nodes;
  int num_edges;
  int num_gpus;

  std::vector<Edge> edges;
  std::vector<Node> nodes;

  Graph(const std::string& path) :
    num_gpus  {static_cast<int>(tf::cuda_get_num_devices())} {

    std::ifstream ifs(path);

    if(!ifs) throw std::runtime_error("failed to open the file");
    
    ifs >> num_nodes >> num_edges;

    nodes.resize(num_nodes);
    for(int i=0; i<num_nodes; ++i) {
      nodes[i].v = i;
      ifs >> nodes[i].g;
    }

    for(int i=0; i<num_edges; ++i) {
      Edge e;
      ifs >> e.u >> e.v;
      edges.push_back(e);
    }
  }

  Graph(int V, int E, int cuda_ratio) : 
    num_nodes {V}, 
    num_edges {E},
    num_gpus  {static_cast<int>(tf::cuda_get_num_devices())} {

    std::unordered_set<std::pair<int, int>, pair_hash> set;

    num_edges = std::min(num_edges, (num_nodes)*(num_nodes-1)/2);

    for(int j=0; j<num_nodes; j++) {
      Node v;
      v.v = j;
      v.g = rand()%cuda_ratio == 0 ? rand()%num_gpus : -1;
      nodes.push_back(v);
    }

    for (int j=0; j<num_edges; j++) {

      std::pair<int, int> p;
      p.first = rand() % num_nodes;
      p.second = rand() % num_nodes;

      while(set.find(p) != set.end() || p.first >= p.second) {
        p.first = rand() % num_nodes;
        p.second = rand() % num_nodes;
        if(p.first >= p.second) {
          std::swap(p.first, p.second);
        }
      };

      set.insert(p);
    }

    for (auto& pair : set) {
      Edge e;
      e.u = pair.first;
      e.v = pair.second;
      edges.push_back(e);
    }
    set.clear();
  }

  void dump(std::ostream& os) {
    os << num_nodes << ' ' << num_edges << '\n';
    for(const auto& v : nodes) {
      os << v.g << '\n';
    }
    for(const auto& e : edges) {
      os << e.u << ' ' << e.v << '\n';
    }
  }

  size_t size() const {
    return nodes.size() + edges.size();
  }
};

// saxpy kernel
template <typename T>
__global__ void add(T* x, T* y, T* z, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = x[i] + y[i];
  }
}

std::chrono::microseconds measure_time_taskflow(const Graph&, unsigned, unsigned);
std::chrono::microseconds measure_time_tbb(const Graph&, unsigned, unsigned);
std::chrono::microseconds measure_time_omp(const Graph&, unsigned, unsigned);


