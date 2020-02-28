#include <taskflow/cudaflow.hpp>
#include <taskflow/taskflow.hpp>

__global__ void k_set(int* ptr, size_t N, int value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] = value;
  }
}

__global__ void k_add(int* ptr, size_t N, int value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

int main() {

  unsigned num = 500;

  int* host = new int[num];
  int* dev;

  TF_CHECK_CUDA(cudaMalloc(&dev, num*sizeof(int)), "ff");

  tf::cudaGraph graph2;
  tf::cudaFlow graph(graph2);

  auto h2d = graph.copy(dev, host, num);
  auto d2h = graph.copy(host, dev, num);
  auto ker = graph.kernel({(num+256-1)/256, 1, 1}, {256, 1, 1}, 0, k_set, dev, num, 5);
  auto ad1 = graph.kernel({(num+256-1)/256, 1, 1}, {256, 1, 1}, 0, k_add, dev, num, 2);
  auto ad2 = graph.kernel({(num+256-1)/256, 1, 1}, {256, 1, 1}, 0, k_add, dev, num, 1);

  //printf("h2d: %p\n", h2d);
  //printf("d2h: %p\n", d2h);

  h2d.precede(ker);
  //std::cout << "hi2\n";
  ker.precede(ad1);
  //std::cout << "hi3\n";
  ad1.precede(ad2);
  //std::cout << "hi4\n";
  ad2.precede(d2h);
  //graph.run();

  //for(int i=0; i<num; ++i) {
    //std::cout << host[i] << '\n';
  //  assert(host[i] == 8);
  //}

  return 0;
}





