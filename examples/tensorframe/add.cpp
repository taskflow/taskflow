#include <taskflow/tensorframe.hpp>

int main(){

  tf::Tensor<float> tensor({2, 3, 3, 4}, 10);

  tensor.dump(std::cout);
  
  std::cout << tensor.flat_chunk_index(1, 2, 2, 3) << '\n';
  std::cout << tensor.flat_index(1, 2, 2, 3) << '\n';
  std::cout << tensor.chunk_size() << '\n';


  return 0;
}

