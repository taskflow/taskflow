#include <taskflow/tensorframe.hpp>

int main(){
 
  using type = decltype(std::declval<int>() + std::declval<int>());
  
  tf::Tensor<float> tensor1({2, 3, 3, 4}, 10);
  tf::Tensor<float> tensor2({2, 3, 3, 4}, 10);
  tf::Tensor<float> tensor3({2, 3, 3, 4}, 10);

  tensor1.dump(std::cout);
  
  std::cout << tensor1.flat_chunk_index(1, 2, 2, 3) << '\n';
  std::cout << tensor1.flat_index(1, 2, 2, 3) << '\n';
  std::cout << tensor1.chunk_size() << '\n';

  tf::TensorFrame<float> frame;

  auto expr1 = frame.input(tensor1);
  auto expr2 = frame.input(tensor2);
  auto expr3 = frame.add(expr1, expr2);
  auto expr4 = frame.output(tensor3);

  // todo
  // tf::Executor executor;
  // frame.optimize(OptimizationLevel=CPU);
  // executor.run(frame).wait();
  // 
  // // now tensor2 has the value of tensor


  return 0;
}



