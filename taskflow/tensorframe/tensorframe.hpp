#pragma once

#include "tensor_expr.hpp"
#include "tensor_ops.hpp"

namespace tf {

/** @class TensorFrame

@brief a tensor frame represents multiple tensor operations in a graph format

*/
template <typename T>
class TensorFrame {

  friend class Executor;

  public:


    /**
    @brief creates an input tensor expression
    */
    TensorExpr<T> input(Tensor<T>& tensor);
    
    /**
    @brief creates an output tensor expression
    */
    TensorExpr<T> output(Tensor<T>& tensor);
    
    /**
    @brief creates a tensor expression that performs element-wise add
    */
    TensorExpr<T> add(TensorExpr<T> lexpr, TensorExpr<T> rexpr);

    /**
    @brief creates a tensor expression that performs element-wise multiplication
    */
    TensorExpr<T> multiply(TensorExpr<T> lexpr, TensorExpr<T> rexpr);

    void optimize();

  private:

    Taskflow _taskflow;

    std::vector<std::unique_ptr<TensorNode<T>>> _tensor_nodes;
};

template <typename T>
TensorExpr<T> TensorFrame<T>::input(Tensor<T>& tensor) {
  return TensorExpr<T>(_tensor_nodes.emplace_back(std::make_unique<TensorNode<T>>(
    std::in_place_type_t<typename TensorNode<T>::Input>{}, tensor
  )).get());
}

template <typename T>
TensorExpr<T> TensorFrame<T>::output(Tensor<T>& tensor) {
  return TensorExpr<T>(_tensor_nodes.emplace_back(std::make_unique<TensorNode<T>>(
    std::in_place_type_t<typename TensorNode<T>::Output>{}, tensor
  )).get());
}

template <typename T>
TensorExpr<T> TensorFrame<T>::add(TensorExpr<T> lhs, TensorExpr<T> rhs) {
  
  auto res = TensorExpr<T>(_tensor_nodes.emplace_back(std::make_unique<TensorNode<T>>(
    std::in_place_type_t<typename TensorNode<T>::Add>{}, 
    lhs._tensor_node, 
    rhs._tensor_node
  )).get());

  res.succeed(lhs, rhs);

  return res; 
}


}  // end of namespace tf -----------------------------------------------------









