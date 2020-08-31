#pragma once

#include "tensor.hpp"

namespace tf {

class TensorNode {

  friend class TensorExpr;
  friend class TensorFrame;

  using tensor_t = std::variant<
    Tensor<int>*, Tensor<float>*, Tensor<double>*
  >;

  struct Input {
    tensor_t tensor;
  };

  struct Output {
    tensor_t tensor;
  };

  struct Add {
    TensorNode* lhs;
    TensorNode* rhs;
    tensor_t tensor;
  };

  using handle_t = std::variant<
    std::monostate,  // placeholder
    Input, 
    Output, 
    Add
  >;

  public:

  private:

    std::string _name;

    handle_t _handle;


};


}  // end of namespace tf -----------------------------------------------------









