#pragma once

#include "tensor_graph.hpp"

namespace tf {

/** @class TensorExpr

@brief handle to a tensor expression created by a tensorframe

*/
class TensorExpr {

  friend class TensorFrame;
  
  public:


  private:

    TensorExpr(TensorNode* tensor_node);

    TensorNode* _tensor_node {nullptr};
};

// constructor
inline TensorExpr::TensorExpr(TensorNode* node) : _tensor_node {node} {
}

}  // end of namespace tf -----------------------------------------------------









