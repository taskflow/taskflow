#pragma once

#include "tensor.hpp"

namespace tf {

template <typename T>
class TensorNode {
  
  template <typename U>
  friend class TensorExpr;

  template <typename U>
  friend class TensorFrame;

  //using tensor_t = std::variant<
  //  std::monostate,    // not yet assigned - placeholder
  //  std::shared_ptr<Tensor<int>>, 
  //  std::shared_ptr<Tensor<float>>
  //>;

  struct Input {
    std::shared_ptr<Tensor<T>> tensor;
    Input(Tensor<T>&);
  };

  struct Output {
    std::shared_ptr<Tensor<T>> tensor;
    Output(Tensor<T>&);
  };

  struct Add {
    std::shared_ptr<Tensor<T>> tensor;
    TensorNode* lhs {nullptr};
    TensorNode* rhs {nullptr};
    Add(TensorNode*, TensorNode*);
  };

  using handle_t = std::variant<
    Input, 
    Output, 
    Add
  >;

  public:
    
    template <typename... Args>
    TensorNode(Args&&... args);

  private:

    std::string _name;

    handle_t _handle;
    
    std::vector<TensorNode*> _successors;
    std::vector<TensorNode*> _dependents;

    void _precede(TensorNode*);
};

// ----------------------------------------------------------------------------
// TensorNode::Input
// ----------------------------------------------------------------------------
template <typename T>
TensorNode<T>::Input::Input(Tensor<T>& in) :
  tensor { std::shared_ptr<Tensor<T>>(&in, [](Tensor<T>*){}) } {
  //std::cout << "input " << in.index() << '\n';
}

// ----------------------------------------------------------------------------
// TensorNode::Output
// ----------------------------------------------------------------------------
template <typename T>
TensorNode<T>::Output::Output(Tensor<T>& out) :
  tensor { std::shared_ptr<Tensor<T>>(&out, [](Tensor<T>*){}) }  {
  //std::cout << "output " << out.index() << '\n';
}

// ----------------------------------------------------------------------------
// TensorNode::Add
// ----------------------------------------------------------------------------
template <typename T>
TensorNode<T>::Add::Add(TensorNode* l, TensorNode* r) :
  lhs {l}, rhs {r} {
  std::cout << "add: " << l << ' ' << r << '\n';
}

// ----------------------------------------------------------------------------
// TensorNode member definition
// ----------------------------------------------------------------------------

// Constructor
template <typename T>
template <typename... Args>
TensorNode<T>::TensorNode(Args&&... args) : _handle{std::forward<Args>(args)...} {
} 

// Procedure: _precede
template <typename T>
void TensorNode<T>::_precede(TensorNode* v) {
  _successors.push_back(v);
  v->_dependents.push_back(this);
}

}  // end of namespace tf -----------------------------------------------------









