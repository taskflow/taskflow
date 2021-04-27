#pragma once

#include "tensor_graph.hpp"

namespace tf {

/** @class TensorExpr

@brief handle to a tensor expression created by a tensorframe

*/
template <typename T>
class TensorExpr {

  template <typename U>
  friend class TensorFrame;
  
  public:

    /**
    @brief constructs an empty tensor expression
    */
    TensorExpr() = default;
    
    /**
    @brief copy constructor of the tensor expression
    */
    TensorExpr(const TensorExpr& rhs);    

    /**
    @brief move constructor of the tensor expression

    After the move, @c rhs becomes an empty tensor expression that is not associated
    with any tensor node.
    */
    TensorExpr(TensorExpr&& rhs);    

    /**
    @brief copy assignment of the tensor expression
    */
    TensorExpr& operator = (const TensorExpr& rhs);

    /**
    @brief move assignment of the tensor expression

    After the move, @c rhs becomes an empty tensor expression that is not associated
    with any tensor node.
    */
    TensorExpr& operator = (TensorExpr&& rhs);

    /**
    @brief assigns a name to the tensor expression
    */
    TensorExpr& name(const std::string& name); 

    /**
    @brief queries the name of the tensor expression
    */
    const std::string& name() const;
    
    /**
    @brief adds precedence links from @c this to other tensor expressions

    @tparam Es... parameter pack

    @param exprs one or multiple expressions to precede

    @return @c *this
    */
    template <typename... Es>
    TensorExpr& precede(Es&&... exprs);
    
    /**
    @brief adds succeeding links from other tensor expressions to @c this

    @tparam Es... parameter pack

    @param exprs one or multiple expressions to succeed

    @return @c *this
    */
    template <typename... Es>
    TensorExpr& succeed(Es&&... exprs);

  private:

    TensorExpr(TensorNode<T>* tensor_node);

    TensorNode<T>* _tensor_node {nullptr};
};

// copy constructor
template <typename T>
TensorExpr<T>::TensorExpr(const TensorExpr& rhs) :
  _tensor_node {rhs._tensor_node} {
}

// move constructor
template <typename T>
TensorExpr<T>::TensorExpr(TensorExpr&& rhs) : 
  _tensor_node {rhs._tensor_node} {
  rhs._tensor_node = nullptr;  
}

// constructor
template <typename T>
TensorExpr<T>::TensorExpr(TensorNode<T>* node) : _tensor_node {node} {
}

// copy assignment
template <typename T>
TensorExpr<T>& TensorExpr<T>::operator = (const TensorExpr& rhs) {
  _tensor_node = rhs._tensor_node;
  return *this;
}

// move assignment
template <typename T>
TensorExpr<T>& TensorExpr<T>::operator = (TensorExpr&& rhs) {
  _tensor_node = rhs._tensor_node;
  rhs._tensor_node = nullptr;  
  return *this;
}

// assigns a name
template <typename T>
TensorExpr<T>& TensorExpr<T>::name(const std::string& name) {
  _tensor_node->_name = name;
  return *this;
}

// queries the name
template <typename T>
const std::string& TensorExpr<T>::name() const {
  return _tensor_node->_name;
}

// Function: precede
template <typename T>
template <typename... Es>
TensorExpr<T>& TensorExpr<T>::precede(Es&&... exprs) {
  (_tensor_node->_precede(exprs._tensor_node), ...);
  return *this;
}

// Function: succeed
template <typename T>
template <typename... Es>
TensorExpr<T>& TensorExpr<T>::succeed(Es&&... tasks) {
  (tasks._tensor_node->_precede(_tensor_node), ...);
  return *this;
}

}  // end of namespace tf -----------------------------------------------------









