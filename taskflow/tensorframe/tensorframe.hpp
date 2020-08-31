#pragma once

#include "tensor_expr.hpp"

namespace tf {

class TensorFrame : protected Taskflow {

  public:
    
    template <typename T>
    TensorExpr input(Tensor<T>& tensor);

    template <typename T>
    TensorExpr output(Tensor<T>& tensor);
    
    TensorExpr add(TensorExpr lexpr, TensorExpr rexpr);

  private:

    std::vector<TensorNode> _tensor_nodes;
};

//auto TensorExpr = TensorGraph.tensor_frame()

//
// TensorFrame tf1({1000, 1000, 1000}), tf2({1000, 1000, 1000}), tf3, tf4;
// ExpressionFlow ef;
//
// auto expr1 = ef.input(tf1);
// auto expr2 = ef.input(tf2);
// auto expr3 = ef.input(tf3);
// auto expr4 = ef.input(tf4);
// auto sum12 = ef.add(expr1, expr2);
// auto sum34 = ef.add(expr3, expr4);
// auto s1234 = ef.add(sum12, sum34);
// ef.evaluate(s1234, tensor_frame);

//executor.run(ef);


}  // end of namespace tf -----------------------------------------------------









