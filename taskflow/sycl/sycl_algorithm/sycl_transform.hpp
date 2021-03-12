#pragma once

#include "../sycl_flow.hpp"

namespace tf {

// Function: transform
template <typename I, typename C, typename... S>
syclTask syclFlow::transform(I first, I last, C&& op, S... srcs) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  auto node = _graph.emplace_back(
  [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {

    size_t _N = (N % B == 0) ? N : (N + B - N % B);
      
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(_N), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
        size_t i = item.get_global_id(0);
        if(i < N) {
          *(first + i) = op(*(srcs + i)...); 
        }
      }
    );

  });
  
  return syclTask(node);
}
      

}  // end of namespace tf -----------------------------------------------------
