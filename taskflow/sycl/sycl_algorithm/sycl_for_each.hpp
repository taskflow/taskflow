#pragma once

#include "../sycl_flow.hpp"

namespace tf {

// Function: for_each
template <typename I, typename C>
syclTask syclFlow::for_each(I first, I last, C&& op) {

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
          op(*(first + i));
        }
      }
    );

  });
  
  return syclTask(node);
}

// Function: for_each_index
template <typename I, typename C>
syclTask syclFlow::for_each_index(I beg, I end, I inc, C&& op) {

  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?

  size_t N = distance(beg, end, inc);
  size_t B = _default_group_size(N);

  auto node = _graph.emplace_back(
  [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {

    size_t _N = (N % B == 0) ? N : (N + B - N % B);
      
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(_N), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
        size_t i = item.get_global_id(0);
        if(i < N) {
          op(static_cast<I>(i)*inc + beg);
        }
      }
    );

  });
  
  return syclTask(node);
}

// ----------------------------------------------------------------------------
// rebind
// ----------------------------------------------------------------------------

// Function: rebind_for_each
template <typename I, typename C>
void syclFlow::rebind_for_each(syclTask task, I first, I last, C&& op) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  task._node->_func =
  [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {

    size_t _N = (N % B == 0) ? N : (N + B - N % B);
      
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(_N), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
        size_t i = item.get_global_id(0);
        if(i < N) {
          op(*(first + i));
        }
      }
    );
  };
}

// Function: for_each_index
template <typename I, typename C>
void syclFlow::rebind_for_each_index(syclTask task, I beg, I end, I inc, C&& op) {

  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?

  size_t N = distance(beg, end, inc);
  size_t B = _default_group_size(N);

  task._node->_func = 
  [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {

    size_t _N = (N % B == 0) ? N : (N + B - N % B);
      
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(_N), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
        size_t i = item.get_global_id(0);
        if(i < N) {
          op(static_cast<I>(i)*inc + beg);
        }
      }
    );

  };
}
      

}  // end of namespace tf -----------------------------------------------------
