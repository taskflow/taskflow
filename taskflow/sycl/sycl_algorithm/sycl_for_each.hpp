#pragma once

#include "../sycl_flow.hpp"

namespace tf {

// command group function object of for_each
template <typename I, typename C>
auto syclFlow::_for_each_cgh(I first, I last, C&& op) {
  
  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);
  
  return [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {
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
  
// command group function object of for_each_index
template <typename I, typename C>  
auto syclFlow::_for_each_index_cgh(I first, I last, I step, C&& op) {
  
  if(is_range_invalid(first, last, step)) {
    TF_THROW("invalid range [", first, ", ", last, ") with step size ", step);
  }

  // TODO: special case when N is 0?
  size_t N = distance(first, last, step);
  size_t B = _default_group_size(N);

  return [=, op=std::forward<C>(op)] (sycl::handler& handler) mutable {
    size_t _N = (N % B == 0) ? N : (N + B - N % B);
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(_N), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
        size_t i = item.get_global_id(0);
        if(i < N) {
          op(static_cast<I>(i)*step + first);
        }
      }
    );
  };
}

// ----------------------------------------------------------------------------
// for_each and for_each_index algorithms
// ----------------------------------------------------------------------------

// Function: for_each
template <typename I, typename C>
syclTask syclFlow::for_each(I first, I last, C&& op) {
  auto node = _graph.emplace_back(
    _graph, _for_each_cgh(first, last, std::forward<C>(op))
  );
  return syclTask(node);
}

// Function: for_each_index
template <typename I, typename C>
syclTask syclFlow::for_each_index(I beg, I end, I inc, C&& op) {
  auto node = _graph.emplace_back(
    _graph, _for_each_index_cgh(beg, end, inc, std::forward<C>(op))
  );
  return syclTask(node);
}

// ----------------------------------------------------------------------------
// rebind
// ----------------------------------------------------------------------------

// Function: rebind_for_each
template <typename I, typename C>
void syclFlow::rebind_for_each(syclTask task, I first, I last, C&& op) {
  task._node->_func = _for_each_cgh(first, last, std::forward<C>(op));
}

// Function: rebind_for_each_index
template <typename I, typename C>
void syclFlow::rebind_for_each_index(syclTask task, I beg, I end, I inc, C&& op) {
  task._node->_func = _for_each_index_cgh(beg, end, inc, std::forward<C>(op));
}
      

}  // end of namespace tf -----------------------------------------------------
