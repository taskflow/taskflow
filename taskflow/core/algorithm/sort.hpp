#pragma once

#include "../executor.hpp"

namespace tf {

// cutoff threshold
template <typename I>
constexpr size_t qsort3w_cutoff() {

  using value_type = std::decay_t<decltype(*std::declval<I>())>;

  size_t object_size = sizeof(value_type);

  if constexpr(std::is_same_v<value_type, std::string>) {
    return 128;
  }
  else {
    if(object_size < 16) return 4096;
    else if(object_size < 32) return 2048;
    else if(object_size < 64) return 1024;
    else if(object_size < 128) return 768;
    else if(object_size < 256) return 512;
    else if(object_size < 512) return 256;
    else return 128;
  }
}

// 3-way quick sort
template <typename I, typename C>
void qsort3w(tf::Subflow& sf, I first, I last, C& compare) {
  
  using namespace std::string_literals;
    
  constexpr auto cutoff = qsort3w_cutoff<I>();

  if(static_cast<size_t>(last - first) < cutoff) {
    std::sort(first, last+1, compare);
    return;
  }

  auto l = first;
  auto r = last;
  auto f = std::next(first, 1);
  bool is_swapped_l = false;
  bool is_swapped_r = false;

  while(f <= r) {
    if(compare(*f, *l)) {
      is_swapped_l = true;
      std::iter_swap(l, f);
      l++;
      f++;
    }
    else if(compare(*l, *f)) {
      is_swapped_r = true;
      std::iter_swap(r, f);
      r--;
    }
    else {
      f++;
    }
  }

  if(l - first > 1 && is_swapped_l) {
    sf.emplace([&](tf::Subflow& sfl){
      qsort3w(sfl, first, l-1, compare);
    });
  }

  if(last - r > 1 && is_swapped_r) {
    sf.emplace([&](tf::Subflow& sfr){
      qsort3w(sfr, r+1, last, compare);
    });
  }

  sf.join();
}

// Function: sort
template <typename B, typename E, typename C>
Task FlowBuilder::sort(B&& beg, E&& end, C&& cmp) {
  
  using I = stateful_iterator_t<B, E>;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   c=std::forward<C>(cmp)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= qsort3w_cutoff<I>()) {
      std::sort(beg, end, c);
      return;
    }

    qsort3w(sf, beg, end-1, c);
  });  

  return task;
}

}  // namespace tf ------------------------------------------------------------

