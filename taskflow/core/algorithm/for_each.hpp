// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp


#pragma once

#include "../executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default parallel for
// ----------------------------------------------------------------------------

// Function: for_each
template <typename B, typename E, typename C>
Task FlowBuilder::for_each(B&& beg, E&& end, C&& c) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   c=std::forward<C>(c)] (Subflow& sf) mutable {
    
    // fetch the stateful values
    I beg = b;
    I end = e;

    if(beg == end) {
      return;
    }
  
    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      std::for_each(beg, end, c);
      return;
    }
    
    if(N < W) {
      W = N;
    }

    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      //sf.emplace([&next, beg, N, chunk_size, W, &c] () mutable {
      sf.silent_async([&next, beg, N, chunk_size, W, &c] () mutable {
        
        size_t z = 0;
        size_t p1 = 2 * W * (chunk_size + 1);
        double p2 = 0.5 / static_cast<double>(W);
        size_t s0 = next.load(std::memory_order_relaxed);

        while(s0 < N) {
          
          size_t r = N - s0;
          
          // fine-grained
          if(r < p1) {
            while(1) {
              s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
              if(s0 >= N) {
                return;
              }
              size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
              std::advance(beg, s0-z);
              for(size_t x=s0; x<e0; x++) {
                c(*beg++);
              }
              z = e0;
            }
            break;
          }
          // coarse-grained
          else {
            size_t q = static_cast<size_t>(p2 * r);
            if(q < chunk_size) {
              q = chunk_size;
            }
            size_t e0 = (q <= r) ? s0 + q : N;
            if(next.compare_exchange_strong(s0, e0, std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
              std::advance(beg, s0-z);
              for(size_t x = s0; x< e0; x++) {
                c(*beg++);
              }
              z = e0;
              s0 = next.load(std::memory_order_relaxed);
            }
          }
        }
      //}).name("pfg_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

// Function: for_each_index
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::for_each_index(B&& beg, E&& end, S&& inc, C&& c){
  
  using I = stateful_index_t<B, E, S>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   a=std::forward<S>(inc), 
   c=std::forward<C>(c)] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
    I inc = a;

    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = distance(beg, end, inc);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }
    
    if(N < W) {
      W = N;
    }
    
    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      //sf.emplace([&next, beg, inc, N, chunk_size, W, &c] () mutable {
      sf.silent_async([&next, beg, inc, N, chunk_size, W, &c] () mutable {
        
        size_t p1 = 2 * W * (chunk_size + 1);
        double p2 = 0.5 / static_cast<double>(W);
        size_t s0 = next.load(std::memory_order_relaxed);

        while(s0 < N) {
        
          size_t r = N - s0;
          
          // find-grained
          if(r < p1) {
            while(1) { 
              s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
              if(s0 >= N) {
                return;
              }
              size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
              auto s = static_cast<I>(s0) * inc + beg;
              for(size_t x=s0; x<e0; x++, s+=inc) {
                c(s);
              }
            }
            break;
          }
          // coarse-grained
          else {
            size_t q = static_cast<size_t>(p2 * r);
            if(q < chunk_size) {
              q = chunk_size;
            }
            size_t e0 = (q <= r) ? s0 + q : N;
            if(next.compare_exchange_strong(s0, e0, std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
              auto s = static_cast<I>(s0) * inc + beg;
              for(size_t x=s0; x<e0; x++, s+= inc) {
                c(s);
              }
              s0 = next.load(std::memory_order_relaxed); 
            }
          }
        } 
      //}).name("pfg_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

}  // end of namespace tf -----------------------------------------------------



