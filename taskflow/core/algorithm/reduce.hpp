#pragma once

#include "../executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default reduction
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O>
Task FlowBuilder::reduce(
  B&& beg, E&& end, T& init, O&& bop
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   o=std::forward<O>(bop)
   //c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    //size_t C = (c == 0) ? 1 : c;
    size_t C = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= C) {
      for(; beg!=end; r = o(r, *beg++));
      return;
    }
    
    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      if(w*2 >= N) {
        break;
      }

      //sf.emplace([&mutex, &next, &r, beg, N, W, &o, C] () mutable {
      sf.silent_async([&mutex, &next, &r, beg, N, W, &o, C] () mutable {
        
        size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

        if(s0 >= N) {
          return;
        }
          
        std::advance(beg, s0);

        if(N - s0 == 1) {
          std::lock_guard<std::mutex> lock(mutex);
          r = o(r, *beg);
          return;
        }

        auto beg1 = beg++;
        auto beg2 = beg++;
        
        T sum = o(*beg1, *beg2);
              
        size_t z = s0 + 2;
        size_t p1 = 2 * W * (C + 1);
        double p2 = 0.5 / static_cast<double>(W);
        s0 = next.load(std::memory_order_relaxed);

        while(s0 < N) {
          
          size_t r = N - s0;
          
          // fine-grained
          if(r < p1) {
            while(1) {
              s0 = next.fetch_add(C, std::memory_order_relaxed);
              if(s0 >= N) {
                break;
              }
              size_t e0 = (C <= (N - s0)) ? s0 + C : N;
              std::advance(beg, s0-z);
              for(size_t x=s0; x<e0; x++, beg++) {
                sum = o(sum, *beg); 
              }
              z = e0;
            }
            break;
          }
          // coarse-grained
          else {
            size_t q = static_cast<size_t>(p2 * r);
            if(q < C) {
              q = C;
            }
            size_t e0 = (q <= r) ? s0 + q : N;
            if(next.compare_exchange_strong(s0, e0, std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
              std::advance(beg, s0-z);
              for(size_t x = s0; x<e0; x++, beg++) {
                sum = o(sum, *beg); 
              }
              z = e0;
              s0 = next.load(std::memory_order_relaxed);
            }
          }
        }

        std::lock_guard<std::mutex> lock(mutex);
        r = o(r, sum);
      //}).name("prg_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// default transform and reduction
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename BOP, typename UOP>
Task FlowBuilder::transform_reduce(
  B&& beg, E&& end, T& init, BOP&& bop, UOP&& uop
) {

  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   bop=std::forward<BOP>(bop),
   uop=std::forward<UOP>(uop)
   //c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    //size_t C = (c == 0) ? 1 : c;
    size_t C = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= C) {
      for(; beg!=end; r = bop(r, uop(*beg++)));
      return;
    }
    
    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      if(w*2 >= N) {
        break;
      }

      //sf.emplace([&mutex, &next, &r, beg, N, W, &bop, &uop, C] () mutable {
      sf.silent_async([&mutex, &next, &r, beg, N, W, &bop, &uop, C] () mutable {
        
        size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

        if(s0 >= N) {
          return;
        }
          
        std::advance(beg, s0);

        if(N - s0 == 1) {
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(r, uop(*beg));
          return;
        }

        auto beg1 = beg++;
        auto beg2 = beg++;
        
        T sum = bop(uop(*beg1), uop(*beg2));
              
        size_t z = s0 + 2;
        size_t p1 = 2 * W * (C + 1);
        double p2 = 0.5 / static_cast<double>(W);
        s0 = next.load(std::memory_order_relaxed);

        while(s0 < N) {
          
          size_t r = N - s0;
          
          // fine-grained
          if(r < p1) {
            while(1) {
              s0 = next.fetch_add(C, std::memory_order_relaxed);
              if(s0 >= N) {
                break;
              }
              size_t e0 = (C <= (N - s0)) ? s0 + C : N;
              std::advance(beg, s0-z);
              for(size_t x=s0; x<e0; x++, beg++) {
                sum = bop(sum, uop(*beg)); 
              }
              z = e0;
            }
            break;
          }
          // coarse-grained
          else {
            size_t q = static_cast<size_t>(p2 * r);
            if(q < C) {
              q = C;
            }
            size_t e0 = (q <= r) ? s0 + q : N;
            if(next.compare_exchange_strong(s0, e0, std::memory_order_acquire,
                                                    std::memory_order_relaxed)) {
              std::advance(beg, s0-z);
              for(size_t x = s0; x<e0; x++, beg++) {
                sum = bop(sum, uop(*beg)); 
              }
              z = e0;
              s0 = next.load(std::memory_order_relaxed);
            }
          }
        }

        std::lock_guard<std::mutex> lock(mutex);
        r = bop(r, sum);

      //}).name("prg_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

}  // end of namespace tf -----------------------------------------------------




