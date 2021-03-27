#pragma once

#include "../executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default reduction
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O>
Task FlowBuilder::reduce(
  B&& beg, 
  E&& end, 
  T& init, 
  O&& bop
) {
  return reduce_guided(
    std::forward<B>(beg),
    std::forward<E>(end),
    init,
    std::forward<O>(bop),
    1
  );
}

// ----------------------------------------------------------------------------
// guided partition
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O, typename H>
Task FlowBuilder::reduce_guided(
  B&& beg, E&& end, T& init, O&& bop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   o=std::forward<O>(bop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = (c == 0) ? 1 : c;
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
// reduce_dynamic
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O, typename H>
Task FlowBuilder::reduce_dynamic(
  B&& beg, E&& end, T& init, O&& bop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   o=std::forward<O>(bop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = (c == 0) ? 1 : c;
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

      //sf.emplace([&mutex, &next, &r, beg, N, &o, C] () mutable {
      sf.silent_async([&mutex, &next, &r, beg, N, &o, C] () mutable {
        
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

        std::lock_guard<std::mutex> lock(mutex);
        r = o(r, sum);
      //}).name("prd_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// reduce_static
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O, typename H>
Task FlowBuilder::reduce_static(
  B&& beg, E&& end, T& init, O&& bop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   o=std::forward<O>(bop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = c;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= C) {
      for(; beg!=end; r = o(r, *beg++));
      return;
    }
    
    std::mutex mutex;
    std::atomic<size_t> next(0);
    
    // even partition
    if(C == 0) {

      const size_t q0 = N / W;
      const size_t t0 = N % W;
      
      for(size_t i=0; i<W; ++i) {

        size_t items = i < t0 ? q0 + 1 : q0;

        if(items == 0) {
          break;
        }
        
        //sf.emplace([&mutex, &next, &r, beg, items, &o] () mutable {
        sf.silent_async([&mutex, &next, &r, beg, items, &o] () mutable {

          size_t s0 = next.fetch_add(items, std::memory_order_relaxed);
          std::advance(beg, s0);

          if(items == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            r = o(r, *beg);
            return;
          }
          
          auto beg1 = beg++;
          auto beg2 = beg++;
          
          T sum = o(*beg1, *beg2);

          for(size_t i=2; i<items; i++, beg++) {
            sum = o(sum, *beg); 
          }
          
          std::lock_guard<std::mutex> lock(mutex);
          r = o(r, sum);

        //}).name("prs_"s + std::to_string(i));
        });
      }
    }
    // chunk-by-chunk partition
    else {
      for(size_t w=0; w<W; ++w) {
        
        // initial
        if(w*C >= N) {
          break;
        }
        
        //sf.emplace([&mutex, &next, &r, beg, end, C, N, W, &o] () mutable {
        sf.silent_async([&mutex, &next, &r, beg, end, C, N, W, &o] () mutable {

          size_t trip = W*C;
          size_t s0 = next.fetch_add(C, std::memory_order_relaxed);

          std::advance(beg, s0);

          T sum;

          if(C == 1) {
            if(s0 + trip >= N) { // last trip
              std::lock_guard<std::mutex> lock(mutex);
              r = o(r, *beg);
              return;
            }
            else {  // one more trip
              auto beg1 = beg;
              auto beg2 = std::next(beg, trip);
              sum = o(*beg1, *beg2);
              s0 += trip*2;
              if(s0 >= N) {
                goto end_reduce;
              }
              beg = std::next(beg2, trip);
            }
          }
          else {
            if(N - s0 == 1) {
              std::lock_guard<std::mutex> lock(mutex);
              r = o(r, *beg);
              return;
            }
            auto beg1 = beg++;
            auto beg2 = beg++;
            sum = o(*beg1, *beg2);
            I e = beg;
            size_t i;
            for(i=2; i<C && e != end; i++, e++) {
              sum = o(sum, *e); 
            }
            s0 += trip;
            if(i != C || s0 >= N) {
              goto end_reduce;
            }
            std::advance(beg, trip-2);
          }

          while(1) {

            size_t i;

            I e = beg;

            for(i=0; i<C && e != end; ++i, ++e) {
              sum = o(sum, *e); 
            }

            s0 += trip;

            if(i != C || s0 >= N) {
              break;
            }

            std::advance(beg, trip);
          }
          
          end_reduce:

          std::lock_guard<std::mutex> lock(mutex);
          r = o(r, sum);

        //}).name("prs_"s + std::to_string(w));
        });
      }
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
  B&& beg, 
  E&& end, 
  T& init, 
  BOP&& bop,
  UOP&& uop
) {
  return transform_reduce_guided(
    std::forward<B>(beg),
    std::forward<E>(end),
    init,
    std::forward<BOP>(bop),
    std::forward<UOP>(uop),
    1
  );
}

// ----------------------------------------------------------------------------
// guided partition
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename BOP, typename UOP, typename H>
Task FlowBuilder::transform_reduce_guided(
  B&& beg, E&& end, T& init, BOP&& bop, UOP&& uop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   bop=std::forward<BOP>(bop),
   uop=std::forward<UOP>(uop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = (c == 0) ? 1 : c;
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

// ----------------------------------------------------------------------------
// transform_reduce_dynamic
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename BOP, typename UOP, typename H>
Task FlowBuilder::transform_reduce_dynamic(
  B&& beg, E&& end, T& init, BOP&& bop, UOP&& uop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   bop=std::forward<BOP>(bop),
   uop=std::forward<UOP>(uop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = (c == 0) ? 1 : c;
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

      //sf.emplace([&mutex, &next, &r, beg, N, &bop, &uop, C] () mutable {
      sf.silent_async([&mutex, &next, &r, beg, N, &bop, &uop, C] () mutable {
        
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

        std::lock_guard<std::mutex> lock(mutex);
        r = bop(r, sum);

      //}).name("prd_"s + std::to_string(w));
      });
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// transform_reduce_static
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename BOP, typename UOP, typename H>
Task FlowBuilder::transform_reduce_static(
  B&& beg, E&& end, T& init, BOP&& bop, UOP&& uop, H&& chunk_size
) {
  
  using I = stateful_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   &r=init,
   bop=std::forward<BOP>(bop),
   uop=std::forward<UOP>(uop),
   c=std::forward<H>(chunk_size)
   ] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }

    size_t C = c;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= C) {
      for(; beg!=end; r = bop(r, uop(*beg++)));
      return;
    }
    
    std::mutex mutex;
    std::atomic<size_t> next(0);
    
    // even partition
    if(C == 0) {

      const size_t q0 = N / W;
      const size_t t0 = N % W;
      
      for(size_t i=0; i<W; ++i) {

        size_t items = i < t0 ? q0 + 1 : q0;

        if(items == 0) {
          break;
        }
        
        //sf.emplace([&mutex, &next, &r, beg, items, &bop, &uop] () mutable {
        sf.silent_async([&mutex, &next, &r, beg, items, &bop, &uop] () mutable {

          size_t s0 = next.fetch_add(items, std::memory_order_relaxed);
          std::advance(beg, s0);

          if(items == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            r = bop(r, uop(*beg));
            return;
          }
          
          auto beg1 = beg++;
          auto beg2 = beg++;
          
          T sum = bop(uop(*beg1), uop(*beg2));

          for(size_t i=2; i<items; i++, beg++) {
            sum = bop(sum, uop(*beg)); 
          }
          
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(r, sum);

        //}).name("prs_"s + std::to_string(i));
        });
      }
    }
    // chunk-by-chunk partition
    else {
      for(size_t w=0; w<W; ++w) {
        
        // initial
        if(w*C >= N) {
          break;
        }
        
        //sf.emplace([&mutex, &next, &r, beg, end, C, N, W, &bop, &uop] () mutable {
        sf.silent_async([&mutex, &next, &r, beg, end, C, N, W, &bop, &uop] () mutable {

          size_t trip = W*C;
          size_t s0 = next.fetch_add(C, std::memory_order_relaxed);

          std::advance(beg, s0);

          T sum;

          if(C == 1) {
            if(s0 + trip >= N) { // last trip
              std::lock_guard<std::mutex> lock(mutex);
              r = bop(r, uop(*beg));
              return;
            }
            else {  // one more trip
              auto beg1 = beg;
              auto beg2 = std::next(beg, trip);
              sum = bop(uop(*beg1), uop(*beg2));
              s0 += trip*2;
              if(s0 >= N) {
                goto end_transform_reduce;
              }
              beg = std::next(beg2, trip);
            }
          }
          else {
            if(N - s0 == 1) {
              std::lock_guard<std::mutex> lock(mutex);
              r = bop(r, uop(*beg));
              return;
            }
            auto beg1 = beg++;
            auto beg2 = beg++;
            sum = bop(uop(*beg1), uop(*beg2));
            I e = beg;
            size_t i;
            for(i=2; i<C && e != end; i++, e++) {
              sum = bop(sum, uop(*e)); 
            }
            s0 += trip;
            if(i != C || s0 >= N) {
              goto end_transform_reduce;
            }
            std::advance(beg, trip-2);
          }

          while(1) {

            size_t i;

            I e = beg;

            for(i=0; i<C && e != end; ++i, ++e) {
              sum = bop(sum, uop(*e)); 
            }

            s0 += trip;

            if(i != C || s0 >= N) {
              break;
            }

            std::advance(beg, trip);
          }
          
          end_transform_reduce:

          std::lock_guard<std::mutex> lock(mutex);
          r = bop(r, sum);

        //}).name("prs_"s + std::to_string(w));
        });
      }
    }
    
    sf.join();
  });  

  return task;
}

}  // end of namespace tf -----------------------------------------------------




