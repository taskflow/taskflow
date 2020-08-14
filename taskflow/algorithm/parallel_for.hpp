// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp


#pragma once

#include "../core/executor.hpp"

namespace tf {

// STL-styled iterator
template <typename B, typename E>
struct underlying_iterator {

  using TB = std::decay_t<unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<unwrap_ref_decay_t<E>>;
  
  static_assert(std::is_same<TB, TE>::value, "decayed iterator types must match");

  using type = TB;
};

template <typename B, typename E>
using underlying_iterator_t = typename underlying_iterator<B, E>::type;

// raw integral index
template <typename B, typename E, typename S>
struct underlying_index {

  using TB = std::decay_t<unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<unwrap_ref_decay_t<E>>;
  using TS = std::decay_t<unwrap_ref_decay_t<S>>;

  static_assert(
    std::is_integral<TB>::value, "decayed beg index must be an integral type"
  );
  
  static_assert(
    std::is_integral<TE>::value, "decayed end index must be an integral type"
  );
  
  static_assert(
    std::is_integral<TS>::value, "decayed step must be an integral type"
  );

  static_assert(
    std::is_same<TB, TE>::value && std::is_same<TE, TS>::value,
    "decayed index and step types must match"
  );

  using type = TB;
};

template <typename B, typename E, typename S>
using underlying_index_t = typename underlying_index<B, E, S>::type;

// ----------------------------------------------------------------------------
// default parallel for
// ----------------------------------------------------------------------------

// Function: parallel_for
template <typename B, typename E, typename C>
Task FlowBuilder::parallel_for(B&& beg, E&& end, C&& c) {
  return parallel_for_guided(
    std::forward<B>(beg), std::forward<E>(end), std::forward<C>(c), 1
  );
}

// Function: parallel_for
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::parallel_for(B&& beg, E&& end, S&& inc, C&& c){
  return parallel_for_guided(
    std::forward<B>(beg), 
    std::forward<E>(end), 
    std::forward<S>(inc), 
    std::forward<C>(c),
    1
  );
}

// ----------------------------------------------------------------------------
// parallel for using the guided partition algorithm
// - Polychronopoulos, C. D. and Kuck, D. J. 
//   "Guided Self-Scheduling: A Practical Scheduling Scheme 
//    for Parallel Supercomputers," 
//   IEEE Transactions on Computers, C-36(12):1425–1439 (1987).
// ----------------------------------------------------------------------------

// Function: parallel_for_guided
template <typename B, typename E, typename C>
Task FlowBuilder::parallel_for_guided(B&& beg, E&& end, C&& c, size_t chunk_size){
  
  using I = underlying_iterator_t<B, E>;
  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }
  
  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }
  
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

      sf.emplace([&next, beg, N, chunk_size, W, &c] () mutable {
        
        size_t z = 0;
        size_t p1 = 2 * W * (chunk_size + 1);
        double p2 = 0.5 / W;
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
            if(next.compare_exchange_strong(s0, e0, std::memory_order_release,
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

      }).name("pfg_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// Function: parallel_for_guided
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::parallel_for_guided(
  B&& beg, E&& end, S&& inc, C&& c, size_t chunk_size
){

  using I = underlying_index_t<B, E, S>;
  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   i=std::forward<S>(inc), 
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
    I inc = i;

    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
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

      sf.emplace([&next, beg, inc, N, chunk_size, W, &c] () mutable {
        
        size_t p1 = 2 * W * (chunk_size + 1);
        double p2 = 0.5 / W;
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
            if(next.compare_exchange_strong(s0, e0, std::memory_order_release,
                                                    std::memory_order_relaxed)) {
              auto s = static_cast<I>(s0) * inc + beg;
              for(size_t x=s0; x<e0; x++, s+= inc) {
                c(s);
              }
              s0 = next.load(std::memory_order_relaxed); 
            }
          }
        } 
      }).name("pfg_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// Factoring algorithm
// - Hummel, S. F., Schonberg, E., and Flynn, L. E., 
//   "Factoring: a practical and robust method for scheduling parallel loops," 
//   IEEE/ACM SC, 1991
// ----------------------------------------------------------------------------

// Function: parallel_for_factoring
template <typename B, typename E, typename C>
Task FlowBuilder::parallel_for_factoring(B&& beg, E&& end, C&& c){
  
  using I = underlying_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg),
   e=std::forward<E>(end), 
   c=std::forward<C>(c)] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }
  
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 1) {
      std::for_each(beg, end, c);
      return;
    }
    
    if(N < W) {
      W = N;
    }
    
    std::atomic<size_t> batch(0);
    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      sf.emplace([&batch, &next, beg, N, W, &c] () mutable {

        size_t z = 0;
        
        while(1) {

          size_t c0 = batch.fetch_add(1, std::memory_order_relaxed) + 1;
          size_t b0 = (c0 + W - 1) / W;
          size_t ck = static_cast<size_t>((N >> b0) / (double)W);

          if(ck == 0) {
            ck = 1;
          }

          size_t s0 = next.fetch_add(ck, std::memory_order_relaxed);
          if(s0 >= N) {
            return;
          }
          size_t e0 = (ck <= (N - s0)) ? s0 + ck : N;
          std::advance(beg, s0-z);
          for(size_t x=s0; x<e0; x++) {
            c(*beg++);
          }
          z = e0;
        }
      }).name("pfg_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// Function: parallel_for_factoring
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::parallel_for_factoring(B&& beg, E&& end, S&& inc, C&& c){

  using I = underlying_index_t<B, E, S>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   i=std::forward<S>(inc), 
   c=std::forward<C>(c)] (Subflow& sf) mutable {
    
    // fetch the iterator values
    I beg = b;
    I end = e;
    I inc = i;

    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
    size_t W = sf._executor.num_workers();
    size_t N = distance(beg, end, inc);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 1) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }
    
    if(N < W) {
      W = N;
    }
    
    std::atomic<size_t> batch(0);
    std::atomic<size_t> next(0);

    for(size_t w=0; w<W; w++) {

      sf.emplace([&batch, &next, beg, inc, N, W, &c] () mutable {

        while(1) {

          size_t c0 = batch.fetch_add(1, std::memory_order_relaxed) + 1;
          size_t b0 = (c0 + W - 1) / W;
          size_t ck = static_cast<size_t>((N >> b0) / (double)(W));

          if(ck == 0) {
            ck = 1;
          }

          size_t s0 = next.fetch_add(ck, std::memory_order_relaxed);
          if(s0 >= N) {
            return;
          }
          size_t e0 = (ck <= (N - s0)) ? s0 + ck : N;
          auto s = static_cast<I>(s0) * inc + beg;
          for(size_t x=s0; x<e0; x++, s+=inc) {
            c(s);
          }
        }

      }).name("pff_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// Function: parallel_for_dynamic
// ----------------------------------------------------------------------------

// Function: parallel_for_dynamic
template <typename B, typename E, typename C>
Task FlowBuilder::parallel_for_dynamic(B&& beg, E&& end, C&& c, size_t chunk_size){

  using I = underlying_iterator_t<B, E>;
  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {

    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }
  
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

      sf.emplace([&next, beg, N, chunk_size, &c] () mutable {
        
        size_t z = 0;

        while(1) {

          size_t s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
          
          if(s0 >= N) {
            break;
          }
          
          size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
          std::advance(beg, s0-z);
          for(size_t x=s0; x<e0; x++) {
            c(*beg++);
          }
          z = e0;
        }

      }).name("pfd_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

template <typename B, typename E, typename S, typename C>
Task FlowBuilder::parallel_for_dynamic(
  B&& beg, E&& end, S&& inc, C&& c, size_t chunk_size
){
  
  using I = underlying_index_t<B, E, S>;
  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end),
   i=std::forward<S>(inc),
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {

    I beg = b;
    I end = e;
    I inc = i;
  
    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
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

      sf.emplace([&next, beg, inc, N, chunk_size, &c] () mutable {

        while(1) {
          
          size_t s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);

          if(s0 >= N) {
            break;
          }
          
          size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
          I s = static_cast<I>(s0) * inc + beg;
          for(size_t x=s0; x<e0; x++, s+=inc) {
            c(s);
          }
        }
      }).name("pfd_"s + std::to_string(w));
    }
    
    sf.join();
  });  
          

  return task;
}

// ----------------------------------------------------------------------------
// parallel for using the static partition algorithm
// ----------------------------------------------------------------------------

// Function: parallel_for_static
// static scheduling with chunk size
template <typename B, typename E, typename C>
Task FlowBuilder::parallel_for_static(
  B&& beg, E&& end, C&& c, size_t chunk_size
){
  
  using I = underlying_iterator_t<B, E>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {
    
    // fetch the iterator
    I beg = b;
    I end = e;
  
    if(beg == end) {
      return;
    }
    
    const size_t W = sf._executor.num_workers();
    const size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      std::for_each(beg, end, c);
      return;
    }

    std::atomic<size_t> next(0);
    
    // even partition
    if(chunk_size == 0 ){
    
      // zero-based start and end points
      const size_t q0 = N / W;
      const size_t t0 = N % W;

      for(size_t i=0; i<W; ++i) {

        size_t items = i < t0 ? q0 + 1 : q0;

        if(items == 0) {
          break;
        }
        
        sf.emplace([&next, beg, items, &c] () mutable {
          size_t s0 = next.fetch_add(items, std::memory_order_relaxed);
          std::advance(beg, s0);
          for(size_t i=0; i<items; i++) {
            c(*beg++);
          }
        }).name("pfs_"s + std::to_string(i));
      }

    }
    // chunk-by-chunk partition
    else {
      for(size_t i=0; i<W; ++i) {
        
        // initial
        if(i*chunk_size >= N) {
          break;
        }

        sf.emplace([&next, beg, end, chunk_size, N, W, &c] () mutable {

          size_t trip = W*chunk_size;
          size_t s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);

          std::advance(beg, s0);

          while(1) {

            size_t items;

            I e = beg;

            for(items=0; items<chunk_size && e != end; ++items, ++e) {
              c(*e); 
            }

            s0 += trip;

            if(items != chunk_size || s0 >= N) {
              break;
            }

            std::advance(beg, trip);
          }

        }).name("pfs_"s + std::to_string(i));
      }
    }

    sf.join();
  });  

  return task;
}

// Function: parallel_for_static
// static scheduling with chunk size
template <typename B, typename E, typename S, typename C>
Task FlowBuilder::parallel_for_static(
  B&& beg, E&& end, S&& inc, C&& c, size_t chunk_size
){

  using I = underlying_index_t<B, E, S>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=std::forward<B>(beg), 
   e=std::forward<E>(end), 
   i=std::forward<S>(inc), 
   c=std::forward<C>(c),
   chunk_size] (Subflow& sf) mutable {
    
    // fetch the indices
    I beg = b;
    I end = e;
    I inc = i;
    
    if(is_range_invalid(beg, end, inc)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
    // configured worker count
    const size_t W = sf._executor.num_workers();
    const size_t N = distance(beg, end, inc);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }

    std::atomic<size_t> next(0);
    
    if(chunk_size == 0) {
      // zero-based start and end points
      const size_t q0 = N / W;
      const size_t t0 = N % W;
      for(size_t i=0; i<W; ++i) {

        size_t items = i < t0 ? q0 + 1 : q0;

        if(items == 0) {
          break;
        }
        
        sf.emplace([&next, beg, &inc, items, &c] () mutable {

          size_t s0 = next.fetch_add(items, std::memory_order_relaxed);
        
          I s = static_cast<I>(s0) * inc + beg;

          for(size_t x=0; x<items; x++, s+=inc) {
            c(s);
          }
        }).name("pfs_"s + std::to_string(i));
      }

    }
    else {
      for(size_t i=0; i<W; ++i) {
        
        // initial
        if(i*chunk_size >= N) {
          break;
        }

        sf.emplace([&next, beg, inc, chunk_size, N, W, &c] () mutable {

          size_t trip = W * chunk_size;
          size_t s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
          
          while(1) {

            size_t e0 = s0 + chunk_size;

            if(e0 > N) {
              e0 = N;
            }

            I s = static_cast<I>(s0) * inc + beg;

            for(size_t x=s0; x<e0; x++, s+=inc) {
              c(s);
            }

            if(e0 == N) {
              break;
            }

            s0 += trip;

            if(s0 >= N) {
              break;
            }
          }

        }).name("pfs_"s + std::to_string(i));
      }
    }

    sf.join();

  });  

  return task;
}



}  // end of namespace tf -----------------------------------------------------



