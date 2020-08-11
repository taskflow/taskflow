// reference: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c

#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default parallel for
// ----------------------------------------------------------------------------

// Function: parallel_for
template <typename I, typename C>
Task FlowBuilder::parallel_for(I beg, I end, C&& c) {
  return parallel_for_guided(beg, end, std::forward<C>(c));
}

template <typename I, typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
Task FlowBuilder::parallel_for(I beg, I end, I inc, C&& c){
  return parallel_for_guided(beg, end, inc, std::forward<C>(c));
}

// ----------------------------------------------------------------------------
// parallel for using the static partition algorithm
// ----------------------------------------------------------------------------

// Function: parallel_for_static
// static scheduling with even partition
template <typename I, typename C>
Task FlowBuilder::parallel_for_static(I beg, I end, C&& c){
  
  using namespace std::string_literals;

  Task task = emplace(
  [beg, end, c=std::forward<C>(c)] (Subflow& sf) mutable {

    if(beg == end) {
      return;
    }
  
    const size_t W = sf._executor.num_workers();
    const size_t N = std::distance(beg, end);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 1) {
      std::for_each(beg, end, c);
      return;
    }
  
    // zero-based start and end points
    const size_t q0 = N / W;
    const size_t t0 = N % W;
    
    std::atomic<size_t> next(0);

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

    sf.join();
  });  

  return task;
}

// Function: parallel_for_static
// static scheduling with chunk size
template <typename I, typename C>
Task FlowBuilder::parallel_for_static(
  I beg, I end, C&& c, size_t chunk_size
){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    return parallel_for_static(beg, end, std::forward<C>(c));
  }

  Task task = emplace(
  [beg, end, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
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

    sf.join();
  });  

  return task;
}

// Function: parallel_for_static
// static scheduling with even partition
template <typename I, typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
Task FlowBuilder::parallel_for_static(I beg, I end, I inc, C&& c){

  using namespace std::string_literals;

  Task task = emplace(
  [beg, end, inc, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
    if((inc == 0 && beg != end) || 
       (beg < end && inc <=  0) || 
       (beg > end && inc >=  0)) {
      TF_THROW("invalid range [", beg, ", ", end, ") with step size ", inc);
    }
    
    const size_t W = sf._executor.num_workers();
    const size_t N = distance(beg, end, inc);
    
    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 1) {
      for(size_t x=0; x<N; x++, beg+=inc) {
        c(beg);
      }
      return;
    }
  
    // zero-based start and end points
    const size_t q0 = N / W;
    const size_t t0 = N % W;

    std::atomic<size_t> next(0);

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

    sf.join();
  });  

  return task;
}

// Function: parallel_for_static
// static scheduling with chunk size
template <typename I, typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
Task FlowBuilder::parallel_for_static(
  I beg, I end, I inc, C&& c, size_t chunk_size
){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    return parallel_for_static(beg, end, inc, std::forward<C>(c));
  }

  Task task = emplace(
  [beg, end, inc, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
    if((inc == 0 && beg != end) || 
       (beg < end && inc <=  0) || 
       (beg > end && inc >=  0)) {
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

    sf.join();

  });  

  return task;
}

// ----------------------------------------------------------------------------
// parallel for using the guided partition algorithm
// ----------------------------------------------------------------------------

// Function: parallel_for_guided
template <typename I, typename C>
Task FlowBuilder::parallel_for_guided(I beg, I end, C&& c, size_t chunk_size){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [beg, end, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
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
        size_t s = next.load(std::memory_order_relaxed);

        while(s != N) {
          
          size_t r = N - s;
          size_t q = (r + W - 1) / W;

          if(q < chunk_size) {
            q = chunk_size;
          }

          size_t e = (q <= r) ? s + q : N;

          if(next.compare_exchange_strong(s, e, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            std::advance(beg, s-z);
            for(size_t x = s; x < e; x++, beg++) {
              c(*beg);
            }
            z = e;
            s = next.load(); 
          }
        }

      }).name("pfg_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// Function: parallel_for_guided
template <typename I, typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
Task FlowBuilder::parallel_for_guided(
  I beg, I end, I inc, C&& c, size_t chunk_size
){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [beg, end, inc, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
    if((inc == 0 && beg != end) || 
       (beg < end && inc <=  0) || 
       (beg > end && inc >=  0)) {
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
          
        size_t e0;
        size_t s0 = next.load(std::memory_order_relaxed);
          
        while(s0 != N) {
          
          size_t r = N - s0;
          size_t q = (r + W - 1) / W;

          if(q < chunk_size) {
            q = chunk_size;
          }

          e0 = (q <= r) ? s0 + q : N;

          if(next.compare_exchange_strong(s0, e0, std::memory_order_release,
                                                  std::memory_order_relaxed)) {
            I s = static_cast<I>(s0) * inc + beg;
            for(size_t x=s0; x<e0; x++, s+= inc) {
              c(s);
            }
            s0 = next.load(); 
          }
        }
      }).name("pfg_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

// ----------------------------------------------------------------------------
// Function: parallel_for_dynamic
// ----------------------------------------------------------------------------

// Function: parallel_for_dynamic
template <typename I, typename C>
Task FlowBuilder::parallel_for_dynamic(I beg, I end, C&& c, size_t chunk_size){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [beg, end, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
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
        size_t s = next.load(std::memory_order_relaxed);

        while(s != N) {
          
          size_t r = N - s;
          size_t e = (chunk_size <= r) ? s + chunk_size : N;

          if(next.compare_exchange_strong(s, e, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            std::advance(beg, s-z);
            for(size_t x = s; x < e; x++, beg++) {
              c(*beg);
            }
            z = e;
            s = next.load(); 
          }
        }

      }).name("pfd_"s + std::to_string(w));
    }
    
    sf.join();
  });  

  return task;
}

template <typename I, typename C, 
  std::enable_if_t<std::is_integral<std::decay_t<I>>::value, void>*
>
Task FlowBuilder::parallel_for_dynamic(
  I beg, I end, I inc, C&& c, size_t chunk_size
){

  using namespace std::string_literals;

  if(chunk_size == 0) {
    chunk_size = 1;
  }

  Task task = emplace(
  [beg, end, inc, chunk_size, c=std::forward<C>(c)] (Subflow& sf) mutable {
  
    if((inc == 0 && beg != end) || 
       (beg < end && inc <=  0) || 
       (beg > end && inc >=  0)) {
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
          
        size_t s0 = next.load(std::memory_order_relaxed);
          
        while(s0 != N) {
          
          size_t r = N - s0;
          size_t e0 = (chunk_size <= r) ? s0 + chunk_size : N;

          if(next.compare_exchange_strong(s0, e0, std::memory_order_release,
                                                  std::memory_order_relaxed)) {
            I s = static_cast<I>(s0) * inc + beg;
            for(size_t x=s0; x<e0; x++, s+= inc) {
              c(s);
            }
            s0 = next.load(); 
          }
        }
      }).name("pfd_"s + std::to_string(w));
    }
    
    sf.join();
  });  
          

  return task;
}



}  // end of namespace tf -----------------------------------------------------



