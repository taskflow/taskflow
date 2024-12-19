#pragma once

#include "../taskflow.hpp"

namespace tf {

/*

Block-parallel scan algorithm:

-----------------------------------------------------------------
|    block 1    |    block 2    |    block 3    |    block 4    |
-----------------------------------------------------------------

                -----------------------------
                |  B1  |  B2  |  B3  |  B4  |  // scan block sum to auxilinary array
                -----------------------------
                |                           |
                v                           v
                -----------------------------
                |  B1  |  B2  |  B3  |  B4  |  // scan block sums
                -----------------------------
                   |
                   |                           // add scanned block sum i to all 
                   |                           // values of scanned block i+1
                   v
-----------------------------------------------------------------
|    block 1    |    block 2    |    block 3    |    block 4    |
-----------------------------------------------------------------

*/

namespace detail {

template <typename T>
struct ScanData {

  ScanData(size_t N, size_t c) : buf(N), counter(c) {}

  std::vector<CachelineAligned<T>> buf;
  std::atomic<size_t> counter;
};

// Function: scan_loop
template <typename S, typename Iterator, typename B>
void scan_loop(
  S& sdata,
  B bop, 
  Iterator d_beg, 
  size_t W,
  size_t w, 
  size_t block_size
){
  // whoever finishes the last performs global scan
  if(sdata.counter.fetch_add(1, std::memory_order_acq_rel) == W-1) {
    for(size_t i=1; i<sdata.buf.size(); i++) {
      sdata.buf[i].data = bop(sdata.buf[i-1].data, sdata.buf[i].data);
    }
    sdata.counter.store(0, std::memory_order_release);
  }

  // first worker no need to do any work
  if(w==0) {
    return;
  } 
  
  // simply do a loop until the counter becomes zero; we don't do corun
  // as the block scan is typically very fast, and stealing a task can cause
  // the worker to evict cached data from the block scan
  spin_until([&](){ 
    return sdata.counter.load(std::memory_order_acquire) == 0; }
  );
  
  // block addup
  for(size_t i=0; i<block_size; i++) {
    *d_beg++ = bop(sdata.buf[w-1].data, *d_beg);
  }
}

}  // end of namespace tf::detail ---------------------------------------------


// Function: make_inclusive_scan_task
template <typename B, typename E, typename D, typename BOP>
auto make_inclusive_scan_task(B first, E last, D d_first, BOP bop) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::inclusive_scan(s_beg, s_end, d_beg, bop);
      return;
    }

    PreemptionGuard preemption_guard(rt);

    if(N < W) {
      W = N;
    }
    
    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);

    size_t Q = N/W;
    size_t R = N%W;
    
    for(size_t w=0; w<W;) {

      size_t block_size = Q + (w<R);

      // block scan
      auto task = [=, &rt] () mutable {
        // prefetch the beginning of the block
        auto result = d_beg;

        // local scan per worker
        auto& init = scan_data->buf[w].data;
        *d_beg++ = init = *s_beg++;
        for(size_t i=1; i<block_size; i++){
          *d_beg++ = init = bop(init, *s_beg++); 
        }

        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };
      
      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
      
      (++w == W) ? task() : rt.silent_async(task);
    }
  };
}

// Function: make_inclusive_scan_task
template <typename B, typename E, typename D, typename BOP, typename T>
auto make_inclusive_scan_task(B first, E last, D d_first, BOP bop, T init) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::inclusive_scan(s_beg, s_end, d_beg, bop, init);
      return;
    }
    
    PreemptionGuard preemption_guard(rt);

    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);

    if(N < W) {
      W = N;
    }
    
    // set up the initial value for the first worker
    scan_data->buf[0].data = std::move(init);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0; w<W;) {

      size_t block_size = Q + (w < R);

      // block scan
      auto task = [=, &rt] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = scan_data->buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, *s_beg++) : *s_beg++;

        for(size_t i=1; i<block_size; i++){
          *d_beg++ = local = bop(local, *s_beg++); 
        }
        
        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };

      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
      (++w == W) ? task() : rt.silent_async(task);
    }
  };
}

// ----------------------------------------------------------------------------
// Transform Inclusive Scan
// ----------------------------------------------------------------------------

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP>
auto make_transform_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, UOP uop
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::transform_inclusive_scan(s_beg, s_end, d_beg, bop, uop);
      return;
    }

    PreemptionGuard preemption_guard(rt);
    
    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);

    if(N < W) {
      W = N;
    } 
    
    size_t Q = N/W;
    size_t R = N%W;
    
    for(size_t w=0; w<W;) {

      size_t block_size = Q + (w < R);

      // block scan
      auto task = [=, &rt] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& init = scan_data->buf[w].data;
        *d_beg++ = init = uop(*s_beg++);

        for(size_t i=1; i<block_size; i++){
          *d_beg++ = init = bop(init, uop(*s_beg++)); 
        }

        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };
      
      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
      (++w == W) ? task() : rt.silent_async(task);
    }
  };
}

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename T>
auto make_transform_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, UOP uop, T init
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::transform_inclusive_scan(s_beg, s_end, d_beg, bop, uop, init);
      return;
    }

    PreemptionGuard preemption_guard(rt);

    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);

    if(N < W) {
      W = N;
    }
    
    // set up the initial value for the first worker
    scan_data->buf[0].data = std::move(init);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0; w<W;) {

      size_t block_size = Q + (w < R);

      // block scan
      auto task = [=, &rt] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = scan_data->buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, uop(*s_beg++)) : uop(*s_beg++);

        for(size_t i=1; i<block_size; i++){
          *d_beg++ = local = bop(local, uop(*s_beg++)); 
        }
        
        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };

      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
      (++w == W) ? task() : rt.silent_async(task);
    }
  };
}

// ----------------------------------------------------------------------------
// Exclusive Scan
// ----------------------------------------------------------------------------

// Function: make_exclusive_scan_task
template <typename B, typename E, typename D, typename T, typename BOP>
auto make_exclusive_scan_task(
  B first, E last, D d_first, T init, BOP bop
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::exclusive_scan(s_beg, s_end, d_beg, init, bop);
      return;
    }

    PreemptionGuard preemption_guard(rt);

    if(N < W) {
      W = N;
    }
    
    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);

    size_t Q = N/W;
    size_t R = N%W;

    // fetch the init value
    auto s_beg_temp = s_beg;
    for(size_t w=0, curr_b=0; w<W; ++w) {
      size_t block_size = Q + (w<R);  
      scan_data->buf[w].data = w ? *s_beg_temp : std::move(init);
      std::advance(s_beg_temp, block_size - !w);
      curr_b += block_size;
    }
    
    for(size_t w=0; w<W;) {

      size_t block_size = (Q + (w < R));

      // block scan
      auto task = [=, &rt] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = scan_data->buf[w].data;

        for(size_t i=1; i<block_size; i++) {
          auto v = local;
          local = bop(local, *s_beg++);
          *d_beg++ = std::move(v);
        }
        *d_beg++ = local;
        
        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };
      
      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
        
      (++w == W) ? task() : rt.silent_async(task);
    }

  };
}

// ----------------------------------------------------------------------------
// Transform Exclusive Scan
// ----------------------------------------------------------------------------

// Function: 
template <typename B, typename E, typename D, typename T, typename BOP, typename UOP>
auto make_transform_exclusive_scan_task(
  B first, E last, D d_first, T init, BOP bop, UOP uop
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  
  return [=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::transform_exclusive_scan(s_beg, s_end, d_beg, init, bop, uop);
      return;
    }

    PreemptionGuard preemption_guard(rt);

    if(N < W) {
      W = N;
    }
    
    auto scan_data = std::make_shared< detail::ScanData<value_type> >(W, 0);
    
    size_t Q = N/W;
    size_t R = N%W;

    // fetch the init value
    auto s_beg_temp = s_beg;
    for(size_t w=0, curr_b=0; w<W; ++w) {
      size_t block_size = Q + (w < R);
      scan_data->buf[w].data = w ? uop(*s_beg_temp) : std::move(init);
      std::advance(s_beg_temp, block_size - !w);
      curr_b += block_size;
    }
    
    for(size_t w=0; w<W;) {

      size_t block_size = Q + (w < R);

      // block scan
      auto task = [=, &rt] () mutable {

        auto result = d_beg;

        // local scan per worker
        auto& local = scan_data->buf[w].data;

        for(size_t i=1; i<block_size; i++) {
          auto v = local;
          local = bop(local, uop(*s_beg++));
          *d_beg++ = std::move(v);
        }
        *d_beg++ = local;
        
        // block scan
        detail::scan_loop(*scan_data, bop, result, W, w, block_size);
      };
      
      std::advance(s_beg, block_size);
      std::advance(d_beg, block_size);
      
      (++w == W) ? task() : rt.silent_async(task);
    } 
  };
}


// ----------------------------------------------------------------------------
// Inclusive Scan
// ----------------------------------------------------------------------------

// Function: inclusive_scan
template <typename B, typename E, typename D, typename BOP>
Task FlowBuilder::inclusive_scan(B first, E last, D d_first, BOP bop) {
  return emplace(make_inclusive_scan_task(first, last, d_first, bop));
}

// Function: inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename T>
Task FlowBuilder::inclusive_scan(B first, E last, D d_first, BOP bop, T init) {
  return emplace(make_inclusive_scan_task(first, last, d_first, bop, init));
}

// ----------------------------------------------------------------------------
// Transform Inclusive Scan
// ----------------------------------------------------------------------------

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP>
Task FlowBuilder::transform_inclusive_scan(
  B first, E last, D d_first, BOP bop, UOP uop
) {
  return emplace(make_transform_inclusive_scan_task(
    first, last, d_first, bop, uop
  ));
}

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename T>
Task FlowBuilder::transform_inclusive_scan(
  B first, E last, D d_first, BOP bop, UOP uop, T init
) {
  return emplace(make_transform_inclusive_scan_task(
    first, last, d_first, bop, uop, init
  ));  
}

// ----------------------------------------------------------------------------
// Exclusive Scan
// ----------------------------------------------------------------------------

// Function: exclusive_scan
template <typename B, typename E, typename D, typename T, typename BOP>
Task FlowBuilder::exclusive_scan(B first, E last, D d_first, T init, BOP bop) {
  return emplace(make_exclusive_scan_task(first, last, d_first, init, bop));
}

// ----------------------------------------------------------------------------
// Transform Exclusive Scan
// ----------------------------------------------------------------------------

// Function: transform_exclusive_scan
template <typename B, typename E, typename D, typename T, typename BOP, typename UOP>
Task FlowBuilder::transform_exclusive_scan(
  B first, E last, D d_first, T init, BOP bop, UOP uop
) {
  return emplace(make_transform_exclusive_scan_task(
    first, last, d_first, init, bop, uop
  )); 
}

}  // end of namespace tf -----------------------------------------------------

