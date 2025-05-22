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

Example OpenMP implementation for inclusive scan:

void inclusive_scan(std::vector<int>& data) {

  int n = data.size();
  int num_threads;

  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  std::vector<int> partial_sums(num_threads, 0);

  // Step 1: Up-sweep
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int chunk_size = (n + num_threads - 1) / num_threads;
    int start = tid * chunk_size;
    int end = std::min(start + chunk_size, n);

    // Compute partial sum
    for (int i = start + 1; i < end; ++i) {
      data[i] += data[i - 1];
    }
    partial_sums[tid] = data[end - 1];
  }

  // Step 2: Propagate partial sums
  for (int i = 1; i < num_threads; ++i) {
    partial_sums[i] += partial_sums[i - 1];
  }

  // Step 3: Down-sweep
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int chunk_size = (n + num_threads - 1) / num_threads;
    int start = tid * chunk_size;
    int end = std::min(start + chunk_size, n);

    // Adjust with partial sums
    if (tid > 0) {
      for (int i = start; i < end; ++i) {
        data[i] += partial_sums[tid - 1];
      }
    }
  }
}

*/

namespace detail {

template <typename T>
struct ScanData {

  ScanData(size_t N, size_t c) : buf(N), counter(c) {}

  std::vector<CachelineAligned<T>> buf;
  std::atomic<size_t> counter;
};

// down scan task
template <typename S, typename I, typename B>
auto make_dscan_task(
  std::shared_ptr<S> sdata, 
  I d_beg,
  B bop,
  size_t w, 
  size_t block_size
) {
  return [=, sdata=std::move(sdata)]() mutable {
    for(size_t i=0; i<block_size; i++) {
      *d_beg++ = bop(sdata->buf[w-1].data, *d_beg);
    }
  };
}

// middle scan task
template <typename S, typename B>
auto make_mscan_task(std::shared_ptr<S> sdata, B bop) {
  return [=, sdata=std::move(sdata)](){
    for(size_t i=1; i<sdata->buf.size(); i++) {
      sdata->buf[i].data = bop(sdata->buf[i-1].data, sdata->buf[i].data);
    }
  };
}

template <typename Iterator, typename BufferT, typename B>
void scan_loop(
  tf::Runtime& rt,
  std::atomic<size_t>& counter,
  BufferT& buf,
  B bop,
  Iterator d_beg,
  size_t W,
  size_t w,
  size_t chunk_size
){
  // whoever finishes the last performs global scan
  if(counter.fetch_add(1, std::memory_order_acq_rel) == W-1) {
    for(size_t i=1; i<buf.size(); i++) {
      buf[i].data = bop(buf[i-1].data, buf[i].data);
    }
    counter.store(0, std::memory_order_release);
  }

  // first worker no need to do any work
  if(w==0) {
    return;
  }

  // need to do public corun because multiple workers can call this
  rt.executor().corun_until([&counter](){
    return counter.load(std::memory_order_acquire) == 0;
  });

  // block addup
  for(size_t i=0; i<chunk_size; i++) {
    *d_beg++ = bop(buf[w-1].data, *d_beg);
  }
}


}  // end of namespace tf::detail ---------------------------------------------


// Function: make_inclusive_scan_task
template <typename B, typename E, typename D, typename BOP>
auto make_inclusive_scan_task(B first, E last, D d_first, BOP bop) {
   
  using namespace std::string_literals;
  
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

    if(N < W) {
      W = N;
    }
    
    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);
    
      auto task = [=, &rt, &bop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& init = buf[w].data;
        *d_beg++ = init = *s_beg++;

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = init = bop(init, *s_beg++);
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };

      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;

    }
    
    rt.corun();
  };
}

// Function: make_inclusive_scan_task
template <typename B, typename E, typename D, typename BOP, typename T>
auto make_inclusive_scan_task(B first, E last, D d_first, BOP bop, T init) {
  
  using namespace std::string_literals;

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

    if(N < W) {
      W = N;
    }

    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    // set up the initial value for the first worker
    buf[0].data = std::move(init);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto task = [=, &rt, &bop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, *s_beg++) : *s_beg++;

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = local = bop(local, *s_beg++);
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };

      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }

    rt.corun_all();
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
  
  using namespace std::string_literals;

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
    
    if(N < W) {
      W = N;
    } 

    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto task = [=, &rt, &bop, &uop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& init = buf[w].data;
        *d_beg++ = init = uop(*s_beg++);

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = init = bop(init, uop(*s_beg++));
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };
      
      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }

    rt.corun_all();
  };
}

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename T>
auto make_transform_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, UOP uop, T init
) {
  
  using namespace std::string_literals;

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

    if(N < W) {
      W = N;
    }
    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    // set up the initial value for the first worker
    buf[0].data = std::move(init);

    size_t Q = N/W;
    size_t R = N%W;

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto task = [=, &rt, &bop, &uop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, uop(*s_beg++)) : uop(*s_beg++);

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = local = bop(local, uop(*s_beg++));
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };
      
      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }
    rt.corun_all();
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

  using namespace std::string_literals;
  
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

    if(N < W) {
      W = N;
    }
    
    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    size_t Q = N/W;
    size_t R = N%W;
    
    // fetch the init value
    auto s_beg_temp = s_beg;
    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {
      chunk_size = std::min(Q + (w<R), N - curr_b);
      buf[w].data = w ? *s_beg_temp : std::move(init);
      std::advance(s_beg_temp, chunk_size - !w);
      curr_b += chunk_size;
    }
    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto task = [=, &rt, &bop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;

        for(size_t i=1; i<chunk_size; i++) {
          auto v = local;
          local = bop(local, *s_beg++);
          *d_beg++ = std::move(v);
        }
        *d_beg++ = local;

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };
      
      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }
    rt.corun_all();
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

  using namespace std::string_literals;
  
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

    if(N < W) {
      W = N;
    }

    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);
    
    size_t Q = N/W;
    size_t R = N%W;
    // fetch the init value
    auto s_beg_temp = s_beg;
    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {
      chunk_size = std::min(Q + (w<R), N - curr_b);
      buf[w].data = w ? uop(*s_beg_temp) : std::move(init);
      std::advance(s_beg_temp, chunk_size - !w);
      curr_b += chunk_size;
    }

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto task = [=, &rt, &bop, &uop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;

        for(size_t i=1; i<chunk_size; i++) {
          auto v = local;
          local = bop(local, uop(*s_beg++));
          *d_beg++ = std::move(v);
        }
        *d_beg++ = local;

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      };
      
      (w == W-1) ? task() : rt.silent_async(task);

      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }
    rt.corun();
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


