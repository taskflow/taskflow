#pragma once

#include "launch.hpp"

namespace tf {

namespace detail {

// Function: scan_loop
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
template <typename B, typename E, typename D, typename BOP, typename P = DefaultPartitioner,
  std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>* = nullptr
>
auto make_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, P part = P()
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
      launch_loop(part, [&](){
        std::inclusive_scan(s_beg, s_end, d_beg, bop);
      });
      return;
    }

    if(N < W) {
      W = N;
    }
    
    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    size_t Q = N/W;
    size_t R = N%W;
    
    //auto orig_d_beg = d_beg;
    //ExecutionPolicy<StaticPartitioner> policy;

    for(size_t w=0, curr_b=0, chunk_size; w<W && curr_b < N; ++w) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      launch_loop(W, w, rt, part, [=, &rt, &bop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& init = buf[w].data;
        *d_beg++ = init = *s_beg++;

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = init = bop(init, *s_beg++); 
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
        
        //size_t offset = R ? Q + 1 : Q;
        //size_t rest   = N - offset;
        //size_t rest_Q = rest / W;
        //size_t rest_R = rest % W;
        //
        //chunk_size = policy.chunk_size() == 0 ? 
        //             rest_Q + (w < rest_R) : policy.chunk_size();
        //
        //size_t curr_b = policy.chunk_size() == 0 ? 
        //                offset + (w<rest_R ? w*(rest_Q + 1) : rest_R + w*rest_Q) :
        //                offset + w*policy.chunk_size();

        //policy(N, W, curr_b, chunk_size,
        //  [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
        //    std::advance(orig_d_beg, curr_b - prev_e);
        //    for(size_t x = curr_b; x<curr_e; x++) {
        //      size_t j = x < (Q+1)*R ? x/(Q+1) : (x-(Q+1)*R)/Q + R;
        //      *orig_d_beg++ = bop(buf[j-1].data, *orig_d_beg);
        //    }
        //    prev_e = curr_e;
        //  }
        //);
      });
      
      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }

    rt.corun_all();
  };
}

// Function: make_inclusive_scan_task
template <typename B, typename E, typename D, typename BOP, typename T, typename P = DefaultPartitioner,
  std::enable_if_t<!is_partitioner_v<std::decay_t<T>>, void>* = nullptr
>
auto make_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, T init, P part = P()
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
      launch_loop(part, [&](){
        std::inclusive_scan(s_beg, s_end, d_beg, bop, init);
      });
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
      launch_loop(W, w, rt, part, [=, &rt, &bop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, *s_beg++) : *s_beg++;

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = local = bop(local, *s_beg++); 
        }
        
        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      });

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
template <typename B, typename E, typename D, typename BOP, typename UOP, typename P = DefaultPartitioner,
  std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>* = nullptr
>
auto make_transform_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, UOP uop, P part = P()
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
      launch_loop(part, [&](){
        std::transform_inclusive_scan(s_beg, s_end, d_beg, bop, uop); 
      });
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
      launch_loop(W, w, rt, part, [=, &rt, &bop, &uop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& init = buf[w].data;
        *d_beg++ = init = uop(*s_beg++);

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = init = bop(init, uop(*s_beg++)); 
        }

        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      });
      
      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }

    rt.corun_all();
  };
}

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename T, typename P = DefaultPartitioner,
  std::enable_if_t<!is_partitioner_v<std::decay_t<T>>, void>* = nullptr
>
auto make_transform_inclusive_scan_task(
  B first, E last, D d_first, BOP bop, UOP uop, T init, P part = P()
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
      launch_loop(part, [&](){
        std::transform_inclusive_scan(s_beg, s_end, d_beg, bop, uop, init);
      });
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
      launch_loop(W, w, rt, part, [=, &rt, &bop, &uop, &buf, &counter] () mutable {
        auto result = d_beg;

        // local scan per worker
        auto& local = buf[w].data;
        *d_beg++ = local = (w == 0) ? bop(local, uop(*s_beg++)) : uop(*s_beg++);

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = local = bop(local, uop(*s_beg++)); 
        }
        
        // block scan
        detail::scan_loop(rt, counter, buf, bop, result, W, w, chunk_size);
      });

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
template <typename B, typename E, typename D, typename T, typename BOP, typename P = DefaultPartitioner>
auto make_exclusive_scan_task(
  B first, E last, D d_first, T init, BOP bop, P part = P()
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
      launch_loop(part, [&](){
        std::exclusive_scan(s_beg, s_end, d_beg, init, bop);
      });
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
      launch_loop(W, w, rt, part, [=, &rt, &bop, &buf, &counter] () mutable {
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
      });
      
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
template <typename B, typename E, typename D, typename T, typename BOP, typename UOP, typename P = DefaultPartitioner>
auto make_transform_exclusive_scan_task(
  B first, E last, D d_first, T init, BOP bop, UOP uop, P part = P()
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
      launch_loop(part, [&](){
        std::transform_exclusive_scan(s_beg, s_end, d_beg, init, bop, uop);
      });
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
      launch_loop(W, w, rt, part, [=, &rt, &bop, &uop, &buf, &counter] () mutable {
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
      });
      
      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
      curr_b += chunk_size;
    }

    rt.corun_all();
    
  };
}


// ----------------------------------------------------------------------------
// Inclusive Scan
// ----------------------------------------------------------------------------

// Function: inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename P,
  std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>*
>
Task FlowBuilder::inclusive_scan(B first, E last, D d_first, BOP bop, P part) {
  return emplace(make_inclusive_scan_task(first, last, d_first, bop, part));
}

// Function: inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename T, typename P,
  std::enable_if_t<!is_partitioner_v<std::decay_t<T>>, void>*
>
Task FlowBuilder::inclusive_scan(B first, E last, D d_first, BOP bop, T init, P part) {
  return emplace(make_inclusive_scan_task(first, last, d_first, bop, init, part));
}

// ----------------------------------------------------------------------------
// Transform Inclusive Scan
// ----------------------------------------------------------------------------

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename P,
  std::enable_if_t<is_partitioner_v<std::decay_t<P>>, void>*
>
Task FlowBuilder::transform_inclusive_scan(
  B first, E last, D d_first, BOP bop, UOP uop, P part
) {
  return emplace(make_transform_inclusive_scan_task(
    first, last, d_first, bop, uop, part
  ));
}

// Function: transform_inclusive_scan
template <typename B, typename E, typename D, typename BOP, typename UOP, typename T, typename P,
  std::enable_if_t<!is_partitioner_v<std::decay_t<T>>, void>*
>
Task FlowBuilder::transform_inclusive_scan(
  B first, E last, D d_first, BOP bop, UOP uop, T init, P part
) {
  return emplace(make_transform_inclusive_scan_task(
    first, last, d_first, bop, uop, init, part
  ));  
}

// ----------------------------------------------------------------------------
// Exclusive Scan
// ----------------------------------------------------------------------------

// Function: exclusive_scan
template <typename B, typename E, typename D, typename T, typename BOP, typename P>
Task FlowBuilder::exclusive_scan(B first, E last, D d_first, T init, BOP bop, P part) {
  return emplace(make_exclusive_scan_task(
    first, last, d_first, init, bop, part
  ));
}

// ----------------------------------------------------------------------------
// Transform Exclusive Scan
// ----------------------------------------------------------------------------

// Function: transform_exclusive_scan
template <typename B, typename E, typename D, typename T, typename BOP, typename UOP, typename P>
Task FlowBuilder::transform_exclusive_scan(
  B first, E last, D d_first, T init, BOP bop, UOP uop, P part
) {
  return emplace(make_transform_exclusive_scan_task(
    first, last, d_first, init, bop, uop, part
  )); 
}

}  // end of namespace tf -----------------------------------------------------

