#pragma once

#include "../core/executor.hpp"

namespace tf{

template <typename B, typename E, typename D, typename BOP>
Task FlowBuilder::inclusive_scan(B first, E last, D d_first, BOP bop) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using D_t = std::decay_t<unwrap_ref_decay_t<D>>;
  using value_type = typename std::iterator_traits<B_t>::value_type;
  using namespace std::string_literals;
  
  Task task = emplace([=] (Runtime& rt) mutable {

    // fetch the stateful values
    B_t s_beg = first;
    E_t s_end = last;
    D_t d_beg = d_first;

    if(s_beg == s_end) {
      return;
    }

    size_t W = rt._executor.num_workers();
    size_t N = std::distance(s_beg, s_end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= 2) {
      std::inclusive_scan(s_beg, s_end, d_beg, bop);
      return;
    }

    if(N < W) {
      W = N;
    }
    
    size_t curr_b = 0;
    size_t chunk_size;
    
    std::vector<CachelineAligned<value_type>> buf(W);
    std::atomic<size_t> counter(0);

    size_t Q = N/W;
    size_t R = N%W;
    
    //auto orig_d_beg = d_beg;
    //ExecutionPolicy<StaticPartitioner> policy;

    for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {

      chunk_size = std::min(Q + (w < R), N - curr_b);

      // block scan
      auto partial_scan = [=, &buf, &counter] () mutable {

        auto res_d_beg = d_beg;

        // local scan per worker
        auto& init = buf[w].data;
        init = *s_beg++;
        *d_beg++ = init;

        for(size_t i=1; i<chunk_size; i++){
          *d_beg++ = init = bop(init, *s_beg++); 
        }

        // whoever finishes the last performs global scan
        if(counter.fetch_add(1, std::memory_order_acq_rel) == W-1) {
          for(size_t i=1; i<buf.size(); i++) {
            buf[i].data = bop(buf[i-1].data, buf[i].data);
          }
          counter.store(0, std::memory_order_release);
        }
        
        // first partition is done 
        if(w == 0) {
          return;
        }
        
        // synchronize without blocking
        rt._executor.corun_until([&counter](){
          return counter.load(std::memory_order_acquire) == 0;
        });

        
        // local scan based on the global scanned value
        for(size_t i=0; i<chunk_size; i++) {
          *res_d_beg++ = bop(buf[w-1].data, *res_d_beg);
        }
        
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
      };

      if(w == W-1) {
        partial_scan();
      }
      else {
        rt._silent_async(rt._worker, "loop-"s + std::to_string(w), partial_scan);
      }
      
      std::advance(s_beg, chunk_size);
      std::advance(d_beg, chunk_size);
    }

    rt.join();
    
  });
  
  return task;
}

}  // end of namespace tf -----------------------------------------------------
