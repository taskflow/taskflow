#pragma once

#include "../core/executor.hpp"

namespace tf {

namespace detail {

template <typename Iterator, typename Predicate>
TF_FORCE_INLINE bool find_if_loop(
  std::atomic<size_t>& offset, 
  Iterator& beg,
  size_t& prev_e,
  size_t  curr_b, 
  size_t  curr_e,
  Predicate predicate
) {
  // early prune
  if(offset.load(std::memory_order_relaxed) < curr_b) {
    return true;
  }
  std::advance(beg, curr_b - prev_e);
  for(size_t x = curr_b; x<curr_e; x++) {
    if(predicate(*beg++)) {
      atomic_min(offset, x);
      return true;
    }
  }
  prev_e = curr_e;
  return false;
}

}  // namespace detail --------------------------------------------------------

// Function: find_if
template <typename B, typename E, typename UOP, typename T, typename P>
Task tf::FlowBuilder::find_if(B first, E last, UOP predicate, T& result, P&& part) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=first, e=last, predicate, &result, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = rt._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      result = std::find_if(beg, end, predicate);
      return;
    }

    if(N < W) {
      W = N;
    }
    
    std::atomic<size_t> offset(N);
    
    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {

      size_t curr_b = 0;
      size_t chunk_size;

      for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = part.adjusted_chunk_size(N, W, w);

        auto loop = [=, &offset, &part] () mutable {
          part.loop_until(N, W, curr_b, chunk_size,
            [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
              // early prune
              if(offset.load(std::memory_order_relaxed) < curr_b) {
                return true;
              }
              std::advance(beg, curr_b - prev_e);
              for(size_t x = curr_b; x<curr_e; x++) {
                if(predicate(*beg++)) {
                  atomic_min(offset, x);
                  return true;
                }
              }
              prev_e = curr_e;
              return false;
            }
          ); 
        };

        if(w == W-1) {
          loop();
        }
        else {
          rt._silent_async(rt._worker, "loop-"s + std::to_string(w), loop);
        }
      }

      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);

      auto loop = [=, &offset, &next, &part] () mutable {
        part.loop_until(N, W, next, 
          [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
            // early prune
            if(offset.load(std::memory_order_relaxed) < curr_b) {
              return true;
            }
            std::advance(beg, curr_b - prev_e);
            for(size_t x = curr_b; x<curr_e; x++) {
              if(predicate(*beg++)) {
                atomic_min(offset, x);
                return true;
              }
            }
            prev_e = curr_e;
            return false;
          }
        ); 
      };

      for(size_t w=0; w<W; w++) {
        auto r = N - next.load(std::memory_order_relaxed);
        // no more loop work to do - finished by previous async tasks
        if(!r) {
          break;
        }
        // tail optimization
        if(r <= part.chunk_size() || w == W-1) {
          loop();
          break;
        }
        else {
          rt._silent_async(rt._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      // need to join here in case next goes out of scope
      rt.join();
    }

    // update the result iterator by the offset
    auto loaded_offset = offset.load(std::memory_order_relaxed);
    result = (loaded_offset == N) ? end : std::next(beg, loaded_offset);
  });

  return task;
}

}  // end of namespace tf -----------------------------------------------------
