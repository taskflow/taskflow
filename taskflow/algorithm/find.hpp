#pragma once

#include "launch.hpp"

namespace tf {

namespace detail {

// Function: find_if_loop
template <typename Iterator, typename Predicate>
TF_FORCE_INLINE bool find_if_loop(
  std::atomic<size_t>& offset, 
  Iterator& beg,
  size_t& prev_e,
  size_t  curr_b, 
  size_t  curr_e,
  Predicate&& predicate
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

// Function: find_if_not_loop
template <typename Iterator, typename Predicate>
TF_FORCE_INLINE bool find_if_not_loop(
  std::atomic<size_t>& offset, 
  Iterator& beg,
  size_t& prev_e,
  size_t  curr_b, 
  size_t  curr_e,
  Predicate&& predicate
) {

  // early prune
  if(offset.load(std::memory_order_relaxed) < curr_b) {
    return true;
  }
  std::advance(beg, curr_b - prev_e);
  for(size_t x = curr_b; x<curr_e; x++) {
    if(!predicate(*beg++)) {
      atomic_min(offset, x);
      return true;
    }
  }
  prev_e = curr_e;
  return false;
}

}  // namespace detail --------------------------------------------------------

// Function: make_find_if_task
template <typename B, typename E, typename T, typename UOP, typename P = GuidedPartitioner>
TF_FORCE_INLINE auto make_find_if_task(
  B first, E last, T& result, UOP predicate, P&& part = P()
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return 
  [b=first, e=last, predicate, &result, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
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

      size_t chunk_size;

      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = part.adjusted_chunk_size(N, W, w);

        launch_loop(W, w, rt,
          [N, W, curr_b, chunk_size, beg, &predicate, &offset, &part] 
          () mutable {
            part.loop_until(N, W, curr_b, chunk_size,
              [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
                return detail::find_if_loop(
                  offset, beg, prev_e, curr_b, curr_e, predicate
                );
              }
            ); 
          }
        );
      }

      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, 
        [N, W, beg, &predicate, &offset, &next, &part] () mutable {
          part.loop_until(N, W, next, 
            [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
              return detail::find_if_loop(
                offset, beg, prev_e, curr_b, curr_e, predicate
              );
            }
          ); 
        }
      );
    }

    // update the result iterator by the offset
    result = std::next(beg, offset.load(std::memory_order_relaxed));
  };
}

// Function: make_find_if_not_task
template <typename B, typename E, typename T, typename UOP, typename P = GuidedPartitioner>
TF_FORCE_INLINE auto make_find_if_not_task(
  B first, E last, T& result, UOP predicate, P&& part = P()
) {
  
  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return
  [b=first, e=last, predicate, &result, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the stateful values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      result = std::find_if_not(beg, end, predicate);
      return;
    }

    if(N < W) {
      W = N;
    }
    
    std::atomic<size_t> offset(N);
    
    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {

      size_t chunk_size;

      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = part.adjusted_chunk_size(N, W, w);

        launch_loop(W, w, rt,
          [N, W, curr_b, chunk_size, beg, &predicate, &offset, &part] () mutable {
            part.loop_until(N, W, curr_b, chunk_size,
              [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
                return detail::find_if_not_loop(
                  offset, beg, prev_e, curr_b, curr_e, predicate
                );
              }
            ); 
          }
        );
      }

      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part,
        [N, W, beg, &predicate, &offset, &next, &part] () mutable {
          part.loop_until(N, W, next, 
            [&, prev_e=size_t{0}](size_t curr_b, size_t curr_e) mutable {
              return detail::find_if_not_loop(
                offset, beg, prev_e, curr_b, curr_e, predicate
              );
            }
          ); 
        }
      );
    }

    // update the result iterator by the offset
    result = std::next(beg, offset.load(std::memory_order_relaxed));
  };
}

// Function: make_min_element_task
template <typename B, typename E, typename T, typename C, typename P = GuidedPartitioner>
TF_FORCE_INLINE auto make_min_element_task(
  B first, E last, T& result, C comp, P&& part = P()
) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return 
  [b=first, e=last, &result, comp, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      result = std::min_element(beg, end, comp);
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    
    // initialize the result to the first element
    result = beg++;
    N--;

    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {
      
      size_t chunk_size;

      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        
        // we force chunk size to be at least two because the temporary
        // variable sum needs to avoid copy at the first step
        chunk_size = std::max(size_t{2}, part.adjusted_chunk_size(N, W, w));
        
        launch_loop(W, w, rt,
        [beg, curr_b, N, W, chunk_size, &comp, &mutex, &result, &part] () mutable {

          std::advance(beg, curr_b);

          if(N - curr_b == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            if(comp(*beg, *result)) {
              result = beg;
            }
            return;
          }

          auto beg1 = beg++;
          auto beg2 = beg++;
          T smallest = comp(*beg1, *beg2) ? beg1 : beg2;
        
          // loop reduce
          part.loop(N, W, curr_b, chunk_size,
            [&, prev_e=curr_b+2](size_t curr_b, size_t curr_e) mutable {

              if(curr_b > prev_e) {
                std::advance(beg, curr_b - prev_e);
              }
              else {
                curr_b = prev_e;
              }

              for(size_t x=curr_b; x<curr_e; x++, beg++) {
                if(comp(*beg, *smallest)) {
                  smallest = beg;
                }
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          if(comp(*smallest, *result)) {
            result = smallest;
          }
        });
      }
      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part, 
        [beg, N, W, &next, &comp, &mutex, &result, &part] () mutable {
          // pre-reduce
          size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

          if(s0 >= N) {
            return;
          }

          std::advance(beg, s0);

          if(N - s0 == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            if(comp(*beg, *result)) {
              result = beg;
            }
            return;
          }

          auto beg1 = beg++;
          auto beg2 = beg++;

          T smallest = comp(*beg1, *beg2) ? beg1 : beg2;
          
          // loop reduce
          part.loop(N, W, next, 
            [&, prev_e=s0+2](size_t curr_b, size_t curr_e) mutable {
              std::advance(beg, curr_b - prev_e);
              for(size_t x=curr_b; x<curr_e; x++, beg++) {
                if(comp(*beg, *smallest)) {
                  smallest = beg;
                }
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          if(comp(*smallest, *result)) {
            result = smallest;
          }
        }
      );
    }
  };
}

// Function: make_max_element_task
template <typename B, typename E, typename T, typename C, typename P = GuidedPartitioner>
TF_FORCE_INLINE auto make_max_element_task(
  B first, E last, T& result, C comp, P&& part = P()
) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  return 
  [b=first, e=last, &result, comp, part=std::forward<P>(part)] 
  (Runtime& rt) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    size_t W = rt.executor().num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      result = std::max_element(beg, end, comp);
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    
    // initialize the result to the first element
    result = beg++;
    N--;

    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {
      
      size_t chunk_size;

      for(size_t w=0, curr_b=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        
        // we force chunk size to be at least two because the temporary
        // variable sum needs to avoid copy at the first step
        chunk_size = std::max(size_t{2}, part.adjusted_chunk_size(N, W, w));
        
        launch_loop(W, w, rt,
        [beg, curr_b, N, W, chunk_size, &comp, &mutex, &result, &part] () mutable {

          std::advance(beg, curr_b);

          if(N - curr_b == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            if(comp(*result, *beg)) {
              result = beg;
            }
            return;
          }

          auto beg1 = beg++;
          auto beg2 = beg++;
          T largest = comp(*beg1, *beg2) ? beg2 : beg1;
        
          // loop reduce
          part.loop(N, W, curr_b, chunk_size,
            [&, prev_e=curr_b+2](size_t curr_b, size_t curr_e) mutable {

              if(curr_b > prev_e) {
                std::advance(beg, curr_b - prev_e);
              }
              else {
                curr_b = prev_e;
              }

              for(size_t x=curr_b; x<curr_e; x++, beg++) {
                if(comp(*largest, *beg)) {
                  largest = beg;
                }
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          if(comp(*result, *largest)) {
            result = largest;
          }
        });
      }
      rt.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
      launch_loop(N, W, rt, next, part,
        [beg, N, W, &next, &comp, &mutex, &result, &part] () mutable {
          // pre-reduce
          size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

          if(s0 >= N) {
            return;
          }

          std::advance(beg, s0);

          if(N - s0 == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            if(comp(*result, *beg)) {
              result = beg;
            }
            return;
          }

          auto beg1 = beg++;
          auto beg2 = beg++;

          T largest = comp(*beg1, *beg2) ? beg2 : beg1;
          
          // loop reduce
          part.loop(N, W, next, 
            [&, prev_e=s0+2](size_t curr_b, size_t curr_e) mutable {
              std::advance(beg, curr_b - prev_e);
              for(size_t x=curr_b; x<curr_e; x++, beg++) {
                if(comp(*largest, *beg)) {
                  largest = beg;
                }
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          if(comp(*result, *largest)) {
            result = largest;
          }
        }
      );
    }
  };
}



// Function: find_if
template <typename B, typename E, typename T, typename UOP, typename P>
Task tf::FlowBuilder::find_if(B first, E last, T& result, UOP predicate, P&& part) {
  return emplace(make_find_if_task(
    first, last, result, predicate, std::forward<P>(part)
  ));
}

// Function: find_if_not
template <typename B, typename E, typename T, typename UOP, typename P>
Task tf::FlowBuilder::find_if_not(B first, E last, T& result, UOP predicate, P&& part) {
  return emplace(make_find_if_not_task(
    first, last, result, predicate, std::forward<P>(part)
  ));
}

// ----------------------------------------------------------------------------
// min_element
// ----------------------------------------------------------------------------

// Function: min_element
template <typename B, typename E, typename T, typename C, typename P>
Task FlowBuilder::min_element(B first, E last, T& result, C comp, P&& part) {
  return emplace(make_min_element_task(
    first, last, result, comp, std::forward<P>(part)
  ));
}

// ----------------------------------------------------------------------------
// max_element
// ----------------------------------------------------------------------------

// Function: max_element
template <typename B, typename E, typename T, typename C, typename P>
Task FlowBuilder::max_element(B first, E last, T& result, C comp, P&& part) {
  return emplace(make_max_element_task(
    first, last, result, comp, std::forward<P>(part)
  ));
}

}  // end of namespace tf -----------------------------------------------------
