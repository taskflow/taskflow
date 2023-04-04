#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default reduction
// ----------------------------------------------------------------------------

// Function: reduce
template <typename B, typename E, typename T, typename O, typename P>
Task FlowBuilder::reduce(B beg, E end, T& init, O bop, P&& part) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, &r=init, bop, part=std::forward<P>(part)] 
  (Runtime& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      for(; beg!=end; r = bop(r, *beg++));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;

    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {
      
      size_t curr_b = 0;
      size_t chunk_size;

      for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
        
        // we force chunk size to be at least two because the temporary
        // variable sum need to avoid copy at the first step
        chunk_size = std::max(size_t{2}, part.adjusted_chunk_size(N, W, w));
        
        auto loop = [=, &mutex, &r, &part] () mutable {

          std::advance(beg, curr_b);

          if(N - curr_b == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            r = bop(r, *beg);
            return;
          }

          auto beg1 = beg++;
          auto beg2 = beg++;
          T sum = bop(*beg1, *beg2);
        
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
                sum = bop(sum, *beg);
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(r, sum);

        };

        if(w == W-1) {
          loop();
        }
        else {
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      sf.join();
    }
    // dynamic partitioner
    else {

      std::atomic<size_t> next(0);

      auto loop = [=, &mutex, &next, &r, &part] () mutable {
        
        // pre-reduce
        size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

        if(s0 >= N) {
          return;
        }

        std::advance(beg, s0);

        if(N - s0 == 1) {
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(r, *beg);
          return;
        }

        auto beg1 = beg++;
        auto beg2 = beg++;

        T sum = bop(*beg1, *beg2);
        
        // loop reduce
        part.loop(N, W, next, 
          [&, prev_e=s0+2](size_t curr_b, size_t curr_e) mutable {
            std::advance(beg, curr_b - prev_e);
            for(size_t x=curr_b; x<curr_e; x++, beg++) {
              sum = bop(sum, *beg);
            }
            prev_e = curr_e;
          }
        ); 
        
        // final reduce
        std::lock_guard<std::mutex> lock(mutex);
        r = bop(r, sum);
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
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      // need to join here in case next goes out of scope
      sf.join();
    }
  });

  return task;
}

// ----------------------------------------------------------------------------
// default transform and reduction
// ----------------------------------------------------------------------------

// Function: transform_reduce
template <typename B, typename E, typename T, typename BOP, typename UOP, typename P>
Task FlowBuilder::transform_reduce(
  B beg, E end, T& init, BOP bop, UOP uop, P&& part
) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace(
  [b=beg, e=end, &r=init, bop, uop, part=std::forward<P>(part)] 
  (Runtime& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= part.chunk_size()) {
      for(; beg!=end; r = bop(std::move(r), uop(*beg++)));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    
    // static partitioner
    if constexpr(std::is_same_v<std::decay_t<P>, StaticPartitioner>) {
      
      size_t curr_b = 0;
      size_t chunk_size;

      for(size_t w=0; w<W && curr_b < N; ++w, curr_b += chunk_size) {
      
        chunk_size = part.adjusted_chunk_size(N, W, w);

        auto loop = [=, &mutex, &r, &part] () mutable {

          std::advance(beg, curr_b);

          if(N - curr_b == 1) {
            std::lock_guard<std::mutex> lock(mutex);
            r = bop(std::move(r), uop(*beg));
            return;
          }

          //auto beg1 = beg++;
          //auto beg2 = beg++;
          //T sum = bop(uop(*beg1), uop(*beg2));

          T sum = (chunk_size == 1) ? uop(*beg++) : bop(uop(*beg++), uop(*beg++));
        
          // loop reduce
          part.loop(N, W, curr_b, chunk_size,
            [&, prev_e=curr_b+(chunk_size == 1 ? 1 : 2)]
            (size_t curr_b, size_t curr_e) mutable {
              if(curr_b > prev_e) {
                std::advance(beg, curr_b - prev_e);
              }
              else {
                curr_b = prev_e;
              }
              for(size_t x=curr_b; x<curr_e; x++, beg++) {
                sum = bop(std::move(sum), uop(*beg));
              }
              prev_e = curr_e;
            }
          ); 
          
          // final reduce
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(std::move(r), std::move(sum));

        };

        if(w == W-1) {
          loop();
        }
        else {
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      
      sf.join();
    }
    // dynamic partitioner
    else {
      std::atomic<size_t> next(0);
        
      auto loop = [=, &mutex, &next, &r, &part] () mutable {

        // pre-reduce
        size_t s0 = next.fetch_add(2, std::memory_order_relaxed);

        if(s0 >= N) {
          return;
        }

        std::advance(beg, s0);

        if(N - s0 == 1) {
          std::lock_guard<std::mutex> lock(mutex);
          r = bop(std::move(r), uop(*beg));
          return;
        }

        auto beg1 = beg++;
        auto beg2 = beg++;

        T sum = bop(uop(*beg1), uop(*beg2));
        
        // loop reduce
        part.loop(N, W, next, 
          [&, prev_e=s0+2](size_t curr_b, size_t curr_e) mutable {
            std::advance(beg, curr_b - prev_e);
            for(size_t x=curr_b; x<curr_e; x++, beg++) {
              sum = bop(std::move(sum), uop(*beg));
            }
            prev_e = curr_e;
          }
        ); 
        
        // final reduce
        std::lock_guard<std::mutex> lock(mutex);
        r = bop(std::move(r), std::move(sum));
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
          sf._silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
        }
      }
      
      // need to join here in case next goes out of scope
      sf.join();
    }
  });

  return task;
}

}  // end of namespace tf -----------------------------------------------------




