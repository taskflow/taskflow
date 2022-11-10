#pragma once

#include "../core/executor.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// default reduction
// ----------------------------------------------------------------------------

template <typename B, typename E, typename T, typename O>
Task FlowBuilder::reduce(B beg, E end, T& init, O bop) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, &r=init, bop] (Subflow& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    //size_t C = (c == 0) ? 1 : c;
    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      for(; beg!=end; r = bop(r, *beg++));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);
      
    auto loop = [=, &mutex, &next, &r] () mutable {

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

      size_t z = s0 + 2;
      size_t p1 = 2 * W * (chunk_size + 1);
      double p2 = 0.5 / static_cast<double>(W);
      s0 = next.load(std::memory_order_relaxed);

      while(s0 < N) {

        size_t r = N - s0;

        // fine-grained
        if(r < p1) {
          while(1) {
            s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
            if(s0 >= N) {
              break;
            }
            size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
            std::advance(beg, s0-z);
            for(size_t x=s0; x<e0; x++, beg++) {
              sum = bop(sum, *beg);
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
          if(next.compare_exchange_strong(s0, e0, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            std::advance(beg, s0-z);
            for(size_t x = s0; x<e0; x++, beg++) {
              sum = bop(sum, *beg);
            }
            z = e0;
            s0 = next.load(std::memory_order_relaxed);
          }
        }
      }

      std::lock_guard<std::mutex> lock(mutex);
      r = bop(r, sum);
    };

    for(size_t w=0; w<W; w++) {

      //if(w*2 >= N) {
      //  break;
      //}
      //sf._named_silent_async(
      //  sf._worker, "part-"s + std::to_string(w), loop
      //);
      
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= chunk_size || w == W-1) {
        loop(); 
        break;
      }
      else {
        sf._named_silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
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
  B beg, E end, T& init, BOP bop, UOP uop
) {

  using B_t = std::decay_t<unwrap_ref_decay_t<B>>;
  using E_t = std::decay_t<unwrap_ref_decay_t<E>>;
  using namespace std::string_literals;

  Task task = emplace([b=beg, e=end, &r=init, bop, uop] (Subflow& sf) mutable {

    // fetch the iterator values
    B_t beg = b;
    E_t end = e;

    if(beg == end) {
      return;
    }

    //size_t chunk_size = (c == 0) ? 1 : c;
    size_t chunk_size = 1;
    size_t W = sf._executor.num_workers();
    size_t N = std::distance(beg, end);

    // only myself - no need to spawn another graph
    if(W <= 1 || N <= chunk_size) {
      for(; beg!=end; r = bop(std::move(r), uop(*beg++)));
      return;
    }

    if(N < W) {
      W = N;
    }

    std::mutex mutex;
    std::atomic<size_t> next(0);
      
    auto loop = [=, &mutex, &next, &r] () mutable {

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

      size_t z = s0 + 2;
      size_t p1 = 2 * W * (chunk_size + 1);
      double p2 = 0.5 / static_cast<double>(W);
      s0 = next.load(std::memory_order_relaxed);

      while(s0 < N) {

        size_t r = N - s0;

        // fine-grained
        if(r < p1) {
          while(1) {
            s0 = next.fetch_add(chunk_size, std::memory_order_relaxed);
            if(s0 >= N) {
              break;
            }
            size_t e0 = (chunk_size <= (N - s0)) ? s0 + chunk_size : N;
            std::advance(beg, s0-z);
            for(size_t x=s0; x<e0; x++, beg++) {
              sum = bop(std::move(sum), uop(*beg));
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
          if(next.compare_exchange_strong(s0, e0, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            std::advance(beg, s0-z);
            for(size_t x = s0; x<e0; x++, beg++) {
              sum = bop(std::move(sum), uop(*beg));
            }
            z = e0;
            s0 = next.load(std::memory_order_relaxed);
          }
        }
      }

      std::lock_guard<std::mutex> lock(mutex);
      r = bop(std::move(r), std::move(sum));
    };

    for(size_t w=0; w<W; w++) {
      //if(w*2 >= N) {
      //  break;
      //}
      //sf._named_silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
      
      auto r = N - next.load(std::memory_order_relaxed);
      // no more loop work to do - finished by previous async tasks
      if(!r) {
        break;
      }
      // tail optimization
      if(r <= chunk_size || w == W-1) {
        loop(); 
        break;
      }
      else {
        sf._named_silent_async(sf._worker, "loop-"s + std::to_string(w), loop);
      }
    }

    sf.join();
  });

  return task;
}

}  // end of namespace tf -----------------------------------------------------




