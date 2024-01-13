#pragma once

#include "../core/async.hpp"

namespace tf {

#define TF_MAKE_LOOP_TASK(code_block)                                                     \
  if constexpr(std::is_same_v<typename std::decay_t<P>::closure_wrapper_type, DefaultClosureWrapper>) { \
    code_block                                                                            \
  }                                                                                       \
  else {                                                                                  \
    std::invoke(part.closure_wrapper(), [&](){ code_block });                             \
  }                                                                                       

// Function: launch_loop
template <typename P, typename Loop>
TF_FORCE_INLINE void launch_loop(
  size_t N, 
  size_t W, 
  Runtime& rt, 
  std::atomic<size_t>& next, 
  P&& part, 
  Loop&& loop
) {

  //static_assert(std::is_lvalue_reference_v<Loop>, "");
  
  using namespace std::string_literals;

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
      rt.silent_async_unchecked("loop-"s + std::to_string(w), loop);
    }
  }
      
  rt.corun_all();
}

// Function: launch_loop
template <typename Loop>
TF_FORCE_INLINE void launch_loop(
  size_t W,
  size_t w,
  Runtime& rt, 
  Loop&& loop 
) {
  using namespace std::string_literals;
  if(w == W-1) {
    loop();
  }
  else {
    rt.silent_async_unchecked("loop-"s + std::to_string(w), loop);
  }
}

}  // end of namespace tf -----------------------------------------------------
