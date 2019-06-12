// 2019/06/12 - created by Tsung-Wei Huang
//   - modified from tbb and std::random device source

#pragma once

#include <cassert>
#include <thread>

namespace tf {

/** @class FastRandom 
@brief A fast random number generator that uses the linear congruential method
*/
class FastRandom {

  public:

    //! Get a random number.
    unsigned short get() {
      return get(_x);
    }
    
    //! Get a random number for the given seed; update the seed for next use.
    unsigned short get(unsigned& seed) {
      unsigned short number = static_cast<unsigned short>(seed >> 16);
      assert(_c & 1);
      seed = seed*_r + _c;
      return number;
    }

    FastRandom(uint32_t seed) {
      // shuffle the c and x variables
      _c = (seed | 1) * 0xba5703f5;
      _x = _c^(seed >> 1);
    }

  private:

    unsigned _x;
    unsigned _c;
    static const unsigned _r = 0x9e3779b1;
};

template <typename T>
inline T per_thread_seed() {
  return static_cast<T>(std::hash<std::thread::id>()(std::this_thread::get_id()));
}

}  // end of namespace tf -----------------------------------------------------



