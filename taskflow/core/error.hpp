#pragma once

#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

namespace tf {

// node-specific states
// 32-bit state encoding [xxxx|xxxx|xxxx|xxxx|xxxx|xxxx|xxxx|xxxx]
struct NSTATE {

  using underlying_type = int;
  
  // scheduler state bits
  constexpr static underlying_type NONE                = 0x00000000;  
  constexpr static underlying_type CONDITIONED         = 0x10000000;  
  constexpr static underlying_type PREEMPTED           = 0x20000000;  
  constexpr static underlying_type RETAIN_SUBFLOW      = 0x40000000;
  constexpr static underlying_type JOINED_SUBFLOW      = 0x80000000;

  // exception state bits
  constexpr static underlying_type IMPLICITLY_ANCHORED = 0x01000000;

  // mask to isolate state bits - non-state bits store # weak dependents
  constexpr static underlying_type MASK                = 0xFF000000;
};

using nstate_t = NSTATE::underlying_type;

// exception-specific states
struct ESTATE {
  
  using underlying_type = int;  
  
  constexpr static underlying_type NONE                = 0x00000000; 
  
  // Exception state:
  // Explicit anchor needs to be in estate other it can cause data race
  // due to the read/write on creating an AnchorGuard in corun with the nstate read
  // in tear_down_async. When calling corun, all async tasks may have already
  // finished and trigger tear_down_async, causing data race on reading/writing
  // the parent node's nstate.
  constexpr static underlying_type EXCEPTION           = 0x10000000;
  constexpr static underlying_type CAUGHT              = 0x20000000;
  constexpr static underlying_type CANCELLED           = 0x40000000;
  constexpr static underlying_type EXPLICITLY_ANCHORED = 0x80000000;
  
  // Async task state
  //constexpr static underlying_type UNFINISHED = 0x00000000;
  constexpr static underlying_type LOCKED              = 0x01000000;
  constexpr static underlying_type FINISHED            = 0x02000000;
  
  // mask to isolate state bits - non-state bits store # weak dependents
  constexpr static underlying_type MASK                = 0xFF000000;
};

using estate_t = ESTATE::underlying_type;


// Procedure: throw_re
// Throws runtime error under a given error code.
template <typename... ArgsT>
//void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&... args) {
void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  //ostreamize(oss, std::forward<ArgsT>(args)...);
  (oss << ... << args);
#ifdef TF_DISABLE_EXCEPTION_HANDLING
  std::cerr << oss.str();
  std::terminate();
#else
  throw std::runtime_error(oss.str());
#endif
}

}  // ------------------------------------------------------------------------

#define TF_THROW(...) tf::throw_re(__FILE__, __LINE__, __VA_ARGS__);

// ----------------------------------------------------------------------------

#ifdef TF_DISABLE_EXCEPTION_HANDLING
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block) \
    code_block;
#else
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block)  \
    try {                                          \
      code_block;                                  \
    } catch(...) {                                 \
      _process_exception(worker, node);            \
    }
#endif


