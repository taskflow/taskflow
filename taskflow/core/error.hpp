#pragma once

#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

namespace tf {

// node-specific states
struct NSTATE {

  using underlying_type = int;

  constexpr static underlying_type NONE        = 0x00000000;  
  constexpr static underlying_type CONDITIONED = 0x10000000;  
  constexpr static underlying_type DETACHED    = 0x20000000;  
  constexpr static underlying_type PREEMPTED   = 0x40000000;  

  // mask to isolate state bits - non-state bits store # weak dependents
  constexpr static underlying_type MASK        = 0xF0000000;
};

using nstate_t = NSTATE::underlying_type;

// exception-specific states
struct ESTATE {
  
  using underlying_type = int;  
  
  constexpr static underlying_type NONE      = 0; 
  constexpr static underlying_type EXCEPTION = 1;
  constexpr static underlying_type CANCELLED = 2;
  constexpr static underlying_type ANCHORED  = 4;  
};

using estate_t = ESTATE::underlying_type;

// async-specific states
struct ASTATE {
  
  using underlying_type = int;

  constexpr static underlying_type UNFINISHED = 0;
  constexpr static underlying_type LOCKED     = 1;
  constexpr static underlying_type FINISHED   = 2;
};

using astate_t = ASTATE::underlying_type;

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

