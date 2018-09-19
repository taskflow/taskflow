#pragma once

namespace tf {

// Struct: MoC
template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }
  
  mutable T object; 
};


};  // end of namespace tf. ---------------------------------------------------
