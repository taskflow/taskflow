#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <random>
#include <chrono>

namespace tf {

// Class: UUID
//
// A universally unique identifier (UUID) is an identifier standard used in software 
// construction. A UUID is simply a 128-bit value. The meaning of each bit is defined 
// by any of several variants.
// For human-readable display, many systems use a canonical format using hexadecimal 
// text with inserted hyphen characters.
// 
// For example: 123e4567-e89b-12d3-a456-426655440000
//
// The intent of UUIDs is to enable distributed systems to uniquely identify information 
// without significant central coordination. 
//
//   Copyright 2006 Andy Tompkins.
//   Distributed under the Boost Software License, Version 1.0. (See
//   accompanying file LICENSE_1_0.txt or copy at
//   http://www.boost.org/LICENSE_1_0.txt)
//
struct UUID {

  using value_type      = uint8_t;
  using reference       = uint8_t&;
  using const_reference = const uint8_t&;
  using iterator        = uint8_t*;
  using const_iterator  = const uint8_t*;
  using size_type       = size_t;
  using difference_type = ptrdiff_t;

  inline UUID();

  UUID(const UUID&) = default;
  UUID(UUID&&) = default;

  UUID& operator = (const UUID&) = default;
  UUID& operator = (UUID&&) = default;
    
  inline static size_type size(); 
  inline iterator begin(); 
  inline const_iterator begin() const; 
  inline iterator end(); 
  inline const_iterator end() const; 

  inline bool is_nil() const;
  inline void swap(UUID& rhs);
  inline size_t hash_value() const;

  inline bool operator == (const UUID&) const;
  inline bool operator <  (const UUID&) const;
  inline bool operator >  (const UUID&) const;
  inline bool operator != (const UUID&) const;
  inline bool operator >= (const UUID&) const;
  inline bool operator <= (const UUID&) const; 

  uint8_t data[16] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  inline std::string to_string() const;
};

// Constructor
inline UUID::UUID() {

  static thread_local std::default_random_engine engine {
    std::random_device{}()
  };

  std::uniform_int_distribution<unsigned long> distribution(
    (std::numeric_limits<unsigned long>::min)(),
    (std::numeric_limits<unsigned long>::max)()
  );
  
  int i = 0;
  auto random_value = distribution(engine);
  for (auto it=begin(); it!=end(); ++it, ++i) {
    if (i == sizeof(unsigned long)) {
      random_value = distribution(engine);
      i = 0;
    }
    *it = static_cast<UUID::value_type>((random_value >> (i*8)) & 0xFF);
  }
  
  // set variant: must be 0b10xxxxxx
  *(begin()+8) &= 0xBF;
  *(begin()+8) |= 0x80;

  // set version: must be 0b0100xxxx
  *(begin()+6) &= 0x4F; //0b01001111
  *(begin()+6) |= 0x40; //0b01000000 
}
  
// Function: size
inline typename UUID::size_type UUID::size() { 
  return 16;          
}

// Function: begin
inline typename UUID::iterator UUID::begin() { 
  return data;        
}

// Function: begin
inline typename UUID::const_iterator UUID::begin() const { 
  return data;        
}

// Function: end
inline typename UUID::iterator UUID::end() { 
  return data+size(); 
}

// Function: end
inline typename UUID::const_iterator UUID::end() const { 
  return data+size(); 
}

// Function: is_nil
inline bool UUID::is_nil() const {
  for (std::size_t i = 0; i < sizeof(this->data); ++i) {
    if (this->data[i] != 0U) {
      return false;
    }
  }
  return true;
}

// Procedure: swap
inline void UUID::swap(UUID& rhs) {
  UUID tmp = *this;
  *this = rhs;
  rhs = tmp;
}

// Function: hash_value
inline size_t UUID::hash_value() const {
  size_t seed = 0;
  for(auto i=begin(); i != end(); ++i) {
    seed ^= static_cast<size_t>(*i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

// Operator: ==
inline bool UUID::operator == (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) == 0;
}

// Operator: !=
inline bool UUID::operator != (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) != 0;
}

// Operator: <
inline bool UUID::operator < (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) < 0;
}

// Operator: >
inline bool UUID::operator > (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) > 0;
}

// Operator: <=
inline bool UUID::operator <= (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) <= 0;
}

// Operator: >=
inline bool UUID::operator >= (const UUID& rhs) const {
  return std::memcmp(data, rhs.data, sizeof(data)) >= 0;
}

// Function: to_string
inline std::string UUID::to_string() const {

  auto to_char = [](size_t i) {
    if (i <= 9) return static_cast<char>('0' + i);
    return static_cast<char>('a' + (i-10));
  };

  std::string result;
  result.reserve(36);

  std::size_t i=0;
  for (auto it = begin(); it!=end(); ++it, ++i) {

    const size_t hi = ((*it) >> 4) & 0x0F;
    result += to_char(hi);

    const size_t lo = (*it) & 0x0F;
    result += to_char(lo);

    if (i == 3 || i == 5 || i == 7 || i == 9) {
      result += '-';
    }
  }
  return result;
}

// Procedure: swap
inline void swap(UUID& lhs, UUID& rhs) {
  lhs.swap(rhs);
}

// ostream
inline std::ostream& operator << (std::ostream& os, const UUID& rhs) {
  os << rhs.to_string();
  return os;
}

}  // End of namespace tf. ----------------------------------------------------

//-----------------------------------------------------------------------------


namespace std {

// Partial specialization: hash<tf::UUID>
template <>
struct hash<tf::UUID> {
  size_t operator()(const tf::UUID& rhs) const { return rhs.hash_value(); }
};


}  // End of namespace std. ---------------------------------------------------


