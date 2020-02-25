#pragma once

#include <iostream>
#include <string>

namespace tf {

// Procedure: stringify
template <typename T>
void ostreamize(std::ostringstream& oss, T&& token) {
  oss << std::forward<T>(token);  
}

// Procedure: stringify
template <typename T, typename... Rest>
void ostreamize(std::ostringstream& oss, T&& token, Rest&&... rest) {
  oss << std::forward<T>(token);
  ostreamize(oss, std::forward<Rest>(rest)...);
}

}  // end of namespace tf -----------------------------------------------------
