#pragma once

#include <iostream>
#include <string>

namespace tf {

// Procedure: ostreamize
template <typename T>
void ostreamize(std::ostream& os, T&& token) {
  os << std::forward<T>(token);  
}

// Procedure: ostreamize
template <typename T, typename... Rest>
void ostreamize(std::ostream& os, T&& token, Rest&&... rest) {
  os << std::forward<T>(token);
  ostreamize(os, std::forward<Rest>(rest)...);
}


}  // end of namespace tf -----------------------------------------------------

