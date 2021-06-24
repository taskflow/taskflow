#pragma once

#include "nonreachable_sanitizer.hpp"

namespace tf {

class Sanitizer {

  public:
    
    // TODO: (McKay)
    // migrate your implementation to nonreachable_sanitizer.hpp
    // and write some examples to test (in examples/sanitizer.cpp)
    std::vector<Task> check_nonreachable(const Taskflow& taskflow, std::ostream& os) {
      
      os << "hello I am checking your taskflow with nonreachable tasks\n";

      NonReachableSanitizer san(taskflow);

      return san(os);
    }

};


} // end of namespace tf 
