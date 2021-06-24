#pragma once

#include "nonreachable_sanitizer.hpp"
#include "infinite_loop_sanitizer.hpp"
#include "deadlock_sanitizer.hpp"

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

      // TODO (Luke): 
    std::vector<std::vector<Task>> check_infinite_loop(const Taskflow& taskflow, std::ostream& os) {

      os << "hello I am checking your taskflow with infinite loops\n";

      InfiniteLoopSanitizer san(taskflow);

      return san(os);
    }

};


} // end of namespace tf 
