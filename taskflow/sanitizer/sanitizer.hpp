#pragma once

#include "../core/taskflow.hpp"

namespace tf {

class Sanitizer {


  public:

    bool check_nonreachable(const Taskflow& taskflow, std::ostream& os) {
      
      os << "hello I am checking your taskflow with nonreachable tasks\n";
      
      // TODO (McKay): design the graph data structure extracted from taskflow
      //               understand tf::Taskflow::dump

      return false;
    }

};


} // end of namespace tf 
