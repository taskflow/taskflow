#pragma once

#include <taskflow/core/graph.hpp>

namespace tf {

/**
@private
*/
class Algorithm {

  public:

  template <HasGraph T>
  static auto make_module_task(T&);
};

}  // end of namespace tf -----------------------------------------------------
