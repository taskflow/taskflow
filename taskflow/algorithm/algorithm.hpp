#pragma once

namespace tf {

class Algorithm {

  public:

  template <typename T>
  static auto make_module_task(T&&);

};


}  // end of namespace tf -----------------------------------------------------
