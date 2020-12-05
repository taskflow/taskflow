#pragma once

#include "../utility/traits.hpp"
#include "../utility/os.hpp"

namespace tf {

enum EnvironmentVariable {
  TF_ENABLE_PROFILER = 0
};

inline std::string get_env(EnvironmentVariable v) {
  switch(v) {
    case TF_ENABLE_PROFILER: return get_env("TF_ENABLE_PROFILER");
    default: return "";
  }
}

inline bool has_env(EnvironmentVariable v) {
  switch(v) {
    case TF_ENABLE_PROFILER: return has_env("TF_ENABLE_PROFILER");
    default: return false;
  }
}

}  // end of namespace tf -----------------------------------------------------

