#pragma once

#include "core/executor.hpp"
#include "algorithm/for_each.hpp"
#include "algorithm/reduce.hpp"

// TF_VERSION % 100 is the patch level
// TF_VERSION / 100 % 1000 is the minor version
// TF_VERSION / 100000 is the major version

// current version: 2.7.0
#define TF_VERSION 200700

#define TF_MAJOR_VERSION TF_VERSION/100000
#define TF_MINOR_VERSION TF_VERSION/100%1000
#define TF_PATCH_VERSION TF_VERSION%100

namespace tf {

/**
@brief queries the version information in string
*/
constexpr const char* version() {
  return "2.7.0";
}


}  // end of namespace tf -----------------------------------------------------





