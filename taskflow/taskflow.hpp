#pragma once

#include "core/executor.hpp"
#include "core/algorithm/for_each.hpp"
#include "core/algorithm/reduce.hpp"

/**
@file taskflow/taskflow.hpp
*/

// TF_VERSION % 100 is the patch level
// TF_VERSION / 100 % 1000 is the minor version
// TF_VERSION / 100000 is the major version

// current version: 3.0.0
#define TF_VERSION 300000

#define TF_MAJOR_VERSION TF_VERSION/100000
#define TF_MINOR_VERSION TF_VERSION/100%1000
#define TF_PATCH_VERSION TF_VERSION%100

namespace tf {

/**
@brief queries the version information in a string format @c major.minor.patch
*/
constexpr const char* version() {
  return "3.0.0";
}


}  // end of namespace tf -----------------------------------------------------





