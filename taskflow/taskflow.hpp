#pragma once

#include "core/executor.hpp"
#include "algorithm/critical.hpp"
#include "algorithm/for_each.hpp"

/**
@dir taskflow
@brief root taskflow include dir
*/

/**
@dir taskflow/core
@brief taskflow core include dir
*/

/**
@dir taskflow/algorithm
@brief taskflow algorithms include dir
*/

/**
@dir taskflow/cuda
@brief taskflow CUDA include dir
*/

/**
@file taskflow/taskflow.hpp
@brief main taskflow include file
*/

// TF_VERSION % 100 is the patch level
// TF_VERSION / 100 % 1000 is the minor version
// TF_VERSION / 100000 is the major version

// current version: 3.5.0
#define TF_VERSION 300500

#define TF_MAJOR_VERSION TF_VERSION/100000
#define TF_MINOR_VERSION TF_VERSION/100%1000
#define TF_PATCH_VERSION TF_VERSION%100

/**
@brief taskflow namespace
*/
namespace tf {

/**
@private
*/
namespace detail { }


/**
@brief queries the version information in a string format @c major.minor.patch

Release notes are available here: https://taskflow.github.io/taskflow/Releases.html
*/
constexpr const char* version() {
  return "3.5.0";
}


}  // end of namespace tf -----------------------------------------------------





