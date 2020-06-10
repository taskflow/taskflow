#pragma once

// TF_VERSION % 100 is the patch level
// TF_VERSION / 100 % 1000 is the minor version
// TF_VERSION / 100000 is the major version

// current version: 2.6.0
#define TF_VERSION 200600

#define TF_MAJOR_VERSION TF_VERSION/100000
#define TF_MINOR_VERSION TF_VERSION/100%1000
#define TF_PATCH_VERSION TF_VERSION%100

#include <string>

namespace tf {

/**
@brief queries the version information in string
*/
constexpr const char* version() {
  return "2.6.0";
}


}  // end of namespace tf -----------------------------------------------------
