#pragma once

#include <iostream>
#include <sstream>
#include <exception>
#include <system_error>

namespace tf {

/**
@struct Error 

@brief The error category of taskflow.
*/
struct Error : public std::error_category {
  
  /**
  @enum Code 
  @brief Error code definition.
  */
  enum Code : int {
    SUCCESS = 0,
    FLOW_BUILDER,
    EXECUTOR
  };
  
  /**
  @brief returns the name of the taskflow error category
  */
  inline const char* name() const noexcept override final;

  /**
  @brief acquires the singleton instance of the taskflow error category
  */
  inline static const std::error_category& get();
  
  /**
  @brief query the human-readable string of each error code
  */
  inline std::string message(int) const override final;
};

// Function: name
inline const char* Error::name() const noexcept {
  return "Taskflow error";
}

// Function: get 
inline const std::error_category& Error::get() {
  static Error instance;
  return instance;
}

// Function: message
inline std::string Error::message(int code) const {
  switch(auto ec = static_cast<Error::Code>(code); ec) {
    case SUCCESS:
      return "success";
    break;

    case FLOW_BUILDER:
      return "flow builder error";
    break;

    case EXECUTOR:
      return "executor error";
    break;

    default:
      return "unknown";
    break;
  };
}

// Function: make_error_code
// Argument dependent lookup.
inline std::error_code make_error_code(Error::Code e) {
  return std::error_code(static_cast<int>(e), Error::get());
}

}  // end of namespace tf ----------------------------------------------------

// Register for implicit conversion  
namespace std {
  template <>
  struct is_error_code_enum<tf::Error::Code> : true_type {};
}

// ----------------------------------------------------------------------------

namespace tf {

// Procedure: throw_se
// Throws the system error under a given error code.
template <typename... ArgsT>
void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  (oss << ... << args);
  throw std::system_error(c, oss.str());
}

}  // ------------------------------------------------------------------------

#define TF_THROW(...) tf::throw_se(__FILE__, __LINE__, __VA_ARGS__);

