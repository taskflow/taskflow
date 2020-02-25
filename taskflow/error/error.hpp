#pragma once

#include <iostream>
#include <sstream>
#include <exception>
#include <system_error>

namespace tf {

// Procedure: stringify
template <typename T>
void ostreamize(std::ostringstream& oss, T&& token) {
  oss << std::forward<T>(token);  
}

// Procedure: stringify
template <typename T, typename... Rest>
void ostreamize(std::ostringstream& oss, T&& token, Rest&&... rest) {
  oss << std::forward<T>(token);
  ostreamize(oss, std::forward<Rest>(rest)...);
}

// Procedure: throw_se
// Throws the system error under a given error code.
template <typename... ArgsT>
//void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&... args) {
void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  ostreamize(oss, std::forward<ArgsT>(args)...);
  //(oss << ... << args);
  throw std::runtime_error(oss.str());
}

}  // ------------------------------------------------------------------------

#define TF_THROW(...) tf::throw_re(__FILE__, __LINE__, __VA_ARGS__);

