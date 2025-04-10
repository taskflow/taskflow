#pragma once

#include <memory>
#include <string>

namespace tf {

class LazyString {

  public:

  LazyString() = default;
  
  LazyString(const std::string& str) : 
    _str(str.empty() ? nullptr : std::make_unique<std::string>(str)) {
  }

  LazyString(std::string&& str) : 
    _str(str.empty() ? nullptr : std::make_unique<std::string>(std::move(str))) {
  }

  LazyString(const char* str) : 
    _str((!str || str[0] == '\0') ? nullptr : std::make_unique<std::string>(str)) {
  }

  // Modify the operator to return a const reference
  operator const std::string& () const noexcept {
    static const std::string empty_string;
    return _str ? *_str : empty_string;   
  }

  LazyString& operator = (const std::string& str) {
    if(_str == nullptr) {
      _str = std::make_unique<std::string>(str);
    }
    else {
      *_str = str;
    }
    return *this;
  }

  LazyString& operator = (std::string&& str) {
    if(_str == nullptr) {
      _str = std::make_unique<std::string>(std::move(str));
    }
    else {
      *_str = std::move(str);
    }
    return *this;
  }

  bool empty() const noexcept {
    return !_str || _str->empty();
  }

  size_t size() const noexcept {
    return _str ? _str->size() : 0;
  }

  friend std::ostream& operator<<(std::ostream& os, const LazyString& ls) {
    os << (ls._str ? *ls._str : "");
    return os;
  }

  private:

  std::unique_ptr<std::string> _str;

};



}  // end of namespace tf -------------------------------------------------------------------------
