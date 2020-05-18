#include <cstdlib>
#include <cstdio>
#include <string>

namespace tf {

// Function: get_env
inline std::string get_env(const std::string& str) {
#ifdef _MSC_VER
  char *ptr;
  size_t len;
  auto err = _dupenv_s(&ptr, &len, str.c_str());
  if ( err ) {
    return "";
  }
  std::string res(ptr);
  free(ptr);
  return res;

#else
  auto ptr = std::getenv(str.c_str());
  return ptr ? ptr : "";
#endif
}


}  // end of namespace tf -----------------------------------------------------
