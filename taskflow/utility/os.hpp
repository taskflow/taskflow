#include <cstdlib>
#include <cstdio>
#include <string>

namespace tf {

// Function: get_env
inline std::string get_env(const std::string& str) {
#ifdef _MSC_VER
  char *ptr = nullptr;
  size_t len = 0;
  
  if(_dupenv_s(&ptr, &len, str.c_str()) == 0 && ptr != nullptr) {
    std::string res(ptr, len);
    free(ptr);
    return res;
  }
  return "";

#else
  auto ptr = std::getenv(str.c_str());
  return ptr ? ptr : "";
#endif
}


}  // end of namespace tf -----------------------------------------------------
