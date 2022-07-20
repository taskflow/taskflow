#include <iostream>
#include <functional>
#include <stdio.h>
#include <variant>

#define CACHELINE_SIZE 64


constexpr size_t CLPAD(size_t _objSize) {
  return ((_objSize / CACHELINE_SIZE) * CACHELINE_SIZE) +
      (((_objSize % CACHELINE_SIZE) > 0) * CACHELINE_SIZE) -
      _objSize;
}

template<class T, bool = false>
struct padded
{
    using type = struct
    {
        alignas(CACHELINE_SIZE)T myObj;
        char padding[CLPAD(sizeof(T))];
    };
};

template<class T>
struct padded<T, true>
{
    using type = struct
    {
        alignas(CACHELINE_SIZE)T myObj;
    };
};

template<class T>
using padded_t = typename padded<T, (sizeof(T) % CACHELINE_SIZE == 0)>::type;

struct myStruct {
  double a[10];
};

int main() {
  using variant_t = std::variant<int, float, double, myStruct>;
  alignas (CACHELINE_SIZE) std::vector<padded_t<variant_t> > _buffer(10);
  for (int i = 0; i < 5; i++) {
    std::cout << "addr" << i << "=" << static_cast<void*>(&_buffer[i]) << std::endl;
  }
  std::cout << "sizeof(string)=" << sizeof(std::string) << std::endl;
  std::cout << "sizeof(variant_t)=" << sizeof(variant_t) << std::endl;
  std::cout << "alignof(variant_t)=" << alignof(variant_t) << std::endl;
  std::cout << "sizeof(padded_t<variant_t>)=" << sizeof(padded_t<variant_t>) << std::endl;
  std::cout << "alignof(padded_t<variant_t>)=" << alignof(padded_t<variant_t>) << std::endl;
  // std::cout << "sizeof(_buffer)=" << sizeof(_buffer) << std::endl;
  // std::cout << "alignof(_buffer)=" << alignof(_buffer) << std::endl;
}