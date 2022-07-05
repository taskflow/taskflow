#include <iostream>
#include <functional>

int main() {

  // user's perspective
  auto lambda = [] ( const int& )  {  };
  
  // your perspective - DataPipeline
  using C = decltype(lambda);
  using T = int;

  //make_datapipe<int&, std::string&> ==> make_datapipe<int, std::string>, here we always decay for storing the data
  static_assert(std::is_invocable_v<C, T&>, "");  // here, we always call user's callable passing the data by reference
}