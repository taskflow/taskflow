#include <iostream>
#include <functional>
#include <stdio.h>


struct alignas(64) aligned_int
{
  int val;
  aligned_int() {val = 0;}
};

int main() {

  // // user's perspective
  // auto lambda = [] ( const int& )  {  };
  
  // // your perspective - DataPipeline
  // using C = decltype(lambda);
  // using T = int;

  // //make_datapipe<int&, std::string&> ==> make_datapipe<int, std::string>, here we always decay for storing the data
  // static_assert(std::is_invocable_v<C, T&>, "");  // here, we always call user's callable passing the data by reference
  

  aligned_int a[4];
  for (int i = 0; i < 4; i++) {
    printf("%p\n", &a[i]);
  }

}