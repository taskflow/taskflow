#include <taskflow/taskflow.hpp>  // the only include you need

int main(){

  tf::Executor executor;
  
  // demonstration of dependent async (with future)
  printf("Dependent Async\n");
  auto [A, fuA] = executor.dependent_async([](){ printf("A\n"); });
  auto [B, fuB] = executor.dependent_async([](){ printf("B\n"); }, A);
  auto [C, fuC] = executor.dependent_async([](){ printf("C\n"); }, A);
  auto [D, fuD] = executor.dependent_async([](){ printf("D\n"); }, B, C);

  fuD.get();

  // demonstration of silent dependent async (without future)
  printf("Silent Dependent Async\n");
  A = executor.silent_dependent_async([](){ printf("A\n"); });
  B = executor.silent_dependent_async([](){ printf("B\n"); }, A);
  C = executor.silent_dependent_async([](){ printf("C\n"); }, A);
  D = executor.silent_dependent_async([](){ printf("D\n"); }, B, C);

  executor.wait_for_all();

  return 0;
}




