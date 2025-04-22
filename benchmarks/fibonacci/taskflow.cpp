#include <taskflow/taskflow.hpp>
#include "fibonacci.hpp"




tf::Executor& get_executor() {
  static tf::Executor executor;
  return executor;
}



// fibonacci computing : implementation 1
size_t spawn_async1(size_t num_fibonacci) {

  if (num_fibonacci < 2) {
    return num_fibonacci;
  }

  tf::Executor executor;
  size_t res1, res2;

  executor.silent_async([num_fibonacci, &res1](){
    res1 = spawn_async1(num_fibonacci-1); 
  });

  //executor.silent_async([num_fibonacci, &res2](){
  //  res2 = spawn_async1(num_fibonacci-2); 
  //});
  res2 = spawn_async1(num_fibonacci-2);

  executor.wait_for_all();

  return res1 + res2;
}



// fibonacci computation : implementation 2
size_t spawn_async2(size_t num_fibonacci, tf::Runtime& rt) {

  if (num_fibonacci < 2) {
    return num_fibonacci; 
  }
  
  size_t res1, res2;

  rt.silent_async([num_fibonacci, &res1](tf::Runtime& rt1){
    res1 = spawn_async2(num_fibonacci-1, rt1);
  });

  rt.silent_async([num_fibonacci, &res2](tf::Runtime& rt2){
    res2 = spawn_async2(num_fibonacci-2, rt2);
  });

  // use corun to avoid blocking the worker from waiting the two children tasks to finish
  rt.corun();

  return res1 + res2;
}


// fibonacci computation : implementation 3
size_t spawn_async3(size_t num_fibonacci) {
  
  tf::Executor executor;

  std::function<int(int)> fib;

  fib = [&](int num_fibonacci) -> int {

    if (num_fibonacci < 2) {
      return num_fibonacci; 
    }

    std::future<int> fu1, fu2;
    tf::AsyncTask t1, t2;

    std::tie(t1, fu1) = executor.dependent_async([=, &fib](){
      return fib(num_fibonacci-1);
    });
    std::tie(t2, fu2) = executor.dependent_async([=, &fib](){
      return fib(num_fibonacci-2);
    });

    executor.corun_until([&](){ return t1.is_done() && t2.is_done(); });

    return fu1.get() + fu2.get();
  };

  auto [tn, fun] = executor.dependent_async([=, &fib]() { return fib(num_fibonacci); });
  return fun.get();
}



size_t fibonacci_taskflow(size_t num_fibonacci) {
  size_t res;

  
  //get_executor().async([num_fibonacci, &res](){
  //  res = spawn_async1(num_fibonacci);
  //}).get();

  //get_executor().async([num_fibonacci, &res](tf::Runtime& rt){
  //  res = spawn_async2(num_fibonacci, rt);
  //}).get();
 
  res = spawn_async3(num_fibonacci);

  return res;
}





std::chrono::microseconds measure_time_taskflow(unsigned num_threads, unsigned num_fibonacci) {
  auto beg = std::chrono::high_resolution_clock::now();
  auto result = fibonacci_taskflow(num_fibonacci);
  auto end = std::chrono::high_resolution_clock::now();

  assert(result == fibonacci_sequence[num_fibonacci]);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


