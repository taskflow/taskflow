#include <taskflow/taskflow.hpp>

int main() {

  tf::Taskflow tf(4);

  std::vector<int> items;   // uninitialized
  std::atomic<int> sum;     // uninitialized
  
  // the first dependency graph
  auto A = tf.silent_emplace([&] () { 
    items.resize(1024);
  });

  auto B = tf.silent_emplace([&] () { 
    std::iota(items.begin(), items.end(), 0); 
  });
  
  A.precede(B);

  auto fu1 = tf.dispatch();
    
  // the second dependency graph
  // task C to overlap the exeuction of the first graph
  auto C = tf.silent_emplace([&] () {
    sum.store(0, std::memory_order_relaxed); 
  });
  
  // task D can't start until the first graph completes
  auto D = tf.silent_emplace([&] () {
    fu1.get();
    for(auto item : items) {
      sum.fetch_add(item, std::memory_order_relaxed);
    }
  });

  C.precede(D);

  auto fu2 = tf.dispatch();
  
  // wait on the second dependency graph to finish
  fu2.get();

  assert(sum == (0 + 1023) * 1024 / 2);
  
  return 0;
}






