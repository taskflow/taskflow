// This example shows how to create task using either 'emplace'
// or 'silent_emplace'.

#include <cassert>
#include <taskflow/taskflow.hpp>

int main(){

  tf::Taskflow tf;
  
  // The method emplace gives you a future to retrieve the result.
  // You may pass this future to another thread or program context
  // to enable more asynchronous control flows.
  auto [A, fuA] = tf.emplace([] () {
    std::cout << "Task A\n";
    return 1;
  });

  A.name("A");
  
  // The method silent_emplace won't give you a future. 
  // Therefore, you won't be able to get the return value of the callable.
  // Typically, silent_emplace is used for callables with no return values.
  auto B = tf.silent_emplace([] () {
    std::cout << "Task B\n";
    return "no one can get me";
  });

  B.name("B");
  
  // The future of A won't be ready until you execute the taskflow.
  tf.wait_for_all();

  assert(fuA.get() == 1);

  return 0;
}
