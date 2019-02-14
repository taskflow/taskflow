// 2018/11/13 - created by Chun-Xun Lin
//
// The program demonstrates how to use asynchronous dispatch
// with callback invoked on finish.

#include <taskflow/taskflow.hpp>
#include <random>

int main() {

  using namespace std::literals::chrono_literals;

  tf::Taskflow tf;

  const size_t num_jobs {10000};

  std::atomic<size_t> counter {0};
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> dice(1, 100);

  // Insert tasks with sleeping for random ms and increasing the counter
  for(size_t i=0; i<num_jobs; ++i){
    tf.emplace([t=dice(gen), &counter](){ 
      std::this_thread::sleep_for(std::chrono::microseconds(t));
      counter++;
    });
  }

  // Register a callback to check the counter 
  auto fu = tf.dispatch([&](){ 
    assert(counter == num_jobs); 
    std::cout << "Finish dispatched tasks\n";
  });

  fu.get(); // Wait for the dispatched tasks

  // Insert tasks with sleeping for random ms and decreasing the counter
  for(size_t i=0; i<num_jobs; ++i){
    tf.emplace([t=dice(gen), &counter](){ 
      std::this_thread::sleep_for(std::chrono::microseconds(t));
      counter--;
    });
  }

  // Register a callback to check the counter 
  tf.silent_dispatch([&counter](){ 
    assert(counter == 0); 
    std::cout << "Finish silently dispatched tasks\n";
  });

  tf.wait_for_all();

  return 0;
}

