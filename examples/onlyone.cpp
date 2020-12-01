// A simple example with a semaphore constraint: only one task can
// execute at a time.

#include <taskflow/taskflow.hpp>

#include <iostream>
#include <chrono>
#include <thread>

void sl() {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(1s);
}

int main(){
  tf::Executor executor(4);
  tf::Taskflow tf;
  tf::Semaphore s = tf.semaphore(1);

  std::vector<tf::Task> tasks;
  tasks.push_back(tf.emplace([](){ sl(); std::cout << "A" << std::endl; }));
  tasks.push_back(tf.emplace([](){ sl(); std::cout << "B" << std::endl; }));
  tasks.push_back(tf.emplace([](){ sl(); std::cout << "C" << std::endl; }));
  tasks.push_back(tf.emplace([](){ sl(); std::cout << "D" << std::endl; }));
  tasks.push_back(tf.emplace([](){ sl(); std::cout << "E" << std::endl; }));
  for(auto & t : tasks) {
    t.acquire(s);
    t.release(s);
  }

  executor.run(tf).wait();
}
