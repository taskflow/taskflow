#include <taskflow/taskflow.hpp>

int main() {
  
  tf::Taskflow tf(4);

  std::vector<int> items {1, 2, 3, 4, 5, 6, 7, 8};
  
  auto [S, T] = tf.parallel_for(items.begin(), items.end(), [] (int item) {
    std::cout << std::this_thread::get_id() << " runs " << item << std::endl;
  });

  S.work([] () { std::cout << "S\n"; }).name("S");
  T.work([] () { std::cout << "T\n"; }).name("T");

  tf.wait_for_all(); 

  return 0;
}




