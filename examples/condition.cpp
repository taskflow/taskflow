// The example creates the following cyclic graph of one iterative loop:
//
//       A
//       |
//       v
//       B<---|
//       |    |
//       v    |
//       C----|
//       |
//       v
//       D
//
// - A is a task that initializes a counter to zero
// - B is a task that increments the counter
// - C is a condition task that loops with B until the counter
//   reaches a breaking number
// - D is a task that finalizes the result

#include <taskflow/taskflow.hpp>

int main() {

  tf::Executor executor;
  tf::Taskflow taskflow("Conditional Tasking Demo");

  int counter;

  auto A = taskflow.emplace([&](){
    std::cout << "initializes the counter to zero\n";
    counter = 0;
  }).name("A");

  auto B = taskflow.emplace([&](){
    std::cout << "loops to increment the counter\n";
    counter++;
  }).name("B");

  auto C = taskflow.emplace([&](){
    std::cout << "counter is " << counter << " -> ";
    if(counter != 5) {
      std::cout << "loops again (goes to B)\n";
      return 0;
    }
    std::cout << "breaks the loop (goes to D)\n";
    return 1;
  }).name("C");

  auto D = taskflow.emplace([&](){
    std::cout << "done with counter equal to " << counter << '\n';
  }).name("D");

  A.precede(B);
  B.precede(C);
  C.precede(B);
  C.precede(D);

  // visualizes the taskflow
  taskflow.dump(std::cout);

  // executes the taskflow
  executor.run(taskflow).wait();

  assert(counter == 5);

  return 0;
}







