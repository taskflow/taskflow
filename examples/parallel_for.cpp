// This program demonstrates loop-based parallelism using:
//   + STL-styled iterators
//   + plain integral indices

#include <taskflow/taskflow.hpp>

// Procedure: for_each
void for_each(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> range(N);
  std::iota(range.begin(), range.end(), 0);

  taskflow.for_each(range.begin(), range.end(), [&] (int i) { 
    printf("for_each on container item: %d\n", i);
  });

  executor.run(taskflow).get();
}

// Procedure: for_each_index
void for_each_index(int N) {
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  // [0, N) with step size 2
  taskflow.for_each_index(0, N, 2, [] (int i) {
    printf("for_each_index on index: %d\n", i);
  });

  executor.run(taskflow).get();
}


void for_each_index_nested() {
	auto N = 10;

	if (N < 0)
	{
		throw std::runtime_error("N must be non-negative");
	}

	int res;  // result

	tf::Executor executor;
	tf::Taskflow taskflow("nestedfor");

	std::mutex mtx;

	taskflow.for_each_index_nested(0,5,1,[&mtx](int i, tf::Subflow&sfi){
		sfi.for_each_index(25,30,1,[i,&mtx](int j){
			std::scoped_lock lock(mtx);
			std::cout << "foreach_index_nested " << i << " "  << j << std::endl;
		});
	});

	executor.run(taskflow).wait();
	//taskflow.dump(std::cout);


}

// ----------------------------------------------------------------------------

// Function: main
int main() {
  
  for_each(100);
  for_each_index(100);
  for_each_index_nested();
  

  return 0;
}






