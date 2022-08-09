// This program demonstrates how to create a simple vector-add
// application using syclFlow and unified shared memory (USM).

#include <taskflow/sycl/syclflow.hpp>

constexpr size_t N = 10000000;

/*int main() {

  tf::Executor executor;
  tf::Taskflow taskflow;

  sycl::queue queue;

  int* data {nullptr};

  // create an allocate task to allocate a shared memory
  tf::Task allocate = taskflow.emplace(
    [&](){ data = sycl::malloc_shared<int>(N, queue); }
  );

  // create a syclFlow task to add 2 to each element of the vector
  tf::Task syclFlow = taskflow.emplace_on([&](tf::syclFlow& sf){

    tf::syclTask fill = sf.fill(data, 100, N);

    tf::syclTask plus = sf.parallel_for(
      sycl::range<1>(N), [=](sycl::id<1> id) { data[id] += 2; }
    );

    fill.precede(plus);

  }, queue);

  // create a deallocate task that checks the result and frees the memory
  tf::Task deallocate = taskflow.emplace([&](){

    for(size_t i=0; i<N; i++) {
      if(data[i] != 102) {
        std::cout << data << '[' << i << "] = " << data[i] << '\n';
        throw std::runtime_error("incorrect result");
      }
    }
    std::cout << "correct result\n";

    sycl::free(data, queue);
  });

  // create dependencies
  syclFlow.succeed(allocate)
          .precede(deallocate);

  // run the taskflow
  executor.run(taskflow).wait();

  return 0;
} */

int main() {
  constexpr int size = 16;

  std::array<int, size> data;

  sycl::queue Q{}; // Select any device for this queue

  std::cout << "Selected device is: " <<

  Q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  sycl::buffer<float, 1> A{ sycl::range<1>(size) };
  sycl::buffer<float, 1> B{ sycl::range<1>(size) };
  sycl::buffer C{ data };

  Q.submit([&](sycl::handler& h) {
    auto acc = A.get_access<sycl::access::mode::write>(h);
    h.parallel_for(size, [=](auto& idx) {
      acc[idx] = 1000;
    });
  });
  
  Q.submit([&](sycl::handler& h) {
    auto acc = B.get_access<sycl::access::mode::write>(h);
    h.parallel_for(size, [=](auto& idx) {
      acc[idx] = 4000;
    });
  });

  Q.submit([&](sycl::handler& h) {
    auto Aacc = A.get_access<sycl::access::mode::read>(h);
    auto Bacc = B.get_access<sycl::access::mode::read>(h);
    auto Cacc = C.get_access<sycl::access::mode::write>(h);
    h.parallel_for(size , [=](auto&idx){
      Cacc[idx] = Aacc[idx] + Bacc[idx];
    });
  });
    
  sycl::accessor acc = B.get_access<sycl::access::mode::read>();

  for(int i=0; i<size; i++) {
    std::cout << acc[i] << '\n';
  }

  return 0;
}




















