// This program inspects the indexing methods of SYCL kernels
// through nd_range and nd_item.

#include <taskflow/taskflow.hpp>
#include <taskflow/sycl/syclflow.hpp>

constexpr size_t R = 8;
constexpr size_t C = 12;

void print(int* data, const std::string& message) {
  std::cout << message << '\n';
  for(size_t i=0; i<R; i++) {
    for(size_t j=0; j<C; j++) {
      std::cout << std::setw(5) << data[i*C + j];
    }
    std::cout << '\n';
  }
}

int main() {

  sycl::queue queue;

  auto global_id_r = sycl::malloc_shared<int>(R*C, queue);
  auto global_id_c = sycl::malloc_shared<int>(R*C, queue);
  auto global_linear_id = sycl::malloc_shared<int>(R*C, queue);
  auto local_id_r = sycl::malloc_shared<int>(R*C, queue);
  auto local_id_c = sycl::malloc_shared<int>(R*C, queue);
  auto local_linear_id = sycl::malloc_shared<int>(R*C, queue);
  auto group_id_r = sycl::malloc_shared<int>(R*C, queue);
  auto group_id_c = sycl::malloc_shared<int>(R*C, queue);
  auto group_linear_id = sycl::malloc_shared<int>(R*C, queue);

  queue.submit([=](sycl::handler& handler){
    handler.parallel_for(
      sycl::nd_range<2>{sycl::range<2>(R, C), sycl::range<2>(4, 3)},
      [=](sycl::nd_item<2> item){

        auto r = item.get_global_id(0);
        auto c = item.get_global_id(1);
        auto i = r*C + c;

        // inspect global id
        global_id_r[i] = r;
        global_id_c[i] = c;

        // inspect global linear id
        global_linear_id[i] = item.get_global_linear_id();

        // inspect local id
        local_id_r[i] = item.get_local_id(0);
        local_id_c[i] = item.get_local_id(1);

        // inspect local linear id
        local_linear_id[i] = item.get_local_linear_id();

        // inspect group id
        group_id_r[i] = item.get_group(0);
        group_id_c[i] = item.get_group(1);

        // inspect group linear id
        group_linear_id[i] = item.get_group_linear_id();

      }
    );
  }).wait();

  // print the indices
  print(global_id_r, "global_id_r");
  print(global_id_c, "global_id_c");
  print(global_linear_id, "global_linear_id");
  print(local_id_r, "local_id_r");
  print(local_id_c, "local_id_c");
  print(local_linear_id, "local_linear_id");
  print(group_id_r, "group_id_r");
  print(group_id_c, "group_id_c");
  print(group_linear_id, "group_linear_id");

  return 0;
}




