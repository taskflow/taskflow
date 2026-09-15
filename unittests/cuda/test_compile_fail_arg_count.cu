// This file must fail to compile due to the static_assert in the typed kernel() overload.
// Expected error: "kernel: argument count does not match kernel parameter count"
#include <taskflow/cuda/cudaflow.hpp>

__global__ void k_two_args(int* ptr, size_t N) { /* empty */ }

int main() {
    tf::cudaGraph cg;
    // Wrong: k_two_args expects 2 arguments, we pass 3
    cg.kernel({1,1,1}, {1,1,1}, 0, k_two_args, (int*)nullptr, (size_t)0, 42);
    return 0;
}
