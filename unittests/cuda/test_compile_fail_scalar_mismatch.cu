// This file must fail to compile because static_cast<size_t>(incompatible_struct) is ill-formed.
// Expected error: cannot convert struct to size_t
#include <taskflow/cuda/cudaflow.hpp>

struct Incompatible { int x; int y; };

__global__ void k_scalar(size_t N) { /* empty */ }

int main() {
    tf::cudaGraph cg;
    Incompatible s{1, 2};
    // Wrong: k_scalar expects size_t, we pass an incompatible struct
    cg.kernel({1,1,1}, {1,1,1}, 0, k_scalar, s);
    return 0;
}
