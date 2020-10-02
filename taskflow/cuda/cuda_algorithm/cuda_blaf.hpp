#pragma once

#include "../cuda_flow.hpp"
#include "cuda_transpose.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaBLAF definition
// ----------------------------------------------------------------------------

/**
@brief basic linear algebra flow on top of cudaFlow
*/
class cudaBLAF {

  public:

    /**
    @brief constructs a blas builder object

    @param cudaflow a cudaflow object
    */ 
    cudaBLAF(cudaFlow& cudaflow);

    /**
    @brief transposes a two-dimenstional matrix

    @tparam T data type
    @param d_in pointer to the source matrix
    @param d_out pointer to the target matrix
    @param rows number of rows in the source matrix
    @param cols number of columns in the source matrix

    @return cudaTask handle
    */
    template <typename T>
    cudaTask transpose(const T* d_in, T* d_out, size_t rows, size_t cols);
    
    
    //template <typename T>
    //cudaTask matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N);

  private:

    cudaFlow& _cf;
};

// Constructor
inline cudaBLAF::cudaBLAF(cudaFlow& cf) : _cf{cf} {
}

// Function: row-wise matrix transpose
template <typename T>
cudaTask cudaBLAF::transpose(const T* d_in, T* d_out, size_t rows, size_t cols) {

  if(rows == 0 || cols == 0) {
    return _cf.noop();
  }

  size_t grid_dimx = (cols + 31) / 32;
  size_t grid_dimy = (rows + 31) / 32;
  
  return _cf.kernel(
    dim3(grid_dimx, grid_dimy, 1),
    dim3(32, 8, 1),
    0,
    cuda_transpose<T>,
    d_in,
    d_out,
    rows,
    cols
  );

}


}  // end of namespace tf -----------------------------------------------------


