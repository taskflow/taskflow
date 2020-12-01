#pragma once

#include "cuda_transpose.hpp"
#include "cuda_matmul.hpp"

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
    
    template <typename T>
    cudaTask matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N);

    // ------------------------------------------------------------------------
    // update APIs
    // ------------------------------------------------------------------------
    template <typename T>
    void update_transpose(cudaTask ct, const T* d_in, T* d_out, size_t rows, size_t cols);
    
    template <typename T>
    void update_matmul(cudaTask ct, const T* A, const T* B, T* C, size_t M, size_t K, size_t N);

  private:

    cudaFlow& _cf;
};

// Constructor
inline cudaBLAF::cudaBLAF(cudaFlow& cf) : _cf{cf} {
}

// Function: row-wise matrix transpose
template <typename T>
cudaTask cudaBLAF::transpose(const T* d_in, T* d_out, size_t rows, size_t cols) {

  //TODO: throw invalid parameters (e.x. grid_dimx = 0)

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

// Function: row-major matrix transpose
template <typename T>
cudaTask cudaBLAF::matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {

  size_t grid_dimx = (N + 31) / 32;
  size_t grid_dimy = (M + 31) / 32;

  //TODO: throw invalid parameters (e.x. grid_dimx = 0)
  return _cf.kernel(
    dim3(grid_dimx, grid_dimy, 1),
    dim3(32, 32, 1),
    0,
    cuda_matmul<T>,
    A,
    B,
    C,
    M,
    K,
    N
  );
}

// ------------------------------------------------------------------------
// update APIs
// ------------------------------------------------------------------------
template <typename T>
void cudaBLAF::update_transpose(cudaTask ct, const T* d_in, T* d_out, size_t rows, size_t cols) {
  size_t grid_dimx = (cols + 31) / 32;
  size_t grid_dimy = (rows + 31) / 32;
  
  
  _cf.update_kernel(
    ct,
    dim3(grid_dimx, grid_dimy, 1),
    dim3(32, 8, 1),
    0,
    d_in,
    d_out,
    rows,
    cols
  );
}

template <typename T>
void cudaBLAF::update_matmul(cudaTask ct, const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
  size_t grid_dimx = (N + 31) / 32;
  size_t grid_dimy = (M + 31) / 32;

  _cf.update_kernel(
    ct,
    dim3(grid_dimx, grid_dimy, 1),
    dim3(32, 32, 1),
    0,
    A,
    B,
    C,
    M,
    K,
    N
  );
}


}  // end of namespace tf -----------------------------------------------------


