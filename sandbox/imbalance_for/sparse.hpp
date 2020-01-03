/* 
*   Matrix Market I/O example program
*
*   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file.
*   and copies it to stdout.  This porgram does nothing useful, but
*   illustrates common usage of the Matrix Matrix I/O routines.
*   (See http://math.nist.gov/MatrixMarket for details.)
*
*   Usage:  a.out [filename] > output
*
*       
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/


#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <tuple>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include "mmio.hpp"


inline int M, N, nz;
inline int *I {nullptr};
inline int *J {nullptr};
inline double *val {nullptr};

// CSR storage
inline std::vector<unsigned> RowPtr;
inline std::vector<unsigned> Col;
inline std::vector<double> Val;

// https://math.nist.gov/MatrixMarket/
inline int read_matrix(const char *filename) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int i;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("Could not open file!\n");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0)
  {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
      mm_is_sparse(matcode) )  {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
    printf("Format is not supported\n");
    exit(1);
  }

  /* reseve memory for matrices */

  I = (int *) malloc(nz * sizeof(int));
  J = (int *) malloc(nz * sizeof(int));
  val = (double *) malloc(nz * sizeof(double));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (i=0; i<nz; i++)
  {
    auto ret = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
    if(ret != EOF) assert(ret == 3);
    I[i]--;  /* adjust from 1-based to 0-based */
    J[i]--;
    assert(I[i] >= 0);
    assert(J[i] >= 0);
  }

  fclose(f);

  ///************************/
  ///* now write out matrix */
  ///************************/

  //mm_write_banner(stdout, matcode);
  //mm_write_mtx_crd_size(stdout, M, N, nz);
  //for (i=0; i<nz; i++)
  //  fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

  return 0;
}


// https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
// http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html
// Store the matrix in CSR format 
inline void toCSR() {
  assert(I != nullptr);
  assert(J != nullptr);
  assert(val != nullptr);

  std::vector<std::tuple<unsigned, unsigned, double>> tups;
  for(auto i=0; i<nz; i++) {
    assert(I[i] >= 0);
    assert(J[i] >= 0);
    tups.emplace_back(I[i], J[i], val[i]);
  }

  std::sort(tups.begin(), tups.end(), [](const auto& t1, const auto& t2){
    if(std::get<0>(t1) != std::get<0>(t2)) {
      return std::get<0>(t1) < std::get<0>(t2); 
    }
    return std::get<1>(t1) < std::get<1>(t2); 
  });
 
  RowPtr.resize(M+1, 0);
  Col.resize(nz);
  Val.resize(nz);
  RowPtr[0] = 0;

  unsigned pos {0};
  for(auto [i, j, v]: tups) {
    RowPtr[i+1] ++;
    Col[pos] = j;
    Val[pos] = v;
    pos ++;
  }

  // Uncomment the code below to do prefix sum on the RowPtr array
  // This is required for CSR
  //for(unsigned i=1; i<M+1; i++) {
  //  RowPtr[i] += RowPtr[i-1];
  //}
  //assert(RowPtr.back() == nz);
}

// This computation is not meaningful. It just creates a non-trivial workload in single iteration
inline void compute_one_iteration(unsigned id) {
  auto result {1.0};
  auto num = RowPtr[id]*RowPtr[id] << 3;
  for(unsigned j=0; j<num; j++) {
    result *= Val[id] * sin(Val[id]) * cos(Val[id]);
  }
  result *= Val[id];
  Val[id] = result;
}

inline void clear() {
  if(I != nullptr) free(I);
  if(J != nullptr) free(J); 
  if(val != nullptr) free(val);
}

std::chrono::microseconds measure_time_taskflow(unsigned);
std::chrono::microseconds measure_time_omp(unsigned);
std::chrono::microseconds measure_time_tbb(unsigned);
