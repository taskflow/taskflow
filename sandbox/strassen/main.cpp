/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

/***********************************************************************
 * main function & common behaviour of the benchmark.
 **********************************************************************/
#include <CLI11.hpp>
#include "strassen.hpp"

void strassen_alg(
  const std::string& model,
  const unsigned num_threads, 
  const unsigned num_rounds
  ) {

  std::cout << std::setw(12) << "runtime"
            << std::endl;
  
  double runtime {0.0};

  for(unsigned j=0; j<num_rounds; ++j) {
    if(model == "tf") {
      runtime += measure_time_taskflow(num_threads, MatrixA, MatrixB, MatrixC, MATRIX_SIZE).count();
    }
    else if(model == "tbb") {
      runtime += measure_time_tbb(num_threads, MatrixA, MatrixB, MatrixC, MATRIX_SIZE).count();
    }
    else if(model == "omp") {
      runtime += measure_time_omp(num_threads, MatrixA, MatrixB, MatrixC, MATRIX_SIZE).count();
    }
    else assert(false);
  }

  std::cout << std::setw(12) << runtime / num_rounds / 1e3
            << std::endl;
}




/***********************************************************************
 * main: 
 **********************************************************************/
int main(int argc, char* argv[]) {
  CLI::App app{"Strassen algorithm for matrix multiplication"};

  unsigned num_threads {1}; 
  app.add_option("-t,--num_threads", num_threads, "number of threads (default=1)");

  unsigned num_rounds {1};  
  app.add_option("-r,--num_rounds", num_rounds, "number of rounds (default=1)");
  
  bool check_result {false};
  app.add_option("-c,--check", check_result, "compare result with sequential mode (default=false)");

  std::string model = "tf";
  app.add_option("-m,--model", model, "model name tbb|omp|tf (default=tf)")
     ->check([] (const std::string& m) {
        if(m != "tbb" && m != "tf" && m != "omp") {
          return "model name should be \"tbb\", \"omp\", or \"tf\"";
        }
        return "";
     });

  CLI11_PARSE(app, argc, argv);
   
  std::cout << "model=" << model << ' '
            << "num_threads=" << num_threads << ' '
            << "num_rounds=" << num_rounds << ' '
            << "check result=" << check_result << ' '
            << std::endl;


  init_ABC();

  strassen_alg(model, num_threads, num_rounds);

  if(check_result) {
    // Compare against the sequential matrix multiplication
    double *D;
    D = alloc_matrix(MATRIX_SIZE);
    auto beg = std::chrono::high_resolution_clock::now();
    //OptimizedStrassenMultiply_seq(D, MatrixA, MatrixB, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1);    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Seq: " << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()/1e3 << std::endl;
    matrixmul(MATRIX_SIZE, MatrixA, MATRIX_SIZE, MatrixB, MATRIX_SIZE, D, MATRIX_SIZE);
    assert(compare_matrix(MATRIX_SIZE, MatrixC, MATRIX_SIZE, D, MATRIX_SIZE) == 1);
    std::cout << "Correct!\n";
    free(D);
  }

  free(MatrixA);
  free(MatrixB);
  free(MatrixC);

  return 0;
}

