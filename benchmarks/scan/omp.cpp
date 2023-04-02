#include "scan.hpp"
#include <omp.h>

template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op, unsigned nthreads) {
  int i, chunk;
  std::vector<int> last_value_chunk_array(nthreads+1);

  /* Parallel region begins */
  #pragma omp parallel shared(in, out, chunk) private(i) num_threads(nthreads)
  {
    const int num_threads = omp_get_num_threads();          // To get number of threads in machine
    float chunk_in_float = n/(float)num_threads;
    chunk = ceil(chunk_in_float);                           // Defining chunk values
    const int idthread = omp_get_thread_num();              // Get thread ID

    #pragma omp single
    {
      last_value_chunk_array[0] = 0;
    }

    int operation = 0;
    /* For region begins */
    #pragma omp for schedule(static, chunk) nowait
    for(i=0;i<n;i++)
    {
      if((i % chunk) == 0){
          operation = in[i];                      // Breaking at every chunk
          out[i] = in[i];
      }
      else
      {
          out[i] = op(out[i-1], in[i]);           // Performing the required operation
          operation = op(operation, in[i]);
      }
    }
    /* For region ends */
    last_value_chunk_array[idthread+1] = operation;         // Assigning sums of all chunks in last_chunk_value array

    #pragma omp barrier                                     // Syncing all the threads

    int balance = last_value_chunk_array[1];                // Initialising with index 1 value as for thread 0, result has already been calculated
    if(idthread == 1)
    {
      balance = last_value_chunk_array[1];                // For thread ID 1
    }

    for(int i=2; i<(idthread+1); i++)
      balance = op(balance,last_value_chunk_array[i]);    // Creating balance for every thread

    #pragma omp for schedule(static, chunk)                 // To calculate the sum of all chunks
    for(int i=0; i<n; i++) {
      if(idthread != 0)
      {
          out[i] = op(out[i], balance);                   // For thread IDs other than 0
      }
    }
  }
  /* Parallel region ends */
} // omp_scan


// scan_omp
void scan_omp(size_t nthreads) {
  omp_scan(input.size(), input.data(), output.data(), std::plus<int>{}, nthreads);
}

std::chrono::microseconds measure_time_omp(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  scan_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

