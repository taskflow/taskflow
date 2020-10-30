#include <chrono>
#include <unordered_map>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>

int main() {
  
  tf::Taskflow taskflow;
  tf::Executor executor;

  size_t N = 1024;
  float* x = nullptr;
  int* r;
  int res;

  std::vector<float> host(N, 0.0f);
  host[200] = -100.0f;  // artificially set the mid-pos as the largest

  TF_CHECK_CUDA(cudaMalloc(&x, N*sizeof(float)), "failed to malloc x");
  TF_CHECK_CUDA(cudaMalloc(&r, sizeof(int)), "failed to malloc r");

  taskflow.emplace([&](tf::cudaFlow& cf){

    auto h2d = cf.copy(x, host.data(), N).name("h2d");

    auto child = cf.cublas([&](tf::cublasFlow& cbf){  /// cublas
      auto t1 = cbf.amax<float>(N, x, 1, r).name("amax1");  
      auto t2 = cbf.amax<float>(N, x, 1, r).name("amax2");  
      auto t3 = cbf.amax<float>(N, x, 1, r).name("amax3");  
      t2.precede(t1);
      t1.precede(t3);
    }).name("cublas");
    
    auto d2h = cf.copy(&res, r, 1).name("d2h");

    child.succeed(h2d)
         .precede(d2h);
  }).name("cudaflow");

  executor.run(taskflow).wait();

  taskflow.dump(std::cout);

  std::cout << "res: " << res << '\n';
  
  TF_CHECK_CUDA(cudaFree(x), "failed to free x");
  TF_CHECK_CUDA(cudaFree(r), "failed to free r");

  return 0;
}

/*#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (
  cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta
){
  cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
  cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int main (void){

  for(int itr=0; itr<5; itr++) {

    std::cout << "iteration " << itr << '\n';

    cudaError_t cudaStat;
    cublasStatus_t stat;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    auto beg = std::chrono::steady_clock::now();
    auto hptr = tf::cublas_per_thread_handle_pool.acquire(0);
    auto handle = hptr->native_handle;
    auto end = std::chrono::steady_clock::now();

    std::cout << "create handle: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
              << " us\n";

    //int version;
    //cublasGetVersion(handle, &version);
    //std::cout << "version is " << version << '\n';

    beg = std::chrono::steady_clock::now();
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        //cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "set matrix: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
              << " us\n";

    beg = std::chrono::steady_clock::now();
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    end = std::chrono::steady_clock::now();
    std::cout << "modify matrix: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
              << " us\n";

    beg = std::chrono::steady_clock::now();
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        //cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "get matrix: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
              << " us\n";

    cudaFree (devPtrA);

    //beg = std::chrono::steady_clock::now();
    //cublasDestroy(handle);
    //end = std::chrono::steady_clock::now();
    //std::cout << "destroy handle: "
    //          << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
    //          << " us\n";


    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    
    beg = std::chrono::steady_clock::now();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    end = std::chrono::steady_clock::now();
    std::cout << "create stream: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()
              << " us\n";
    
    beg = std::chrono::steady_clock::now();
    cudaStreamDestroy(stream);
    end = std::chrono::steady_clock::now();
    std::cout << "destroy stream: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()
              << " us\n";
    
    beg = std::chrono::steady_clock::now();
    tf::cublas_per_thread_handle_pool.release(std::move(hptr));
    end = std::chrono::steady_clock::now();
    std::cout << "release handle: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()
              << " us\n";
  }
  return EXIT_SUCCESS;
}*/

