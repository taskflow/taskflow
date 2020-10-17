#include <chrono>
#include <unordered_map>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cublas.hpp>

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
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
    auto handle = tf::cublas_per_thread_handle(0);
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
    
    std::function<void()> test =[](){ int dev; cudaGetDevice(&dev);};

    beg = std::chrono::steady_clock::now();
    //int dev;
    //cudaStreamDestroy(stream);
    //cudaGetDevice(&dev);
    test();
    end = std::chrono::steady_clock::now();
    std::cout << "destroy stream: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()
              << " us\n";
  }
  return EXIT_SUCCESS;
}

