#include "dnn.hpp"
#include <omp.h>


void omp_dnn(MNIST_DNN& D, const unsigned num_iteration) {

  const int num_layers = D.num_layers();

  auto dep_f = new int [num_iteration];
  auto dep_b = new int [num_iteration*num_layers];
  auto dep_u = new int [num_iteration*num_layers];

  #pragma omp parallel
  {
    #pragma omp single
    {
      for(auto i=0u; i<num_iteration; i++) {
        // Forward Task 
        if(i == 0) {
          #pragma omp task depend (out: dep_f[i]) shared(D, IMAGES, LABELS)
          {
            forward_task(D, IMAGES, LABELS);
          }
        }
        else {
          switch(num_layers) {
            case 2:
              #pragma omp task depend (in: dep_u[(i-1)*num_iteration], dep_u[(i-1)*num_iteration+1]) depend (out: dep_f[i]) shared(D, IMAGES, LABELS)
              {
                forward_task(D, IMAGES, LABELS);
              }
            break;
            case 3:
              #pragma omp task depend (in: dep_u[(i-1)*num_iteration], dep_u[(i-1)*num_iteration+1], dep_u[(i-1)*num_iteration+2]) depend (out: dep_f[i]) shared(D, IMAGES, LABELS)
              {
                forward_task(D, IMAGES, LABELS);
              }
            break;
            case 4:
              #pragma omp task depend (in: dep_u[(i-1)*num_iteration], dep_u[(i-1)*num_iteration+1], dep_u[(i-1)*num_iteration+2], dep_u[(i-1)*num_iteration+3]) depend (out: dep_f[i]) shared(D, IMAGES, LABELS)
              {
                forward_task(D, IMAGES, LABELS);
              }
            break;
            case 5:
              #pragma omp task depend (in: dep_u[(i-1)*num_iteration], dep_u[(i-1)*num_iteration+1], dep_u[(i-1)*num_iteration+2], dep_u[(i-1)*num_iteration+3], dep_u[(i-1)*num_iteration+4]) depend (out: dep_f[i]) shared(D, IMAGES, LABELS)
              {
                forward_task(D, IMAGES, LABELS);
              }
            break;
            default: assert(false); break;
          }
        } // End of Forward Task 

        // Backward tasks   
        for(int j=num_layers-1; j>=0; j--) {
          if(j == num_layers-1) {
            #pragma omp task depend (in: dep_f[i]) depend (out: dep_b[i*num_layers + j]) firstprivate(j) shared(D, IMAGES)
            {
              backward_task(D, j, IMAGES);
            }
          }
          else {
            #pragma omp task depend (in: dep_b[i*num_layers + j + 1]) depend (out: dep_b[i*num_layers + j]) firstprivate(j) shared(D, IMAGES)
            {
              backward_task(D, j, IMAGES);
            }
          }
        }

        // Update tasks   
        for(int j=num_layers-1; j>=0; j--) {
          #pragma omp task depend (in: dep_b[i*num_layers + j]) depend (out: dep_u[i*num_layers + j]) firstprivate(j) shared(D)
          {
            D.update(j);
          }
        }

        #pragma omp taskwait 
      } // End of one iteration 
    }
  } // End of omp parallel


  delete [] dep_f;
  delete [] dep_b;
  delete [] dep_u;
}

void run_omp(const unsigned num_iterations, const unsigned num_threads) {

  auto dnns = std::make_unique<MNIST_DNN[]>(NUM_DNNS);
  for(auto i=0u; i<NUM_DNNS; i++) {
    init_dnn(dnns[i]); 
  }

  omp_set_num_threads(num_threads); 

  //auto t1 = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    #pragma omp single
    {
      for(auto i=0u; i<num_iterations; i++) {
        for(auto j=0u; j<NUM_DNNS; j++) {
          #pragma omp task firstprivate(j) shared(dnns) 
          {
            omp_dnn(dnns[j], NUM_ITERATIONS);
          }
        }
        #pragma omp taskwait 

        for(auto j=0u; j<NUM_DNNS; j++) {
          //std::cout << "Validate " << j << "th NN: ";
          dnns[j].validate(TEST_IMAGES, TEST_LABELS);
        }
        shuffle(IMAGES, LABELS);       
        //report_runtime(t1);
      }
    }
  }

}


