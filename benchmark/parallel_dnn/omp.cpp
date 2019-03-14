#include "dnn.hpp"
#include <omp.h>


void omp_dnn(MNIST_DNN& D, unsigned num_iteration) {

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

void run_omp(unsigned num_epochs, unsigned num_threads) {

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
      for(auto i=0u; i<num_epochs; i++) {
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



void run_omp(MNIST& D, unsigned num_threads) {
}

/*
void run_omp(MNIST& D, unsigned num_threads) {

  // Create a task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;
  const auto num_storage = num_threads;

  // number of concurrent shuffle tasks
  const auto num_par_shf = std::min(num_storage, D.epoch);
  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  const int num_layers = D.acts.size();

  // Propagation per epoch
  const auto prop_per_e = num_layers*iter_num;

  auto dep_s = new int [D.epoch];
  auto dep_f = new int [D.epoch * iter_num];
  auto dep_b = new int [D.epoch * prop_per_e];
  auto dep_u = new int [D.epoch * prop_per_e];

  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    #pragma omp single
    {
       for(size_t e=0; e<D.epoch; e++) {
         // Shuffle Tasks
         if(e < num_par_shf) {
           #pragma omp task depend (out: dep_s[e]) firstprivate(e, num_par_shf) shared(D, mats, vecs)
           {
             if(e != 0) {
               D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows()); 
             }
           }
         }
         else {
           #pragma omp task depend (in: dep_s[e-num_par_shf], dep_b[(1+e-num_par_shf)*prop_per_e-num_layers]) depend (out: dep_s[e]) firstprivate(e, num_par_shf) shared(D, mats, vecs)
           {
             D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows());
           }
         }

         // DNN operations
         for(size_t i=0; i<iter_num; i++) {
           // Forward tasks
           if(e == 0) {
             if(i == 0) {
               // The first task!!
               #pragma omp task depend (in: dep_s[e]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
               {
                 forward_task(D, i, e%num_par_shf, mats, vecs);
               }
             }
             else {
               // use openmp array sections syntax [lower_bound: length]
               //#pragma omp task depend (in: dep_u[(i-1)*num_layers: num_layers]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
               
               switch(num_layers) {
                 case 2:
                   #pragma omp task depend (in: dep_u[(i-1)*num_layers], dep_u[(i-1)*num_layers+1]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 3:
                   #pragma omp task depend (in: dep_u[(i-1)*num_layers], dep_u[(i-1)*num_layers+1], dep_u[(i-1)*num_layers+2]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 4:
                   #pragma omp task depend (in: dep_u[(i-1)*num_layers], dep_u[(i-1)*num_layers+1], dep_u[(i-1)*num_layers+2], dep_u[(i-1)*num_layers+3]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 5:
                   #pragma omp task depend (in: dep_u[(i-1)*num_layers], dep_u[(i-1)*num_layers+1], dep_u[(i-1)*num_layers+2], dep_u[(i-1)*num_layers+3], dep_u[(i-1)*num_layers+4]) depend (out: dep_f[i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 default: assert(false); break;
               }
             }
           }
           else {
             if(i == 0) {
               switch(num_layers) {
                 case 2: 
                   #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e - num_layers], dep_u[e*prop_per_e - num_layers+1]) depend (out: dep_f[e*iter_num+i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 3: 
                   #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e - num_layers], dep_u[e*prop_per_e - num_layers+1], dep_u[e*prop_per_e - num_layers+2]) depend (out: dep_f[e*iter_num+i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 4: 
                   #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e - num_layers], dep_u[e*prop_per_e - num_layers+1], dep_u[e*prop_per_e - num_layers+2], dep_u[e*prop_per_e - num_layers+3]) depend (out: dep_f[e*iter_num+i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 5: 
                   #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e - num_layers], dep_u[e*prop_per_e - num_layers+1], dep_u[e*prop_per_e - num_layers+2], dep_u[e*prop_per_e - num_layers+3], dep_u[e*prop_per_e - num_layers+4]) depend (out: dep_f[e*iter_num+i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 default: assert(false); break;
               }
                //#pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e - num_layers: num_layers]) depend (out: dep_f[e*iter_num+i]) firstprivate(i, e, num_par_shf) shared(D, mats, vecs)

             }
             else {
               switch(num_layers) {
                 case 2:
                   #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers], dep_u[e*prop_per_e+(i-1)*num_layers+1]) depend (out: dep_f[e*iter_num+i]) firstprivate(i ,e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 3:
                   #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers], dep_u[e*prop_per_e+(i-1)*num_layers+1], dep_u[e*prop_per_e+(i-1)*num_layers+2]) depend (out: dep_f[e*iter_num+i]) firstprivate(i ,e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 4:
                   #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers], dep_u[e*prop_per_e+(i-1)*num_layers+1], dep_u[e*prop_per_e+(i-1)*num_layers+2], dep_u[e*prop_per_e+(i-1)*num_layers+3]) depend (out: dep_f[e*iter_num+i]) firstprivate(i ,e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 case 5:
                   #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers], dep_u[e*prop_per_e+(i-1)*num_layers+1], dep_u[e*prop_per_e+(i-1)*num_layers+2], dep_u[e*prop_per_e+(i-1)*num_layers+3], dep_u[e*prop_per_e+(i-1)*num_layers+4]) depend (out: dep_f[e*iter_num+i]) firstprivate(i ,e, num_par_shf) shared(D, mats, vecs)
                   {
                     forward_task(D, i, e%num_par_shf, mats, vecs);
                   }
                 break;
                 default: assert(false); break;
               }
                //#pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers: num_layers]) depend (out: dep_f[e*iter_num+i]) firstprivate(i ,e, num_par_shf) shared(D, mats, vecs)
             }
           }

           // Backward tasks   
           for(int j=num_layers-1; j>=0; j--) {
             if(j == num_layers-1) {
               #pragma omp task depend (in: dep_f[e*iter_num + i]) depend (out: dep_b[e*prop_per_e + i*num_layers + j]) firstprivate(j, e, num_par_shf) shared(D, mats)
               {
                 backward_task(D, j, e%num_par_shf, mats);
               }
             }
             else {
               #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j + 1]) depend (out: dep_b[e*prop_per_e + i*num_layers + j]) firstprivate(j, e, num_par_shf) shared(D, mats)
               {
                 backward_task(D, j, e%num_par_shf, mats);
               }
             }
           }

           // Update tasks   
           for(int j=num_layers-1; j>=0; j--) {
             #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j]) depend (out: dep_u[e*prop_per_e + i*num_layers + j]) firstprivate(j) shared(D)
             {
               D.update(j);
             }
           }

         } // End of one iteration 
       } // End of one epoch 

    } // End of omp single 
  } // End of omp parallel

  delete [] dep_s;
  delete [] dep_f;
  delete [] dep_b;
  delete [] dep_u;
}
*/

