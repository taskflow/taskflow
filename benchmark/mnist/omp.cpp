#include "dnn.hpp"
#include <omp.h>

void run_omp(MNIST& D, unsigned num_threads) {

  // Create a task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;
  const auto num_storage = num_threads * 2;

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
           #pragma omp task depend (out: dep_s[e])
           {
             if(e != 0) {
               D.shuffle(mats[e%num_par_shf], vecs[e%num_par_shf], D.images.rows()); 
             }
           }
         }
         else {
           #pragma omp task depend (in: dep_s[e-num_par_shf], dep_b[(1+e-num_par_shf)*iter_num*num_layers-1]) depend (out: dep_s[e])
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
               #pragma omp task depend (in: dep_s[e]) depend (out: dep_f[i])
               {
                 forward_task(D, i, e%num_par_shf, mats, vecs);
               }
             }
             else {
               // use openmp array sections syntax [lower_bound: length]
               #pragma omp task depend (in: dep_u[(i-1)*num_layers: num_layers]) depend (out: dep_f[i])
               {
                 forward_task(D, i, e%num_par_shf, mats, vecs);
               }
             }
           }
           else {
             if(i == 0) {
                #pragma omp task depend (in: dep_s[e], dep_u[e*prop_per_e-num_layers: num_layers]) depend (out: dep_f[e*iter_num+i])
                {
                  forward_task(D, i, e%num_par_shf, mats, vecs);
                }
             }
             else {
                #pragma omp task depend (in: dep_u[e*prop_per_e+(i-1)*num_layers: num_layers]) depend (out: dep_f[e*iter_num+i])
                {
                  forward_task(D, i, e%num_par_shf, mats, vecs);
                }
             }
           }

           // Backward tasks   
           for(int j=num_layers-1; j>=0; j--) {
             if(j == num_layers-1) {
               #pragma omp task depend (in: dep_f[e*iter_num + i]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               {
                 backward_task(D, j, e%num_par_shf, mats);
               }
             }
             else {
               #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j + 1]) depend (out: dep_b[e*prop_per_e + i*num_layers + j])
               {
                 backward_task(D, j, e%num_par_shf, mats);
               }
             }
           }

           // Update tasks   
           for(int j=num_layers-1; j>=0; j--) {
             #pragma omp task depend (in: dep_b[e*prop_per_e + i*num_layers + j]) depend (out: dep_u[e*prop_per_e + i*num_layers + j])          
             {
               D.update(j);
             }
           }

         }
       }
    } // End of omp single 
  } // End of omp parallel

  delete [] dep_s;
  delete [] dep_f;
  delete [] dep_b;
  delete [] dep_u;
}

