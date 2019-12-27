#include "poisson.hpp"
#include <taskflow/taskflow.hpp>  
#include <vector>

/* #pragma omp task depend version of SWEEP. */
void taskflow(int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_, int block_size, unsigned num_threads)
{
    int i;
    double* f = f_;
    double* u = u_;
    double* unew = unew_;

    tf::Taskflow flow;

    std::vector<tf::Task> tasks;
    for(i=0; i<nx; i++) {
      tasks.emplace_back(flow.emplace(
        [&, i]() {
          for (int j = 0; j < ny; j++) {
            u[i*ny + j] = unew[i*ny + j];
          }
        }
      ));
    }

    for(i=0; i<nx; i++) {
      auto t = flow.emplace(
        [&, i]() {
          for (int j = 0; j < ny; j++) {
            if(i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
              unew[i*ny + j] = f[i*ny + j];
            } 
            else {
              unew[i*ny + j] = 0.25 * (u[(i-1)*ny + j] + u[i* ny + j + 1] + 
                                       u[(i)*ny + j-1] + u[(i+1)* ny + j] +
                                       f[i*ny + j] * dx * dy);
            }
          }
        }
      );

      tasks[i].precede(t);
      if(i > 0) {
        tasks[i-1].precede(t);
      }
      if(i < nx-1) {
        tasks[i+1].precede(t);
      }
    }

    tf::Executor executor {num_threads};
    executor.run_n(flow, itnew).wait();
}
