# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>
#include <omp.h>
#include <chrono>

# include "poisson.hpp"


double r8mat_rms(int nx, int ny, double *a_);
void rhs(int nx, int ny, double *f_, int block_size);
void timestamp(void);
double u_exact(double x, double y);
double uxxyy_exact(double x, double y);

/*
Purpose:

MAIN is the main program for POISSON_OPENMP.

Discussion:

POISSON_OPENMP is a program for solving the Poisson problem.

This program uses OpenMP for parallel execution.

The Poisson equation

- DEL^2 U(X,Y) = F(X,Y)

is solved on the unit square [0,1] x [0,1] using a grid of NX by
NX evenly spaced points.  The first and last points in each direction
are boundary points.

The boundary conditions and F are set so that the exact solution is

U(x,y) = sin ( pi * x * y)

so that

- DEL^2 U(x,y) = pi^2 * ( x^2 + y^2) * sin ( pi * x * y)

The Jacobi iteration is repeatedly applied until convergence is detected.

For convenience in writing the discretized equations, we assume that NX = NY.

Licensing:

This code is distributed under the GNU LGPL license.

Modified:

14 December 2011

Author:

John Burkardt
*/

/******************************************************************************/

double run(struct user_parameters* params, unsigned num_threads, std::string model)
{
    int matrix_size = params->matrix_size;
    int block_size = params->blocksize;
    int niter = params->titer;
    //if (niter <= 0) {
    //    niter = 4;
    //    params->titer = niter;
    //}
    double dx;
    double dy;
    double error;
    int ii,i;
    int jj,j;
    int nx = matrix_size;
    int ny = matrix_size;
    double *f_ = static_cast<double*>(malloc(nx * nx * sizeof(double)));
    double (*f)[nx][ny] = (double (*)[nx][ny])f_;
    double *u_ = static_cast<double*>(malloc(nx * nx * sizeof(double)));
    double *unew_ = static_cast<double*>(malloc(nx * ny * sizeof(double)));
    double (*unew)[nx][ny] = (double (*)[nx][ny])unew_;

    /* test if valid */
    if ( (nx % block_size) || (ny % block_size) ) {
      params->succeed = 0;
      fprintf(stderr, "*****ERROR: blocsize must divide NX and NY\n");
      return 0;
    }

    /// INITIALISATION
    dx = 1.0 / (double) (nx - 1);
    dy = 1.0 / (double) (ny - 1);

    // Set the right hand side array F.
    rhs(nx, ny, f_, block_size);

    /*
     Set the initial solution estimate UNEW.
     We are "allowed" to pick up the boundary conditions exactly.
    */
    for (j = 0; j < ny; j+= block_size)
      for (i = 0; i < nx; i+= block_size)
        for (jj=j; jj<j+block_size; ++jj)
          for (ii=i; ii<i+block_size; ++ii) {
            if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1) {
              (*unew)[ii][jj] = (*f)[ii][jj];
            } 
            else {
              (*unew)[ii][jj] = 0.0;
            }
          }

    auto beg = std::chrono::high_resolution_clock::now();
    /// KERNEL INTENSIVE COMPUTATION 
    if(model == "seq") {
      sweep_seq(nx, ny, dx, dy, f_, 0, niter, u_, unew_);
    }
    else if(model == "omp") {
      omp_set_num_threads(num_threads);
      //omp_block_for(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size);
      //omp_block_task(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size);
      //omp_block_task_dep(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size);
      //omp_task(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size);
      omp_task_dep(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size);
    }
    else if(model == "tf") {
      taskflow(nx, ny, dx, dy, f_, 0, niter, u_, unew_, block_size, num_threads);
    }
    else {
      assert(false);
    }

    auto end = std::chrono::high_resolution_clock::now();

    if(params->check) {
      double x;
      double y;
      double *udiff_ = static_cast<double*>(malloc(nx * ny * sizeof(double)));
      double (*udiff)[nx][ny] = (double (*)[nx][ny])udiff_;
      /// CHECK OUTPUT
      // Check for convergence.
      for (j = 0; j < ny; j++) {
        y = (double) (j) / (double) (ny - 1);
        for (i = 0; i < nx; i++) {
          x = (double) (i) / (double) (nx - 1);
          (*udiff)[i][j] = (*unew)[i][j] - u_exact(x, y);
        }
      }
      error = r8mat_rms(nx, ny, udiff_);

      double error1;
      // Set the right hand side array F.
      rhs(nx, ny, f_, block_size);

      /*
       Set the initial solution estimate UNEW.
       We are "allowed" to pick up the boundary conditions exactly.
      */
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            (*unew)[i][j] = (*f)[i][j];
          } 
          else {
            (*unew)[i][j] = 0.0;
          }
        }
      }

      sweep_seq(nx, ny, dx, dy, f_, 0, niter, u_, unew_);

      // Check for convergence.
      for (j = 0; j < ny; j++) {
        y = (double) (j) / (double) (ny - 1);
        for (i = 0; i < nx; i++) {
          x = (double) (i) / (double) (nx - 1);
          (*udiff)[i][j] = (*unew)[i][j] - u_exact(x, y);
        }
      }
      error1 = r8mat_rms(nx, ny, udiff_);
      params->succeed = fabs(error - error1) < 1.0E-6;
      assert(params->succeed);
      free(udiff_);
    }

    free(f_);
    free(u_);
    free(unew_);

    return std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
}

/* R8MAT_RMS returns the RMS norm of a vector stored as a matrix. */
double r8mat_rms(int nx, int ny, double *a_) {
  double (*a)[nx][ny] = (double (*)[nx][ny])a_;
  int i;
  int j;
  double v;

  v = 0.0;

  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      v = v + (*a)[i][j] * (*a)[i][j];
    }
  }
  v = sqrt(v / (double) (nx * ny));

  return v;
}

/* RHS initializes the right hand side "vector". */
void rhs(int nx, int ny, double *f_, int block_size) {
  double (*f)[nx][ny] = (double (*)[nx][ny])f_;
  int i,ii;
  int j,jj;
  double x;
  double y;

  // The "boundary" entries of F store the boundary values of the solution.
  // The "interior" entries of F store the right hand sides of the Poisson equation.

  for (j = 0; j < ny; j+=block_size)
    for (i = 0; i < nx; i+=block_size)
      for (jj=j; jj<j+block_size; ++jj) {
        y = (double) (jj) / (double) (ny - 1);
        for (ii=i; ii<i+block_size; ++ii) {
          x = (double) (ii) / (double) (nx - 1);
          if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1)
            (*f)[ii][jj] = u_exact(x, y);
          else
            (*f)[ii][jj] = - uxxyy_exact(x, y);
        }
      }
}

/* Evaluates the exact solution. */
double u_exact(double x, double y) {
  double pi = 3.141592653589793;
  double value;
  value = sin(pi * x * y);
  return value;
}

/* Evaluates (d/dx d/dx + d/dy d/dy) of the exact solution. */
double uxxyy_exact(double x, double y) {
  double pi = 3.141592653589793;
  double value;

  value = - pi * pi * (x * x + y * y) * sin(pi * x * y);

  return value;
}

