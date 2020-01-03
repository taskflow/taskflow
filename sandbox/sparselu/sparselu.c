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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <libgen.h>
//#include "bots.h"
#include "sparselu.h"
#include <omp.h>
#include <taskflow/taskflow.hpp>


#define FALSE 0
#define TRUE  1

/***********************************************************************
 * checkmat: 
 **********************************************************************/
int checkmat (float *M, float *N)
{
   int i, j;
   float r_err;

   for (i = 0; i < submatrix_size; i++) 
   {
      for (j = 0; j < submatrix_size; j++) 
      {
         r_err = M[i*submatrix_size+j] - N[i*submatrix_size+j];
         if ( r_err == 0.0 ) continue;

         if (r_err < 0.0 ) r_err = -r_err;

         if ( M[i*submatrix_size+j] == 0 )
         {
           printf("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n",
                    i,j, M[i*submatrix_size+j], i,j, N[i*submatrix_size+j]);
           return FALSE;
         }
         r_err = r_err / M[i*submatrix_size+j];
         if(r_err > EPSILON)
         {
            printf("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
                    i,j, M[i*submatrix_size+j], i,j, N[i*submatrix_size+j], r_err);
            return FALSE;
         }
      }
   }
   return TRUE;
}
/***********************************************************************
 * genmat: 
 **********************************************************************/
void genmat (float *M[])
{
   int null_entry, init_val, i, j, ii, jj;
   float *p;
   int a=0,b=0;

   init_val = 1325;

   /* generating the structure */
   for (ii=0; ii < matrix_size; ii++)
   {
      for (jj=0; jj < matrix_size; jj++)
      {
         /* computing null entries */
         null_entry=FALSE;
         if ((ii<jj) && (ii%3 !=0)) null_entry = TRUE;
         if ((ii>jj) && (jj%3 !=0)) null_entry = TRUE;
	 if (ii%2==1) null_entry = TRUE;
	 if (jj%2==1) null_entry = TRUE;
	 if (ii==jj) null_entry = FALSE;
	 if (ii==jj-1) null_entry = FALSE;
         if (ii-1 == jj) null_entry = FALSE; 
         /* allocating matrix */
         if (null_entry == FALSE){
            a++;
            M[ii*matrix_size+jj] = (float *) malloc(submatrix_size*submatrix_size*sizeof(float));
	    if (M[ii*matrix_size+jj] == NULL)
            {
               printf("Error: Out of memory\n");
               exit(101);
            }
            /* initializing matrix */
            p = M[ii*matrix_size+jj];
            for (i = 0; i < submatrix_size; i++) 
            {
               for (j = 0; j < submatrix_size; j++)
               {
	            init_val = (3125 * init_val) % 65536;
      	            (*p) = (float)((init_val - 32768.0) / 16384.0);
                    p++;
               }
            }
         }
         else
         {
            b++;
            M[ii*matrix_size+jj] = NULL;
         }
      }
   }
   //bots_debug("allo = %d, no = %d, total = %d, factor = %f\n",a,b,a+b,(float)((float)a/(float)(a+b)));
}
/***********************************************************************
 * print_structure: 
 **********************************************************************/
void print_structure(char *name, float *M[])
{
   int ii, jj;
   printf("Structure for matrix %s @ 0x%p\n",name, M);
   for (ii = 0; ii < matrix_size; ii++) {
     for (jj = 0; jj < matrix_size; jj++) {
        if (M[ii*matrix_size+jj]!=NULL) {printf("x");}
        else printf(" ");
     }
     printf("\n");
   }
   printf("\n");
}
/***********************************************************************
 * allocate_clean_block: 
 **********************************************************************/
float * allocate_clean_block()
{
  int i,j;
  float *p, *q;

  p = (float *) malloc(submatrix_size*submatrix_size*sizeof(float));
  q=p;
  if (p!=NULL){
     for (i = 0; i < submatrix_size; i++) 
        for (j = 0; j < submatrix_size; j++){(*p)=0.0; p++;}
	
  }
  else
  {
      printf("Error: Out of memory\n");
      exit (101);
  }
  return (q);
}

/***********************************************************************
 * lu0: 
 **********************************************************************/
void lu0(float *diag)
{
   int i, j, k;

   for (k=0; k<submatrix_size; k++)
      for (i=k+1; i<submatrix_size; i++)
      {
         diag[i*submatrix_size+k] = diag[i*submatrix_size+k] / diag[k*submatrix_size+k];
         for (j=k+1; j<submatrix_size; j++)
            diag[i*submatrix_size+j] = diag[i*submatrix_size+j] - diag[i*submatrix_size+k] * diag[k*submatrix_size+j];
      }
}

/***********************************************************************
 * bdiv: 
 **********************************************************************/
void bdiv(float *diag, float *row)
{
   int i, j, k;
   for (i=0; i<submatrix_size; i++)
      for (k=0; k<submatrix_size; k++)
      {
         row[i*submatrix_size+k] = row[i*submatrix_size+k] / diag[k*submatrix_size+k];
         for (j=k+1; j<submatrix_size; j++)
            row[i*submatrix_size+j] = row[i*submatrix_size+j] - row[i*submatrix_size+k]*diag[k*submatrix_size+j];
      }
}
/***********************************************************************
 * bmod: 
 **********************************************************************/
void bmod(float *row, float *col, float *inner)
{
   int i, j, k;
   for (i=0; i<submatrix_size; i++)
      for (j=0; j<submatrix_size; j++)
         for (k=0; k<submatrix_size; k++)
            inner[i*submatrix_size+j] = inner[i*submatrix_size+j] - row[i*submatrix_size+k]*col[k*submatrix_size+j];
}
/***********************************************************************
 * fwd: 
 **********************************************************************/
void fwd(float *diag, float *col)
{
   int i, j, k;
   for (j=0; j<submatrix_size; j++)
      for (k=0; k<submatrix_size; k++) 
         for (i=k+1; i<submatrix_size; i++)
            col[i*submatrix_size+j] = col[i*submatrix_size+j] - diag[i*submatrix_size+k]*col[k*submatrix_size+j];
}

void sparselu_init (float ***pBENCH, char *pass)
{
   *pBENCH = (float **) malloc(matrix_size*matrix_size*sizeof(float *));
   genmat(*pBENCH);
   //print_structure(pass, *pBENCH);
}


void sparselu_seq_call(float **BENCH)
{
   int ii, jj, kk;

   for (kk=0; kk<matrix_size; kk++)
   {
      lu0(BENCH[kk*matrix_size+kk]);
      for (jj=kk+1; jj<matrix_size; jj++)
         if (BENCH[kk*matrix_size+jj] != NULL)
         {
            fwd(BENCH[kk*matrix_size+kk], BENCH[kk*matrix_size+jj]);
         }
      for (ii=kk+1; ii<matrix_size; ii++) 
         if (BENCH[ii*matrix_size+kk] != NULL)
         {
            bdiv (BENCH[kk*matrix_size+kk], BENCH[ii*matrix_size+kk]);
         }
      for (ii=kk+1; ii<matrix_size; ii++)
         if (BENCH[ii*matrix_size+kk] != NULL)
            for (jj=kk+1; jj<matrix_size; jj++)
               if (BENCH[kk*matrix_size+jj] != NULL)
               {
                     if (BENCH[ii*matrix_size+jj]==NULL) BENCH[ii*matrix_size+jj] = allocate_clean_block();
                     bmod(BENCH[ii*matrix_size+kk], BENCH[kk*matrix_size+jj], BENCH[ii*matrix_size+jj]);
               }

   }
}

void sparselu_par_call(float **BENCH)
{
  omp_set_num_threads(14);

   int ii, jj, kk;
   
   printf("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
           matrix_size,matrix_size,submatrix_size,submatrix_size);
#pragma omp parallel private(kk)
   {
   for (kk=0; kk<matrix_size; kk++) 
   {
#pragma omp single
      lu0(BENCH[kk*matrix_size+kk]);

#pragma omp for nowait
      for (jj=kk+1; jj<matrix_size; jj++)
         if (BENCH[kk*matrix_size+jj] != NULL)
            #pragma omp task untied firstprivate(kk, jj) shared(BENCH)
         {
            fwd(BENCH[kk*matrix_size+kk], BENCH[kk*matrix_size+jj]);
         }
#pragma omp for
      for (ii=kk+1; ii<matrix_size; ii++) 
         if (BENCH[ii*matrix_size+kk] != NULL)
            #pragma omp task untied firstprivate(kk, ii) shared(BENCH)
         {
            bdiv (BENCH[kk*matrix_size+kk], BENCH[ii*matrix_size+kk]);
         }

#pragma omp for private(jj)
      for (ii=kk+1; ii<matrix_size; ii++)
         if (BENCH[ii*matrix_size+kk] != NULL)
            for (jj=kk+1; jj<matrix_size; jj++)
               if (BENCH[kk*matrix_size+jj] != NULL)
               #pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
               {
                     if (BENCH[ii*matrix_size+jj]==NULL) BENCH[ii*matrix_size+jj] = allocate_clean_block();
                     bmod(BENCH[ii*matrix_size+kk], BENCH[kk*matrix_size+jj], BENCH[ii*matrix_size+jj]);
               }

   }
   }
   printf(" completed!\n");
}

void sparselu_fini (float **BENCH, char *pass) {
   print_structure(pass, BENCH);
}

int sparselu_check(float **SEQ, float **BENCH)
{
   int ii,jj,ok=1;

   for (ii=0; ((ii<matrix_size) && ok); ii++)
   {
      for (jj=0; ((jj<matrix_size) && ok); jj++)
      {
         if ((SEQ[ii*matrix_size+jj] == NULL) && (BENCH[ii*matrix_size+jj] != NULL)) ok = FALSE;
         if ((SEQ[ii*matrix_size+jj] != NULL) && (BENCH[ii*matrix_size+jj] == NULL)) ok = FALSE;
         if ((SEQ[ii*matrix_size+jj] != NULL) && (BENCH[ii*matrix_size+jj] != NULL))
            ok = checkmat(SEQ[ii*matrix_size+jj], BENCH[ii*matrix_size+jj]);
      }
   }
   assert(ok);
   if (ok) return 0; // Success
   else return 1;    // Fail
}



void sparselu_tf_for(float **BENCH)
{

  tf::Taskflow flow;

  const unsigned num_threads = 14;
   
   printf("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
           matrix_size,matrix_size,submatrix_size,submatrix_size);

  int kk = -1;
  auto t1 = flow.emplace([&](){
    kk ++;
    lu0(BENCH[kk*matrix_size+kk]);
  });


  std::vector<tf::Task> tasks;
  for(int i=0; i<num_threads; i++) {
    tasks.emplace_back(
      flow.emplace( [&, id=i] (auto &subflow) {
        int part = (matrix_size - kk - 1 + num_threads - 1)/num_threads;
        int beg = part*id + kk + 1;
        int end = beg + part < matrix_size ? beg+part : matrix_size; 
        for(int jj=beg; jj<end; jj++) {
          if (BENCH[kk*matrix_size+jj] != NULL) {
            subflow.emplace([&, jj](){ fwd(BENCH[kk*matrix_size+kk], BENCH[kk*matrix_size+jj]); });
          }
        }
      })
    );
  }
 
  t1.precede(tasks);
  auto sync1 = flow.emplace([](){});
  for(auto &t: tasks) {
    t.precede(sync1);
  }

  tasks.clear();
  for(int i=0; i<num_threads; i++) {
    tasks.emplace_back(
      flow.emplace([&, id=i](auto &subflow){
        int part = (matrix_size - kk - 1 + num_threads - 1)/num_threads;
        int beg = part*id + kk + 1;
        int end = beg + part < matrix_size ? beg+part : matrix_size; 
        for(int ii=beg; ii<end; ii++) {
          if (BENCH[ii*matrix_size+kk] != NULL) {
            subflow.emplace([&, ii](){ bdiv (BENCH[kk*matrix_size+kk], BENCH[ii*matrix_size+kk]); });
          }
        }
      })
    );
  }

  sync1.precede(tasks);
  sync1 = flow.emplace([](){});
  sync1.gather(tasks);


  tasks.clear();

  for(int i=0; i<num_threads; i++) {
    tasks.emplace_back(
      flow.emplace([&, id=i](auto &subflow){
        int part = (matrix_size - kk - 1 + num_threads - 1)/num_threads;
        int beg = part*id + kk + 1;
        int end = beg + part < matrix_size ? beg+part : matrix_size; 
        for (int ii=beg; ii<end; ii++)
          if (BENCH[ii*matrix_size+kk] != NULL)
            for (int jj=kk+1; jj<matrix_size; jj++)
              if (BENCH[kk*matrix_size+jj] != NULL)
              {
                subflow.emplace([&, ii, jj](){
                 if (BENCH[ii*matrix_size+jj]==NULL) BENCH[ii*matrix_size+jj] = allocate_clean_block();
                 bmod(BENCH[ii*matrix_size+kk], BENCH[kk*matrix_size+jj], BENCH[ii*matrix_size+jj]);
                });
              }
      })
    );
  }

  sync1.precede(tasks);

  tf::Executor executor (num_threads);
  executor.run_n(flow, matrix_size).wait();

  printf(" completed!\n");
}
