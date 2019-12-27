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

/*
 *  Original code from the Cilk project
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

/*
 * this program uses an algorithm that we call `cilksort'.
 * The algorithm is essentially mergesort:
 *
 *   cilksort(in[1..n]) =
 *       spawn cilksort(in[1..n/2], tmp[1..n/2])
 *       spawn cilksort(in[n/2..n], tmp[n/2..n])
 *       sync
 *       spawn cilkmerge(tmp[1..n/2], tmp[n/2..n], in[1..n])
 *
 *
 * The procedure cilkmerge does the following:
 *       
 *       cilkmerge(A[1..n], B[1..m], C[1..(n+m)]) =
 *          find the median of A \union B using binary
 *          search.  The binary search gives a pair
 *          (ma, mb) such that ma + mb = (n + m)/2
 *          and all elements in A[1..ma] are smaller than
 *          B[mb..m], and all the B[1..mb] are smaller
 *          than all elements in A[ma..n].
 *
 *          spawn cilkmerge(A[1..ma], B[1..mb], C[1..(n+m)/2])
 *          spawn cilkmerge(A[ma..m], B[mb..n], C[(n+m)/2 .. (n+m)])
 *          sync
 *
 * The algorithm appears for the first time (AFAIK) in S. G. Akl and
 * N. Santoro, "Optimal Parallel Merging and Sorting Without Memory
 * Conflicts", IEEE Trans. Comp., Vol. C-36 No. 11, Nov. 1987 .  The
 * paper does not express the algorithm using recursion, but the
 * idea of finding the median is there.
 *
 * For cilksort of n elements, T_1 = O(n log n) and
 * T_\infty = O(log^3 n).  There is a way to shave a
 * log factor in the critical path (left as homework).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "sort.hpp"
#include <taskflow/taskflow.hpp> 
//#include "bots.h"
//#include "app-desc.h"



static unsigned long rand_nxt = 0;

static inline unsigned long my_rand(void)
{
  rand_nxt = rand_nxt * 1103515245 + 12345;
  return rand_nxt;
}

static inline void my_srand(unsigned long seed)
{
  rand_nxt = seed;
}

static inline ELM med3(ELM a, ELM b, ELM c) {
  if (a < b) {
    if (b < c) {
      return b;
    } else {
      if (a < c)
        return c;
      else
        return a;
    }
  } else {
    if (b > c) {
      return b;
    } else {
      if (a > c)
        return c;
      else
        return a;
    }
  }
}

/*
 * simple approach for now; a better median-finding
 * may be preferable
 */
static inline ELM choose_pivot(ELM *low, ELM *high)
{
     return med3(*low, *high, low[(high - low) / 2]);
}

static ELM *seqpart(ELM *low, ELM *high) {
  ELM pivot;
  ELM h, l;
  ELM *curr_low = low;
  ELM *curr_high = high;

  pivot = choose_pivot(low, high);

  while (1) {
    while ((h = *curr_high) > pivot)
      curr_high--;

    while ((l = *curr_low) < pivot)
      curr_low++;

    if (curr_low >= curr_high)
      break;

    *curr_high-- = l;
    *curr_low++ = h;
  }

  /*
   * I don't know if this is really necessary.
   * The problem is that the pivot is not always the
   * first element, and the partition may be trivial.
   * However, if the partition is trivial, then
   * *high is the largest element, whence the following
   * code.
   */
  if (curr_high < high)
    return curr_high;
  else
    return curr_high - 1;
}

#define swap(a, b) \
{ \
  ELM tmp;\
  tmp = a;\
  a = b;\
  b = tmp;\
}

static void insertion_sort(ELM *low, ELM *high) {
  ELM *p, *q;
  ELM a, b;

  for (q = low + 1; q <= high; ++q) {
    a = q[0];
    for (p = q - 1; p >= low && (b = p[0]) > a; p--)
      p[1] = b;
    p[1] = a;
  }
}

/*
 * tail-recursive quicksort, almost unrecognizable :-)
 */
void seqquick(ELM *low, ELM *high) {
  ELM *p;

  while (high - low >= qsort_cutoff_value) {
    p = seqpart(low, high);
    seqquick(low, p);
    low = p + 1;
  }

  insertion_sort(low, high);
}

void seqmerge(ELM *low1, ELM *high1, ELM *low2, ELM *high2,
        ELM *lowdest) {
  ELM a1, a2;

  /*
   * The following 'if' statement is not necessary
   * for the correctness of the algorithm, and is
   * in fact subsumed by the rest of the function.
   * However, it is a few percent faster.  Here is why.
   *
   * The merging loop below has something like
   *   if (a1 < a2) {
   *        *dest++ = a1;
   *        ++low1;
   *        if (end of array) break;
   *        a1 = *low1;
   *   }
   *
   * Now, a1 is needed immediately in the next iteration
   * and there is no way to mask the latency of the load.
   * A better approach is to load a1 *before* the end-of-array
   * check; the problem is that we may be speculatively
   * loading an element out of range.  While this is
   * probably not a problem in practice, yet I don't feel
   * comfortable with an incorrect algorithm.  Therefore,
   * I use the 'fast' loop on the array (except for the last 
   * element) and the 'slow' loop for the rest, saving both
   * performance and correctness.
   */

  if (low1 < high1 && low2 < high2) {
    a1 = *low1;
    a2 = *low2;
    for (;;) {
      if (a1 < a2) {
        *lowdest++ = a1;
        a1 = *++low1;
        if (low1 >= high1)
          break;
      } else {
        *lowdest++ = a2;
        a2 = *++low2;
        if (low2 >= high2)
          break;
      }
    }
  }
  if (low1 <= high1 && low2 <= high2) {
    a1 = *low1;
    a2 = *low2;
    for (;;) {
      if (a1 < a2) {
        *lowdest++ = a1;
        ++low1;
        if (low1 > high1)
          break;
        a1 = *low1;
      } else {
        *lowdest++ = a2;
        ++low2;
        if (low2 > high2)
          break;
        a2 = *low2;
      }
    }
  }
  if (low1 > high1) {
    memcpy(lowdest, low2, sizeof(ELM) * (high2 - low2 + 1));
  } else {
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1 + 1));
  }
}

#define swap_indices(a, b) \
{ \
  ELM *tmp;\
  tmp = a;\
  a = b;\
  b = tmp;\
}

ELM *binsplit(ELM val, ELM *low, ELM *high)
{
     /*
      * returns index which contains greatest element <= val.  If val is
      * less than all elements, returns low-1
      */
     ELM *mid;

     while (low != high) {
    mid = low + ((high - low + 1) >> 1);
    if (val <= *mid)
         high = mid - 1;
    else
         low = mid;
     }

     if (*low > val)
    return low - 1;
     else
    return low;
}


void cilkmerge_par(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest)
{
     /*
      * Cilkmerge: Merges range [low1, high1] with range [low2, high2] 
      * into the range [lowdest, ...]  
      */

     ELM *split1, *split2;  /*
         * where each of the ranges are broken for 
         * recursive merge 
         */
     long int lowsize;    /*
         * total size of lower halves of two
         * ranges - 2 
         */

     /*
      * We want to take the middle element (indexed by split1) from the
      * larger of the two arrays.  The following code assumes that split1
      * is taken from range [low1, high1].  So if [low1, high1] is
      * actually the smaller range, we should swap it with [low2, high2] 
      */

     if (high2 - low2 > high1 - low1) {
    swap_indices(low1, low2);
    swap_indices(high1, high2);
     }
     if (high2 < low2) {
    /* smaller range is empty */
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
    return;
     }
     if (high2 - low2 < insertion_cutoff_value ) {
    seqmerge(low1, high1, low2, high2, lowdest);
    return;
     }
     /*
      * Basic approach: Find the middle element of one range (indexed by
      * split1). Find where this element would fit in the other range
      * (indexed by split 2). Then merge the two lower halves and the two
      * upper halves. 
      */

     split1 = ((high1 - low1 + 1) / 2) + low1;
     split2 = binsplit(*split1, low2, high2);
     lowsize = split1 - low1 + split2 - low2;

     /* 
      * directly put the splitting element into
      * the appropriate location
      */
     *(lowdest + lowsize + 1) = *split1;
#pragma omp task untied
     cilkmerge_par(low1, split1 - 1, low2, split2, lowdest);
#pragma omp task untied
     cilkmerge_par(split1 + 1, high1, split2 + 1, high2,
         lowdest + lowsize + 2);
#pragma omp taskwait

     return;
}

void cilksort_par(ELM *low, ELM *tmp, long size)
{
     /*
      * divide the input in four parts of the same size (A, B, C, D)
      * Then:
      *   1) recursively sort A, B, C, and D (in parallel)
      *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
      *   3) merge tmp1 and tmp2 into the original array
      */
     long quarter = size / 4;
     ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

     if (size < merge_cutoff_value ) {
    /* quicksort when less than 1024 elements */
    seqquick(low, low + size - 1);
    return;
     }
     A = low;
     tmpA = tmp;
     B = A + quarter;
     tmpB = tmpA + quarter;
     C = B + quarter;
     tmpC = tmpB + quarter;
     D = C + quarter;
     tmpD = tmpC + quarter;

#pragma omp task untied
     cilksort_par(A, tmpA, quarter);
#pragma omp task untied
     cilksort_par(B, tmpB, quarter);
#pragma omp task untied
     cilksort_par(C, tmpC, quarter);
#pragma omp task untied
     cilksort_par(D, tmpD, size - 3 * quarter);
#pragma omp taskwait

#pragma omp task untied
     cilkmerge_par(A, A + quarter - 1, B, B + quarter - 1, tmpA);
#pragma omp task untied
     cilkmerge_par(C, C + quarter - 1, D, low + size - 1, tmpC);
#pragma omp taskwait

     cilkmerge_par(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
}

void scramble_array( ELM *array )
{
  unsigned long i;
  unsigned long j;

  for (i = 0; i < array_size; ++i) {
    j = my_rand();
    j = j % array_size;
    swap(array[i], array[j]);
  }
}

void fill_array( ELM *array ) {
  unsigned long i;

  /* first, fill with integers 1..size */
  for (i = 0; i < array_size; ++i) {
    array[i] = i;
  }
}

void sort_init ( void ) {
  /* Checking arguments */
  if (array_size < 4) {
    printf("Array size can not be less than 4, using 4 as a parameter.\n");
    array_size = 4;
  }

  if (insertion_cutoff_value < 2) {
    printf("Sequential Insertion cutoff value can not be less than 2, using 2 as a parameter.\n");
    insertion_cutoff_value = 2;
  }
  else if (insertion_cutoff_value > array_size ) {
    printf("Sequential Insertion cutoff value can not be greather than vector size, using Array size as a parameter.\n");
    insertion_cutoff_value = array_size;
  }

  if (merge_cutoff_value > array_size ) {
    printf("Merge cutoff can not be greather than vector size, using Array size as a parameter.\n");
    merge_cutoff_value = array_size;
  }
  if (qsort_cutoff_value > array_size ) {
    printf("Quick sort cutoff value can not be greather than vector size, using Array size as a parameter.\n");
    qsort_cutoff_value = array_size;
  }

  if (qsort_cutoff_value > merge_cutoff_value) {
    printf("Insertion cutoff value can not be greather than Quick sort cutoff value, using Merge cutoff value as a parameter.\n");
    qsort_cutoff_value = merge_cutoff_value;
  }

  std::cout << "array size = " << array_size << std::endl;
  array = (ELM *) malloc(array_size * sizeof(ELM));
  tmp = (ELM *) malloc(array_size * sizeof(ELM));

  my_srand(1);
  fill_array(array);
  scramble_array(array);
}






void cilkmerge_par_tf(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest, auto &flow)
{
     /*
      * Cilkmerge: Merges range [low1, high1] with range [low2, high2] 
      * into the range [lowdest, ...]  
      */

     ELM *split1, *split2;  /*
         * where each of the ranges are broken for 
         * recursive merge 
         */
     long int lowsize;    /*
         * total size of lower halves of two
         * ranges - 2 
         */

     /*
      * We want to take the middle element (indexed by split1) from the
      * larger of the two arrays.  The following code assumes that split1
      * is taken from range [low1, high1].  So if [low1, high1] is
      * actually the smaller range, we should swap it with [low2, high2] 
      */

     if (high2 - low2 > high1 - low1) {
    swap_indices(low1, low2);
    swap_indices(high1, high2);
     }
     if (high2 < low2) {
    /* smaller range is empty */
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
    return;
     }
     if (high2 - low2 < insertion_cutoff_value ) {
    seqmerge(low1, high1, low2, high2, lowdest);
    return;
     }
     /*
      * Basic approach: Find the middle element of one range (indexed by
      * split1). Find where this element would fit in the other range
      * (indexed by split 2). Then merge the two lower halves and the two
      * upper halves. 
      */

     split1 = ((high1 - low1 + 1) / 2) + low1;
     split2 = binsplit(*split1, low2, high2);
     lowsize = split1 - low1 + split2 - low2;

     /* 
      * directly put the splitting element into
      * the appropriate location
      */
     *(lowdest + lowsize + 1) = *split1;

     flow.emplace(
       [&, low1, split1, low2, split2, lowdest](auto &subflow){
         cilkmerge_par_tf(low1, split1 - 1, low2, split2, lowdest, subflow);
       }
     );

     flow.emplace(
       [&, split1, high1, low2, lowsize, high2, split2, lowdest](auto &subflow){
         cilkmerge_par_tf(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2, subflow);
       }
     );
     return;
}




void cilksort_par_tf(ELM *low, ELM *tmp, long size, tf::Subflow &flow)
{
     /*
      * divide the input in four parts of the same size (A, B, C, D)
      * Then:
      *   1) recursively sort A, B, C, and D (in parallel)
      *   2) merge A and B into tmp1, and C and D into tmp2 (in parallel)
      *   3) merge tmp1 and tmp2 into the original array
      */
     long quarter = size / 4;
     ELM *A, *B, *C, *D, *tmpA, *tmpB, *tmpC, *tmpD;

     if (size < merge_cutoff_value ) {
       /* quicksort when less than 1024 elements */
       seqquick(low, low + size - 1);
       return;
     }
     A = low;
     tmpA = tmp;
     B = A + quarter;
     tmpB = tmpA + quarter;
     C = B + quarter;
     tmpC = tmpB + quarter;
     D = C + quarter;
     tmpD = tmpC + quarter;

     auto s1 = flow.emplace([&, A, tmpA, quarter]( auto &flow){cilksort_par_tf(A, tmpA, quarter, flow);});
     auto s2 = flow.emplace([&, B, tmpB, quarter]( auto &flow){cilksort_par_tf(B, tmpB, quarter, flow);});
     auto s3 = flow.emplace([&, C, tmpC, quarter]( auto &flow){cilksort_par_tf(C, tmpC, quarter, flow);});
     auto s4 = flow.emplace([&, D, tmpD, quarter]( auto &flow){cilksort_par_tf(D, tmpD, quarter, flow);});

     auto sync = flow.emplace([](){}).gather(s1, s2, s3, s4);



    auto m1 = flow.emplace([&, A, quarter, B, tmpA]( auto &subflow){
     cilkmerge_par_tf(A, A + quarter - 1, B, B + quarter - 1, tmpA, subflow);
    });
    auto m2 = flow.emplace([&, C, quarter, D, tmpC, low, size]( auto &subflow){
      cilkmerge_par_tf(C, C + quarter - 1, D, low + size - 1, tmpC, subflow);
    });

    sync.precede(m1, m2);
   
    auto m3 = flow.emplace([&, tmpA, tmpC, size, A]( auto &subflow){
     cilkmerge_par_tf(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A, subflow);
    });

    m3.gather(m1, m2);
}




void sort_par ( void )
{
  omp_set_num_threads(14);

  printf("Computing multisort algorithm (n=%d) ", array_size);
  #pragma omp parallel
  #pragma omp single nowait
  #pragma omp task untied
  cilksort_par(array, tmp, array_size);
  printf(" completed!\n");
}




void sort_par_tf ( void )
{
  tf::Taskflow flow;
  printf("Computing multisort algorithm (n=%d) ", array_size);
  flow.emplace([&]( auto &flow ){ cilksort_par_tf(array, tmp, array_size, flow); });
  tf::Executor executor {14};
  executor.run(flow).wait();
  //cilksort_par_tf(array, tmp, array_size, flow);
  printf(" completed!\n");
}

int sort_verify ( void ) {
  int i, success = 1;
  for (i = 0; i < array_size; ++i)
    if (array[i] != i)
      success = 0;

  return success ? 0 : -1;
  //return success ? BOTS_RESULT_SUCCESSFUL : BOTS_RESULT_UNSUCCESSFUL;
}

