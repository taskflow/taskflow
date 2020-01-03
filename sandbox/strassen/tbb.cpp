#include "strassen.hpp"
#include "tbb/task_scheduler_init.h"
#include "tbb/task.h"

class StrassenTBB: public tbb::task {

  REAL *C; 
  REAL *A; 
  REAL *B; 
  unsigned MatrixSize;
  unsigned RowWidthC; 
  unsigned RowWidthA; 
  unsigned RowWidthB; 
  int Depth;

  public:
    StrassenTBB(REAL *C, REAL *A, REAL *B, unsigned MatrixSize,
                unsigned RowWidthC, unsigned RowWidthA, unsigned RowWidthB, int Depth): 
    C(C), A(A), B(B), 
    MatrixSize(MatrixSize), 
    RowWidthC(RowWidthC), RowWidthA(RowWidthA), RowWidthB(RowWidthB), Depth(Depth) {}
  
    task* execute() {
      unsigned QuadrantSize = MatrixSize >> 1; /* MatixSize / 2 */
      unsigned QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize
        + 32;
      unsigned Column, Row;

      /************************************************************************
       ** For each matrix A, B, and C, we'll want pointers to each quandrant
       ** in the matrix. These quandrants will be addressed as follows:
       **  --        --
       **  | A11  A12 |
       **  |          |
       **  | A21  A22 |
       **  --        --
       ************************************************************************/
      REAL /* *A11, *B11, *C11, */ *A12, *B12, *C12,
           *A21, *B21, *C21, *A22, *B22, *C22;

      REAL *S1,*S2,*S3,*S4,*S5,*S6,*S7,*S8,*M2,*M5,*T1sMULT;
#define T2sMULT C22
#define NumberOfVariables 11

      PTR TempMatrixOffset = 0;
      PTR MatrixOffsetA = 0;
      PTR MatrixOffsetB = 0;

      char *Heap;
      void *StartHeap;

      /* Distance between the end of a matrix row and the start of the next row */
      PTR RowIncrementA = ( RowWidthA - QuadrantSize ) << 3;
      PTR RowIncrementB = ( RowWidthB - QuadrantSize ) << 3;
      PTR RowIncrementC = ( RowWidthC - QuadrantSize ) << 3;

      if (MatrixSize <= CUTOFF_SIZE) {
        MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
        return nullptr;
      }

      /* Initialize quandrant matrices */
#define A11 A
#define B11 B
#define C11 C
      A12 = A11 + QuadrantSize;
      B12 = B11 + QuadrantSize;
      C12 = C11 + QuadrantSize;
      A21 = A + (RowWidthA * QuadrantSize);
      B21 = B + (RowWidthB * QuadrantSize);
      C21 = C + (RowWidthC * QuadrantSize);
      A22 = A21 + QuadrantSize;
      B22 = B21 + QuadrantSize;
      C22 = C21 + QuadrantSize;

      /* Allocate Heap Space Here */
      Heap = static_cast<char*>(malloc(QuadrantSizeInBytes * NumberOfVariables));
      StartHeap = Heap;

      /* ensure that heap is on cache boundary */
      if ( ((PTR) Heap) & 31)
        Heap = (char*) ( ((PTR) Heap) + 32 - ( ((PTR) Heap) & 31) );

      /* Distribute the heap space over the variables */
      S1 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S3 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S4 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S6 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S7 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      S8 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      M2 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      M5 = (REAL*) Heap; Heap += QuadrantSizeInBytes;
      T1sMULT = (REAL*) Heap; Heap += QuadrantSizeInBytes;

      /***************************************************************************
       ** Step through all columns row by row (vertically)
       ** (jumps in memory by RowWidth => bad locality)
       ** (but we want the best locality on the innermost loop)
       ***************************************************************************/
      for (Row = 0; Row < QuadrantSize; Row++) {

        /*************************************************************************
         ** Step through each row horizontally (addressing elements in each column)
         ** (jumps linearly througn memory => good locality)
         *************************************************************************/
        for (Column = 0; Column < QuadrantSize; Column++) {

          /***********************************************************
           ** Within this loop, the following holds for MatrixOffset:
           ** MatrixOffset = (Row * RowWidth) + Column
           ** (note: that the unit of the offset is number of reals)
           ***********************************************************/
          /* Element of Global Matrix, such as A, B, C */
          #define E(Matrix)   (* (REAL*) ( ((PTR) Matrix) + TempMatrixOffset ) )
          #define EA(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetA ) )
          #define EB(Matrix)  (* (REAL*) ( ((PTR) Matrix) + MatrixOffsetB ) )

          /* FIXME - may pay to expand these out - got higher speed-ups below */
          /* S4 = A12 - ( S2 = ( S1 = A21 + A22 ) - A11 ) */
          E(S4) = EA(A12) - ( E(S2) = ( E(S1) = EA(A21) + EA(A22) ) - EA(A11) );

          /* S8 = (S6 = B22 - ( S5 = B12 - B11 ) ) - B21 */
          E(S8) = ( E(S6) = EB(B22) - ( E(S5) = EB(B12) - EB(B11) ) ) - EB(B21);

          /* S3 = A11 - A21 */
          E(S3) = EA(A11) - EA(A21);

          /* S7 = B22 - B12 */
          E(S7) = EB(B22) - EB(B12);

          TempMatrixOffset += sizeof(REAL);
          MatrixOffsetA += sizeof(REAL);
          MatrixOffsetB += sizeof(REAL);
        } /* end row loop*/

        MatrixOffsetA += RowIncrementA;
        MatrixOffsetB += RowIncrementB;
      } /* end column loop */

      tbb::task_list list;

      /* M2 = A11 x B11 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(M2, A11, B11, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth+1)
      );

      /* M5 = S1 * S5 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1)
      );

      /* Step 1 of T1 = S2 x S6 + M2 */ 
      list.push_back(
        *new(allocate_child()) StrassenTBB(T1sMULT, S2, S6,  QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth+1)
      );

      /* Step 1 of T2 = T1 + S3 x S7 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(C22, S3, S7, QuadrantSize, RowWidthC /*FIXME*/, QuadrantSize, QuadrantSize, Depth+1)
      );

      /* Step 1 of C11 = M2 + A12 * B21 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(C11, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth+1)
      );
      
      /* Step 1 of C12 = S4 x B22 + T1 + M5 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth+1)
      );

      /* Step 1 of C21 = T2 - A22 * S8 */
      list.push_back(
        *new(allocate_child()) StrassenTBB(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth+1)
      );

      set_ref_count(8);
      spawn_and_wait_for_all(list);

      /**********************************************
       ** Synchronization Point
       **********************************************/
      /***************************************************************************
       ** Step through all columns row by row (vertically)
       ** (jumps in memory by RowWidth => bad locality)
       ** (but we want the best locality on the innermost loop)
       ***************************************************************************/
      for (Row = 0; Row < QuadrantSize; Row++) {
        /*************************************************************************
         ** Step through each row horizontally (addressing elements in each column)
         ** (jumps linearly througn memory => good locality)
         *************************************************************************/
        for (Column = 0; Column < QuadrantSize; Column += 4) {
          REAL LocalM5_0 = *(M5);
          REAL LocalM5_1 = *(M5+1);
          REAL LocalM5_2 = *(M5+2);
          REAL LocalM5_3 = *(M5+3);
          REAL LocalM2_0 = *(M2);
          REAL LocalM2_1 = *(M2+1);
          REAL LocalM2_2 = *(M2+2);
          REAL LocalM2_3 = *(M2+3);
          REAL T1_0 = *(T1sMULT) + LocalM2_0;
          REAL T1_1 = *(T1sMULT+1) + LocalM2_1;
          REAL T1_2 = *(T1sMULT+2) + LocalM2_2;
          REAL T1_3 = *(T1sMULT+3) + LocalM2_3;
          REAL T2_0 = *(C22) + T1_0;
          REAL T2_1 = *(C22+1) + T1_1;
          REAL T2_2 = *(C22+2) + T1_2;
          REAL T2_3 = *(C22+3) + T1_3;
          (*(C11))   += LocalM2_0;
          (*(C11+1)) += LocalM2_1;
          (*(C11+2)) += LocalM2_2;
          (*(C11+3)) += LocalM2_3;
          (*(C12))   += LocalM5_0 + T1_0;
          (*(C12+1)) += LocalM5_1 + T1_1;
          (*(C12+2)) += LocalM5_2 + T1_2;
          (*(C12+3)) += LocalM5_3 + T1_3;
          (*(C22))   = LocalM5_0 + T2_0;
          (*(C22+1)) = LocalM5_1 + T2_1;
          (*(C22+2)) = LocalM5_2 + T2_2;
          (*(C22+3)) = LocalM5_3 + T2_3;
          (*(C21  )) = (- *(C21  )) + T2_0;
          (*(C21+1)) = (- *(C21+1)) + T2_1;
          (*(C21+2)) = (- *(C21+2)) + T2_2;
          (*(C21+3)) = (- *(C21+3)) + T2_3;
          M5 += 4;
          M2 += 4;
          T1sMULT += 4;
          C11 += 4;
          C12 += 4;
          C21 += 4;
          C22 += 4;
        }
        C11 = (REAL*) ( ((PTR) C11 ) + RowIncrementC);
        C12 = (REAL*) ( ((PTR) C12 ) + RowIncrementC);
        C21 = (REAL*) ( ((PTR) C21 ) + RowIncrementC);
        C22 = (REAL*) ( ((PTR) C22 ) + RowIncrementC);
      }
      free(StartHeap);

      return nullptr;
    } // End of execute()

};




void strassen_tbb(unsigned num_threads, REAL *A, REAL *B, REAL *C, int n) {
  tbb::task_scheduler_init init(num_threads);
  auto& root = *new(tbb::task::allocate_root()) StrassenTBB(C, A, B, n, n, n, n, 1);
  tbb::task::spawn_root_and_wait(root);
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads, REAL *A, REAL *B, REAL *C, int n) {
  auto beg = std::chrono::high_resolution_clock::now();
  strassen_tbb(num_threads, A, B, C, n);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

