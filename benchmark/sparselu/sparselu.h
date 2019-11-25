#ifndef SPARSELU_H
#define SPARSELU_H

#define EPSILON 1.0E-6

int checkmat (float *M, float *N);
void genmat (float *M[]);
void print_structure(char *name, float *M[]);
float * allocate_clean_block();
void lu0(float *diag);
void bdiv(float *diag, float *row);
void bmod(float *row, float *col, float *inner);
void fwd(float *diag, float *col);

void sparselu_init (float ***pBENCH, char *pass); 
void sparselu(float **BENCH);
void sparselu_fini (float **BENCH, char *pass); 

void sparselu_seq_call(float **BENCH);
void sparselu_par_call(float **BENCH);

int sparselu_check(float **SEQ, float **BENCH);

inline int matrix_size {64};
inline int submatrix_size {32};

void sparselu_tf_for(float **BENCH);

#endif
