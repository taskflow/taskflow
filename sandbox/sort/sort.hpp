#pragma once

//inline int array_size = 32*1024*1024;
inline int array_size = 8*1024*1024;

inline int merge_cutoff_value = 2*1024;
inline int qsort_cutoff_value = 2*1024;
//inline int insertion_cutoff_value = 20; 
inline int insertion_cutoff_value = 2048; 

typedef long ELM;
inline ELM *array, *tmp;

void seqquick(ELM *low, ELM *high); 
void seqmerge(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest);
ELM *binsplit(ELM val, ELM *low, ELM *high); 
void cilkmerge(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest);
void cilkmerge_par(ELM *low1, ELM *high1, ELM *low2, ELM *high2, ELM *lowdest);
void cilksort(ELM *low, ELM *tmp, long size);
void cilksort_par(ELM *low, ELM *tmp, long size);
void scramble_array( ELM *array ); 
void fill_array( ELM *array ); 
void sort ( void ); 

void sort_par (void);
void sort_init (void);
int sort_verify (void);

void sort_par_tf (void);

#define BOTS_APP_INIT sort_init()

#define KERNEL_INIT
#define KERNEL_CALL sort_par()
#define KERNEL_CHECK sort_verify()



