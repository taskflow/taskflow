//Copyright (c) 2009 Princeton University
//Written by Christian Bienia
//Generate input files for blackscholes benchmark

// This file has no dependency:
//   Compile: g++/gcc -O2 inputgen.c -o inputgen 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Precision to use
#define fptype double

typedef struct OptionData_ {
  fptype s;          // spot price
  fptype strike;     // strike price
  fptype r;          // risk-free interest rate
  fptype divq;       // dividend rate
  fptype v;          // volatility
  fptype t;          // time to maturity or option expiration in years 
  //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
  const char *OptionType;  // Option type.  "P"=PUT, "C"=CALL
  fptype divs;       // dividend vals (not used in this test)
  fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

//Total number of options in optionData.txt
#define MAX_OPTIONS 1000

OptionData data_init[] = {
    #include "optionData.txt"
};

int main (int argc, char **argv) {
  int numOptions;
  char *fileName;
  int rv;
  int i;

  if (argc != 3) {
    printf("Usage:\n\t%s <numOptions> <fileName>\n", argv[0]);
    exit(1);
  }
  numOptions = atoi(argv[1]);
  fileName = argv[2];
  if(numOptions < 1) {
    printf("ERROR: Number of options must at least be 1.\n");
    exit(1);
  }

  if(strcmp(argv[2], "optionData.txt") == 0) {
    printf("Output file cannot be optionData.txt\n");
    exit(1);
  }

  FILE *file;
  file = fopen(fileName, "w");
  if(file == NULL) {
    printf("ERROR: Unable to open file `%s'.\n", fileName);
    exit(1);
  }

  //write number of options
  rv = fprintf(file, "%i\n", numOptions);
  if(rv < 0) {
    printf("ERROR: Unable to write to file `%s'.\n", fileName);
    fclose(file);
    exit(1);
  }

  //write values for options
  for(i=0; i<numOptions; i++) {
    //NOTE: DG RefValues specified exceed double precision, output will deviate
    rv = fprintf(file, "%.2f %.2f %.4f %.2f %.2f %.2f %c %.2f %.18f\n", data_init[i % MAX_OPTIONS].s, data_init[i % MAX_OPTIONS].strike, data_init[i % MAX_OPTIONS].r, data_init[i % MAX_OPTIONS].divq, data_init[i % MAX_OPTIONS].v, data_init[i % MAX_OPTIONS].t, data_init[i % MAX_OPTIONS].OptionType[0], data_init[i % MAX_OPTIONS].divs, data_init[i % MAX_OPTIONS].DGrefval);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", fileName);
      fclose(file);
      exit(1);
    }
  }

  rv = fclose(file);
  if(rv != 0) {
    printf("ERROR: Unable to close file `%s'.\n", fileName);
    exit(1);
  }

  return 0;
}
