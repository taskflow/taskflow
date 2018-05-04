#include "taskflow.hpp"

#define MIN_PER_RANK 1 // fatness 
#define MAX_PER_RANK 10
#define MIN_RANKS 1   // height
#define MAX_RANKS 20
#define PERCENT 20     // chance to have an edge

int main (void) {

  ::srand(0);

  int nodes = 0;
  int ranks = MIN_RANKS + (rand () % (MAX_RANKS - MIN_RANKS + 1));

  printf ("digraph G {\n");

  for(int i = 0; i < ranks; i++) {
    /* New nodes of 'higher' rank than all nodes generated till now.  */
    int new_nodes = MIN_PER_RANK + (rand () % (MAX_PER_RANK - MIN_PER_RANK + 1));
    /* Edges from old nodes ('nodes') to new ones ('new_nodes').  */
    for(int j = 0; j < nodes; j++) {
      for(int k = 0; k < new_nodes; k++) {
        if((rand () % 100) < PERCENT) {
          printf ("  %d -> %d;\n", j, k + nodes); /* An Edge.  */
        }
      }
    }
    nodes += new_nodes; /* Accumulate into old node set.  */
  }

  printf ("}\n");
  return 0;
}
