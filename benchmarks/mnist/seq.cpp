#include "dnn.hpp" 

void run_sequential(MNIST& D, unsigned) {

  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0u; e<D.epoch; e++) { 
    for(auto it=0u; it<iter_num; it++) {
      // Foward propagation
      for(size_t i=0; i<D.acts.size(); i++) {
        if(i == 0){
          D.forward(i, D.images.middleRows(D.beg_row, D.batch_size));
        }
        else {
          D.forward(i, D.Ys[i-1]);
        }
      }

      // Calculate loss  
      D.loss(D.labels);

      // Backward propagation
      for(int i=D.acts.size()-1; i>=0; i--) {
        if(i > 0) {
          D.backward(i, D.Ys[i-1].transpose());
        }
        else {
          D.backward(i, D.images.middleRows(D.beg_row, D.batch_size).transpose());
        }
      }

      // Update parameters
      for(int i=D.acts.size()-1; i>=0; i--) {
        D.update(i);
      }

      // Get next batch
      D.beg_row += D.batch_size;
      if(D.beg_row >= D.images.rows()) {
        D.beg_row = 0;
      }
    } // End of iterations
    // Shuffle input 
    D.shuffle(D.images, D.labels, D.images.rows());
  } // End of epoch
}

