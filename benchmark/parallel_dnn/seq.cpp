#include "dnn.hpp" 

void run_sequential(MNIST& D, unsigned num_threads) {

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
          //D.backward(i, D.Ys[i-1].transpose()); 
          D.backward(i, D.Ys[i-1]);
        }
        else {
          //D.backward(i, D.images.middleRows(D.beg_row, D.batch_size).transpose());
          D.backward(i, D.images.middleRows(D.beg_row, D.batch_size));
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

    D.validate();
    // Shuffle input 
    D.shuffle(D.images, D.labels, D.images.rows());
  } // End of epoch
}




void run_sequential(MNIST_DNN& D, unsigned num_threads) {

  const auto iter_num = IMAGES.rows()/D.batch_size;

  std::cout << D.epoch << std::endl;
  std::cout << iter_num << std::endl;
  std::cout << D.batch_size << std::endl;
  D.epoch = 20;

  for(auto e=0u; e<D.epoch; e++) { 
    for(auto it=0u; it<iter_num; it++) {
      // Foward propagation
      for(size_t i=0; i<D.acts.size(); i++) {
        if(i == 0){
          D.forward(i, IMAGES.middleRows(D.beg_row, D.batch_size));
        }
        else {
          D.forward(i, D.Ys[i-1]);
        }
      }

      // Calculate loss  
      D.loss(LABELS);

      // Backward propagation
      for(int i=D.acts.size()-1; i>=0; i--) {
        if(i > 0) {
          //D.backward(i, D.Ys[i-1].transpose()); 
          D.backward(i, D.Ys[i-1]);
        }
        else {
          //D.backward(i, D.images.middleRows(D.beg_row, D.batch_size).transpose());
          D.backward(i, IMAGES.middleRows(D.beg_row, D.batch_size));
        }
      }

      // Update parameters
      for(int i=D.acts.size()-1; i>=0; i--) {
        D.update(i);
      }

      // Get next batch
      D.beg_row += D.batch_size;
      if(D.beg_row >= IMAGES.rows()) {
        D.beg_row = 0;
      }
    } // End of iterations 

    // Shuffle input 
    shuffle(IMAGES, LABELS);

    D.validate(TEST_IMAGES, TEST_LABELS);

  } // End of epoch
}




//void run_sequential2(MNIST_DNN& D, unsigned num_threads) {
void run_sequential2(unsigned num_epochs, unsigned num_threads) {

  MNIST_DNN D;
  init_dnn(D); 

  for(auto e=0u; e<100; e++) { 
    for(auto it=0u; it<NUM_ITERATIONS; it++) {
      // Foward propagation
      forward_task(D, IMAGES, LABELS);

      // Calculate loss  
      D.loss(LABELS);

      // Backward propagation
      for(int i=D.acts.size()-1; i>=0; i--) {
        backward_task(D, i, IMAGES);
      }

      // Update parameters
      for(int i=D.acts.size()-1; i>=0; i--) {
        D.update(i);
      }

      // Get next batch
      D.beg_row += D.batch_size;
      if(D.beg_row >= IMAGES.rows()) {
        D.beg_row = 0;
      }
    } // End of iterations 

    // Shuffle input 
    shuffle(IMAGES, LABELS);

    D.validate(TEST_IMAGES, TEST_LABELS);
  } // End of epoch
}
