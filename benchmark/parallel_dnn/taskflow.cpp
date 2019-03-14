#include "dnn.hpp"
#include <taskflow/taskflow.hpp>  


struct TF_DNNTrainingPattern : public tf::Framework {

    TF_DNNTrainingPattern() { 
      init_dnn(dnn); 
      build_task_graph();
    };

    void validate(Eigen::MatrixXf &mat, Eigen::VectorXi &vec) {
      dnn.validate(mat, vec);
    }



    void build_task_graph() {
      auto f_task = emplace(
        [&]() { forward_task(dnn, IMAGES, LABELS); }
      );

      std::vector<tf::Task> backward_tasks;
      std::vector<tf::Task> update_tasks;

      for(int j=dnn.acts.size()-1; j>=0; j--) {
        // backward propagation
        auto& b_task = backward_tasks.emplace_back(emplace(
          [&, i=j] () { backward_task(dnn, i, IMAGES); }
        ));

        // update weight 
        auto& u_task = update_tasks.emplace_back(
          emplace([&, i=j] () { dnn.update(i); })
        );

        if(j + 1u == dnn.acts.size()) {
          f_task.precede(b_task);
        }
        else {
          backward_tasks[backward_tasks.size()-2].precede(b_task);
        }
        b_task.precede(u_task);
      }  
    }

    MNIST_DNN dnn;
};

struct TF_DNNTrainingEpoch : public tf::Framework {

    TF_DNNTrainingEpoch(TF_DNNTrainingPattern &dnn_pattern) { 
      std::vector<tf::Task> tasks;
      for(auto i=0u; i<NUM_ITERATIONS; i++) {
        tasks.emplace_back(composed_of(dnn_pattern));
      }
      linearize(tasks);
    }
};

void run_taskflow(unsigned num_epochs, unsigned num_threads) {

  tf::Taskflow tf {num_threads};
  auto dnn_patterns = std::make_unique<TF_DNNTrainingPattern[]>(NUM_DNNS);
  auto dnns = std::make_unique<std::unique_ptr<TF_DNNTrainingEpoch>[]>(NUM_DNNS); 

  for(size_t i=0; i<NUM_DNNS; i++) {
    dnns[i] = std::make_unique<TF_DNNTrainingEpoch>(dnn_patterns[i]);
  }
  
  std::vector<tf::Task> tasks;
  tf::Framework parallel_dnn;
  for(size_t i=0; i<NUM_DNNS; i++) {
    parallel_dnn.composed_of(*(dnns[i]));
  }

  //auto t1 = std::chrono::high_resolution_clock::now();
  parallel_dnn.emplace([&](){
    for(size_t i=0; i<NUM_DNNS; i++) {
      //std::cout << "Validate " << i << "th NN: ";
      dnn_patterns[i].validate(TEST_IMAGES, TEST_LABELS);
    }
    shuffle(IMAGES, LABELS);
    //report_runtime(t1);
  }).gather(tasks);

  tf.run_n(parallel_dnn, num_epochs).get();
  //tf.run_n(parallel_dnn, 4).get();
  //std::cout << parallel_dnn.dump() << std::endl;
}

