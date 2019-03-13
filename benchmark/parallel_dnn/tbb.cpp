#include "dnn.hpp"
#include <memory>  // unique_ptr
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>

using namespace tbb;
using namespace tbb::flow;

struct DNNTrainingPattern {
  DNNTrainingPattern() {
    init_dnn(dnn); 
    build_task_graph();
  }

  void train() {
    f_task->try_put(continue_msg());
    G.wait_for_all();
  }

  void build_task_graph() {

    f_task = std::make_unique<continue_node<continue_msg>>(G, 
        [&](const continue_msg&) { forward_task(dnn, IMAGES, LABELS); }
        );

    for(int j=dnn.acts.size()-1; j>=0; j--) {
      // backward propagation
      auto& b_task = backward_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(G, 
            [&, i=j](const continue_msg&) {
            backward_task(dnn, i, IMAGES);
            }) 
          );

      auto& u_task = update_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(G, 
            [&, i=j](const continue_msg&) {
            dnn.update(i);
            }) 
          );

      if(j + 1u == dnn.acts.size()) {
        make_edge(*f_task, *b_task);
      }
      else {
        make_edge(*backward_tasks[backward_tasks.size()-2], *b_task);
      }

      make_edge(*b_task, *u_task);
    }  
  }

  tbb::flow::graph G;   
  MNIST_DNN dnn;

  std::unique_ptr<continue_node<continue_msg>> f_task;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> backward_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> update_tasks;
};

void run_tbb(unsigned num_epochs, unsigned num_threads) {

  auto dnn_patterns = std::make_unique<DNNTrainingPattern[]>(NUM_DNNS);
  auto dnns = std::make_unique<std::unique_ptr<continue_node<continue_msg>>[]>(NUM_DNNS);

  tbb::flow::graph parallel_dnn;

  for(auto i=0u; i<NUM_DNNS; i++) {
    dnns[i] = std::make_unique<continue_node<continue_msg>>(parallel_dnn,
      [&, id=i](const continue_msg&){
        for(size_t i=0; i<NUM_ITERATIONS; i++) {
          dnn_patterns[id].train();
        }
      }
    );
  }

  auto sync_node = std::make_unique<continue_node<continue_msg>>(parallel_dnn, 
    [&](const continue_msg&) {
      for(size_t i=0; i<NUM_DNNS; i++) {
        std::cout << "Validate " << i << "th NN: ";
        dnn_patterns[i].dnn.validate(TEST_IMAGES, TEST_LABELS);
      }
      shuffle(IMAGES, LABELS);
    }
  );

  for(auto i=0u; i<NUM_DNNS; i++) {
    make_edge(*(dnns[i]), *sync_node);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  for(auto i=0u; i<100; i++) {
    for(auto i=0u; i<NUM_DNNS; i++) {
      dnns[i]->try_put(continue_msg());
    }

    parallel_dnn.wait_for_all();
    report_runtime(t1);
  }
}
