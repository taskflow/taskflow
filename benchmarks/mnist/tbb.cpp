
#include "dnn.hpp"
#include <memory>  // unique_ptr
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

void run_tbb(MNIST& D, unsigned num_threads) {

  using namespace tbb;
  using namespace tbb::flow;

  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  tbb::flow::graph G;

  std::vector<std::unique_ptr<continue_node<continue_msg>>> forward_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> backward_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> update_tasks;
  std::vector<std::unique_ptr<continue_node<continue_msg>>> shuffle_tasks;

  // Number of parallel shuffle 
  const auto num_storage = num_threads;
  const auto num_par_shf = std::min(num_storage, D.epoch);

  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0u; e<D.epoch; e++) {
    for(auto i=0u; i<iter_num; i++) {
      forward_tasks.emplace_back(
        std::make_unique<continue_node<continue_msg>>(
          G, 
          [&, i=i, e=e%num_par_shf](const continue_msg&) {
            forward_task(D, i, e, mats, vecs);
          }
        ) 
      );
      auto& f_task = forward_tasks.back();

      if(i != 0 || (i == 0 && e != 0)) {
        auto sz = update_tasks.size();
        for(auto j=1u; j<=D.acts.size() ;j++) {
          make_edge(*update_tasks[sz-j], *f_task);
        }         
      }

      for(int j=D.acts.size()-1; j>=0; j--) {

        // backward propagation
        backward_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(
            G, 
            [&, i=j, e=e%num_par_shf] (const continue_msg&) {
              backward_task(D, i, e, mats);
            }
          )
        );
        auto& b_task = backward_tasks.back();

        // update weight 
        update_tasks.emplace_back(
          std::make_unique<continue_node<continue_msg>>(
            G, 
            [&, i=j] (const continue_msg&) {
              D.update(i);
            }
          )
        );
        auto& u_task = update_tasks.back();

        if(j + 1u == D.acts.size()) {
          make_edge(*f_task, *b_task);
        }
        else {
          make_edge(*backward_tasks[backward_tasks.size()-2], *b_task);
        }
        make_edge(*b_task, *u_task);
      } // End of backward propagation  
    } // End of all iterations (task flow graph creation)

    if(e == 0) {
      // No need to shuffle in first epoch
      shuffle_tasks.emplace_back(std::make_unique<continue_node<continue_msg>>(
        G, 
        [](const continue_msg&){}
      ));
      make_edge(*shuffle_tasks.back(), *forward_tasks[forward_tasks.size()-iter_num]);
    }
    else {
      shuffle_tasks.emplace_back(
        std::make_unique<continue_node<continue_msg>>(
          G, 
          [&, e=e%num_par_shf](const continue_msg&) {
            D.shuffle(mats[e], vecs[e], D.images.rows());
          }
        )
      );
      auto& t = shuffle_tasks.back();
      make_edge(*t, *forward_tasks[forward_tasks.size()-iter_num]);

      // This shuffle task starts after belows finish
      //   1. previous shuffle on the same storage
      //   2. the last backward task of previous epoch on the same storage 
      if(e >= num_par_shf) {
        auto prev_e = e - num_par_shf;
        make_edge(*shuffle_tasks[prev_e] ,*t);

        int task_id = (prev_e+1)*iter_num*D.acts.size() - 1;
        make_edge(*backward_tasks[task_id], *t);
      }
    }
  } // End of all epoch

  for(size_t i=0; i<num_par_shf; i++) {
    shuffle_tasks[i]->try_put(continue_msg());
  }
  G.wait_for_all();
}


