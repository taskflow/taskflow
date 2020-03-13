#include "dnn.hpp"
#include <taskflow/taskflow.hpp>  

void run_taskflow(MNIST& D, unsigned num_threads) {
  
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  std::vector<tf::Task> forward_tasks;
  std::vector<tf::Task> backward_tasks;
  std::vector<tf::Task> update_tasks;
  std::vector<tf::Task> shuffle_tasks;

  // Number of parallel shuffle 
  const auto num_storage = num_threads;
  const auto num_par_shf = std::min(num_storage, D.epoch);

  std::vector<Eigen::MatrixXf> mats(num_par_shf, D.images);
  std::vector<Eigen::VectorXi> vecs(num_par_shf, D.labels);

  // Create task flow graph
  const auto iter_num = D.images.rows()/D.batch_size;

  for(auto e=0u; e<D.epoch; e++) {
    for(auto i=0u; i<iter_num; i++) {
      forward_tasks.emplace_back(taskflow.emplace(
        [&, i=i, e=e%num_par_shf]() { forward_task(D, i, e, mats, vecs); }
      ));
      auto& f_task = forward_tasks.back();

      if(i != 0 || (i == 0 && e != 0)) {
        auto sz = update_tasks.size();
        for(auto j=1u; j<=D.acts.size() ;j++) {
          update_tasks[sz-j].precede(f_task);
        }         
      }

      for(int j=D.acts.size()-1; j>=0; j--) {
        // backward propagation
        backward_tasks.emplace_back(taskflow.emplace(
          [&, i=j, e=e%num_par_shf] () { backward_task(D, i, e, mats); }
        ));
        auto& b_task = backward_tasks.back();

        // update weight 
        update_tasks.emplace_back(
          taskflow.emplace([&, i=j] () {D.update(i);})
        );
        auto& u_task = update_tasks.back();

        if(j + 1u == D.acts.size()) {
          f_task.precede(b_task);
        }
        else {
          backward_tasks[backward_tasks.size()-2].precede(b_task);
        }
        b_task.precede(u_task);
      } // End of backward propagation 
    } // End of all iterations (task flow graph creation)


    if(e == 0) {
      // No need to shuffle in first epoch
      shuffle_tasks.emplace_back(taskflow.emplace([](){}));
      shuffle_tasks.back().precede(forward_tasks[forward_tasks.size()-iter_num]);           
    }
    else {
      shuffle_tasks.emplace_back(taskflow.emplace(
        [&, e=e%num_par_shf]() { D.shuffle(mats[e], vecs[e], D.images.rows());}
      ));
      auto& t = shuffle_tasks.back();
      t.precede(forward_tasks[forward_tasks.size()-iter_num]);

      // This shuffle task starts after belows finish
      //   1. previous shuffle on the same storage
      //   2. the last backward task of previous epoch on the same storage 
      if(e >= num_par_shf) {
        auto prev_e = e - num_par_shf;
        shuffle_tasks[prev_e].precede(t);

        int task_id = (prev_e+1)*iter_num*D.acts.size() - 1;
        backward_tasks[task_id].precede(t);
      }
    }
  } // End of all epoch

  executor.run(taskflow).get();
}

