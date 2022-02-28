// This program demonstrates how to create a pipeline scheduling framework
// that propagates a series of integers and adds one to the result at each
// stage, using a range of pipes provided by the application.
//
// The pipeline has the following structure:
//
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
// |    |    |
// v    v    v
// o -> o -> o
//
// Then, the program resets the pipeline to a new range of five pipes.
//
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o
// |    |    |    |    |
// v    v    v    v    v
// o -> o -> o -> o -> o

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>
#include <sstream>
#include <thread>

class myStringStream{
  mutable std::mutex seq_alloc_mtx;
  std::stringstream oss;
  public:
  template<class T>
  myStringStream& operator<<(T const &t)
  {
    std::lock_guard<std::mutex> lock(seq_alloc_mtx);
      oss<< t;
      return *this;
  }
  std::string str(){
    return oss.str();
  }
};


int main() {
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor(8);
  myStringStream s;

  tf::Taskflow sub_taskflow("sub_taskflow");
  auto task2 = sub_taskflow.emplace([&s](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){
      std::stringstream oss; 
      oss << "["<<pf.token()<<" "<<pf.line()<<" "<<pf.pipe()<<"]["<<std::this_thread::get_id()<<"]["<<wv.id()<<" "<<wv.queue_size()<<"]Task2\t\n";
      //std::cout<<oss.str()<<std::endl;
      s<<oss.str(); 
    });
  auto task3 = sub_taskflow.emplace([&s](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ 
    std::stringstream oss; 
      oss << "["<<pf.token()<<" "<<pf.line()<<" "<<pf.pipe()<<"]["<<std::this_thread::get_id()<<"]["<<wv.id()<<" "<<wv.queue_size()<<"]Task3\t\n";
      //std::cout<<oss.str()<<std::endl;
      s<<oss.str(); 
      });
  auto task1 = sub_taskflow.emplace([&s](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ 
    std::stringstream oss; 
      oss << "["<<pf.token()<<" "<<pf.line()<<" "<<pf.pipe()<<"]["<<std::this_thread::get_id()<<"]["<<wv.id()<<" "<<wv.queue_size()<<"]Task1\t\n";
      //std::cout<<oss.str()<<std::endl;
      s<<oss.str(); 
      });
  task1.name("task1");
  task2.name("task2");
  task3.name("task3");
  task1.precede(task2);
  task2.precede(task3);



  const size_t num_lines = 128;
  const size_t num_tokens = 1123;
  //s<<"Line="<<num_lines<<"\t"<<"num_tokens"<<num_tokens<<"\n";

  // create data storage
  std::array<int, num_lines> buffer;


  
  // define the pipe callable
  auto pipe_callable = [&buffer, &sub_taskflow, &executor, &s, num_tokens] (tf::Pipeflow& pf) mutable {
    
    std::stringstream oss; 
      oss <<"["<<pf.token()<<" "<<pf.line()<<" "<<pf.pipe()<<"]["<<std::this_thread::get_id()<<"]Running Stage:\n";
      //std::cout<<oss.str()<<std::endl;
      s<<oss.str(); 
    tf::Taskflow *myTaskflow = sub_taskflow.clone();
    if(pf.pipe()== 0 && pf.token() == num_tokens) {
      pf.stop();
    }
    else {
      //buffer[pf.line()] = pf.token();
      executor.run(*myTaskflow, pf).wait();
    }
  };

  // create a vector of three pipes
  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
  for(size_t i=1; i<3; i++) {
    pipes.emplace_back(tf::PipeType::PARALLEL, pipe_callable);
  }
  
  // create a pipeline of four parallel lines using the given vector of pipes
  tf::ScalablePipeline pl(num_lines, pipes.begin(), pipes.end());

  // build the pipeline graph using composition
  // tf::Task init = taskflow.emplace([&s](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ s << "ready\n"; })
  //                         .name("starting pipeline");
  tf::Task task = taskflow.composed_of(pl)
                          .name("pipeline");
  // tf::Task stop = taskflow.emplace([&s](tf::WorkerView wv, tf::TaskView tv, tf::Pipeflow& pf){ s << "stopped\n"; })
  //                         .name("pipeline stopped");

  // create task dependency
  // init.precede(task);
  // task.precede(stop);
  
  // dump the pipeline graph structure (with composition)
 //taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();
  std::cout<<s.str()<<std::endl;

  // reset the pipeline to a new range of five pipes and starts from
  // the initial state (i.e., token counts from zero)
  // for(size_t i=0; i<2; i++) {
  //   pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
  // }
  // pl.reset(pipes.begin(), pipes.end());

  // executor.run(taskflow).wait();

  return 0;
}



