// This program demonstrates how to pipeline a sequence of linearly dependent
// tasks (stage function) over a directed acyclic graph.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

// 1st-stage function
void f1(const std::string& node) {
  printf("f1(%s)\n", node.c_str());
}

// 2nd-stage function
void f2(const std::string& node) {
  printf("f2(%s)\n", node.c_str());
}

// 3rd-stage function
void f3(const std::string& node) {
  printf("f3(%s)\n", node.c_str());
}

int main() {

  tf::Taskflow taskflow("graph processing pipeline");
  tf::Executor executor;

  const size_t num_lines = 2;

  // a topological order of the graph
  //    |-> B
  // A--|
  //    |-> C
  const std::vector<std::string> nodes = {"A", "B", "C"};

  // the pipeline consists of three serial pipes
  // and up to two concurrent scheduling tokens
  tf::Pipeline pl(num_lines,

    // first pipe calls f1
    tf::Pipe{tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) {
      if(pf.token() == nodes.size()) {
        pf.stop();
      }
      else {
        f1(nodes[pf.token()]);
      }
    }},

    // second pipe calls f2
    tf::Pipe{tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) {
      f2(nodes[pf.token()]);
    }},

    // third pipe calls f3
    tf::Pipe{tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) {
      f3(nodes[pf.token()]);
    }}
  );

  // build the pipeline graph using composition
  tf::Task init = taskflow.emplace([](){ std::cout << "ready\n"; })
                          .name("starting pipeline");
  tf::Task task = taskflow.composed_of(pl)
                          .name("pipeline");
  tf::Task stop = taskflow.emplace([](){ std::cout << "stopped\n"; })
                          .name("pipeline stopped");

  // create task dependency
  init.precede(task);
  task.precede(stop);

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}
