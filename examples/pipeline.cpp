#include <taskflow/taskflow.hpp>
#include <taskflow/pipeflow.hpp>

int main() {

  //tf::Pipeflow pf{
  //  4,
  //  tf::Filter{tf::SERIAL, [] (tf::FlowControl& ctrl) { return int{1}; }},  // TODO: argument?
  //  tf::Filter{tf::SERIAL, [] (int arg) { return std::string{"1234"}; }},
  //  tf::Filter{tf::PARALLEL, [] (const std::string& str) {  }}
  //};

  //tf::Executor executor;
  //
  //executor.run(pf).wait();


  return 0;
}
