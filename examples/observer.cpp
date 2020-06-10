// Demonstrates the use of observer to monitor worker activities

#include <taskflow/taskflow.hpp>

struct MyObserver : public tf::ObserverInterface {

  MyObserver(const std::string& name) {
    std::cout << "constructing observer " << name << '\n';
  }

  void set_up(size_t num_workers) override final {
    std::cout << "settting up observer with " << num_workers << " workers\n";
  }

  void on_entry(size_t w, tf::TaskView tv) override final {
    std::ostringstream oss;
    oss << "worker " << w << " ready to run " << tv.name() << '\n';
    std::cout << oss.str();
  }

  void on_exit(size_t w, tf::TaskView tv) override final {
    std::ostringstream oss;
    oss << "worker " << w << " finished running " << tv.name() << '\n';
    std::cout << oss.str();
  }

};

int main(){
  
  tf::Executor executor;
  
  // Create a taskflow of eight tasks
  tf::Taskflow taskflow;

  auto A = taskflow.emplace([] () { std::cout << "1\n"; }).name("A");
  auto B = taskflow.emplace([] () { std::cout << "2\n"; }).name("B");
  auto C = taskflow.emplace([] () { std::cout << "3\n"; }).name("C");
  auto D = taskflow.emplace([] () { std::cout << "4\n"; }).name("D");
  auto E = taskflow.emplace([] () { std::cout << "5\n"; }).name("E");
  auto F = taskflow.emplace([] () { std::cout << "6\n"; }).name("F");
  auto G = taskflow.emplace([] () { std::cout << "7\n"; }).name("G");
  auto H = taskflow.emplace([] () { std::cout << "8\n"; }).name("H");

  // create a default observer
  std::shared_ptr<MyObserver> observer = executor.make_observer<MyObserver>("MyObserver");

  // run the taskflow
  executor.run(taskflow).get();
  
  // remove the observer (optional)
  executor.remove_observer(std::move(observer));

  return 0;
}

