#include <taskflow/taskflow.hpp>

void syncLog(std::string const& msg) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cout << msg << '\n';
}

void grow(tf::SubflowBuilder& subflow, uint64_t depth) {
  syncLog("Depth: " + std::to_string(depth));
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  if(depth < 3) {
    subflow.silent_emplace(
      [depth](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); },
      [depth](tf::SubflowBuilder& subsubflow){ grow(subsubflow, depth+1); });
    subflow.detach();
  }
}

int main(int argc, char *argv[]) {
  tf::Taskflow mainTaskFlow;
  mainTaskFlow.silent_emplace([](tf::SubflowBuilder& subflow){grow(subflow, 0);});
  mainTaskFlow.wait_for_all();

  return 0;
}
