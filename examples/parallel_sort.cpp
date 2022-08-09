// This program demonstrates how to sort a vector of strings
// in parallel using tf::Taskflow::sort and compares it against
// the sequential sort std::sort.
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/sort.hpp>

// generate a random string
std::string random_string(size_t len) {

  std::string tmp_s;
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  std::default_random_engine eng{std::random_device{}()};
  std::uniform_int_distribution<int> dist(1, 100000);

  tmp_s.reserve(len);

  for (size_t i = 0; i < len; ++i) {
    tmp_s += alphanum[dist(eng) % (sizeof(alphanum) - 1)];
  }

  return tmp_s;
}

// generate a vector of random strings
std::vector<std::string> random_strings() {
  std::vector<std::string> strings(1000000);
  std::cout << "generating random strings ...\n";
  for(auto& str : strings) {
    str = random_string(32);
  }
  return strings;
}

// Function: main
int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./parallel_sort s|p" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // sequential sort
  if(argv[1][0] == 's') {
    auto strings = random_strings();
    std::cout << "std::sort ... ";
    auto beg = std::chrono::steady_clock::now();
    std::sort(strings.begin(), strings.end());
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-beg).count()
              << " ms\n";
  }
  // parallel sort
  else if(argv[1][0] == 'p') {
    auto strings = random_strings();
    std::cout << "Taskflow Parallel Sort ... ";
    auto beg = std::chrono::steady_clock::now();
    {
      tf::Taskflow taskflow;
      tf::Executor executor;
      taskflow.sort(strings.begin(), strings.end());
      executor.run(taskflow).wait();
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-beg).count()
              << " ms\n";
  }
  else {
    std::cerr << "uncognized method character '" << argv[1][0] << "'\n";
    std::exit(EXIT_FAILURE);
  }

  return 0;
}






