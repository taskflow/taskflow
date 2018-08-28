// Simple example showing use of the C++14-compatible threadpool
// implementation. The threadpool (alone) may be used when you have
// no inter-task dependencies to express, and when you require
// C++14 for a project.
//
// NOTE: if you are using a fully C++17-compliant compiler, you should
//       be including <taskflow/taskflow.hpp> rather than this file!
//
// Compile with VS2017 using:
//   cl /std:c++14 /W4 /EHsc /O2 /Ipath-to-taskflow threadpool_cxx14.cpp
// Compile with clang (including Apple LLVM version 9.0.0) using:
//   clang++ -std=c++14 -Wall -O3 -Ipath-to-taskflow threadpool_cxx14.cpp -o threadpool_cxx14
// (for gcc, replace clang++ with g++)

#include <taskflow/threadpool/threadpool_cxx14.hpp>
#include <iostream>
#include <random>

// The "noinline" directive is used to prevent the compiler from
// optimizing out the entire calculation, ensuring it actually does
// its (fake) work when we want it to (for testing), at runtime.
#ifdef _WIN32
__declspec(noinline)
#else
__attribute__((noinline))
#endif // _WIN32
int64_t compute(int64_t x, int64_t i, int64_t r)
{
  return x + i + r;
}

int main()
{
  auto numThreads = std::max(1u, std::thread::hardware_concurrency());
  tf::Threadpool tp(numThreads);

  std::default_random_engine gen(17);
  std::uniform_int_distribution<int64_t> d{ 0, 10000 };

  std::vector<std::future<int64_t>> intFutures;
  std::vector<std::future<void>> voidFutures;
  auto start = std::chrono::steady_clock::now();
  for (auto i = 0; i < 2000; ++i) {
    auto r = d(gen);
    auto future = tp.async(
      [i, r]() {
        int64_t sum = 0;
        for (int64_t x = 0; x < 10000000; ++x) {
          sum += compute(x, i, r);
        }
        return sum;
    });
    intFutures.push_back(std::move(future));
    voidFutures.push_back(tp.async([]() {
      // Simulate a very small task that returns no value.
      std::this_thread::sleep_for(std::chrono::microseconds(5));
    }));
  }
  std::cout << "Scheduled " << intFutures.size() + voidFutures.size() << " tasks...\n";

  // Wait for them all to finish (get the results)
  int64_t sum = 0;
  for (auto& f : intFutures) {
    sum += f.get();
  }
  for (auto& f : voidFutures) {
    f.get();
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "...ran in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    << " milliseconds\n";
  std::cout << "Remaining scheduled tasks: " << tp.num_tasks() << "\tWorker threads: " << tp.num_workers() << '\n';
  std::cout << intFutures.size() << " int futures summed to: " << sum << '\n';
  std::cout << "(also ran " << voidFutures.size() << " void futures)\n";

  return 0;
}
