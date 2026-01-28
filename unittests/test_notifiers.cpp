#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_one
// ----------------------------------------------------------------------------

template <typename T>
void notify_one(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<bool> stop(false);
  std::atomic<size_t> num_prewaiters(0);
  std::atomic<size_t> num_wakeups(0);
  size_t R = 100*(N+1);

  std::vector<std::thread> threads;
  for(size_t i=0; i<N; ++i) {
    threads.emplace_back([&, i](){
      while(stop == false) {
        notifier.prepare_wait(i);
        num_prewaiters.fetch_add(1, std::memory_order_relaxed);
        if(stop == true) {
          notifier.cancel_wait(i);
          return;
        }
        notifier.commit_wait(i);
        num_wakeups.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // wait until all N threads enter the 2PC loop
  while(num_prewaiters != N);
  REQUIRE(num_wakeups == 0);
  
  size_t expected_num_prewaiters = N;
  size_t expected_num_wakeups = 0;

  for(size_t r=1; r<=R; ++r) {
    notifier.notify_one();
    expected_num_prewaiters += 1;
    expected_num_wakeups += 1;
    while(num_prewaiters != expected_num_prewaiters);  // wait until the notify_one takes effect
    REQUIRE(num_wakeups == expected_num_wakeups);
  }

  // now request stop
  stop = true;
  for(size_t n=0; n<N; ++n) {
    notifier.notify_one();
  }

  for(auto& thread : threads) {
    thread.join();
  }
}

TEST_CASE("NonblockingNotifier.notify_one.1thread" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.notify_one.2threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.notify_one.3threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(3);
}

TEST_CASE("NonblockingNotifier.notify_one.4threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.notify_one.5threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.notify_one.6threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(6);
}

TEST_CASE("NonblockingNotifier.notify_one.7threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(7);
}

TEST_CASE("NonblockingNotifier.notify_one.8threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.notify_one.15threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.notify_one.31threads" * doctest::timeout(300)) {
  notify_one<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_all
// ----------------------------------------------------------------------------

template <typename T>
void notify_all(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<bool> stop(false);
  std::atomic<size_t> num_prewaiters(0);
  std::atomic<size_t> num_wakeups(0);
  size_t R = 100*(N+1);

  std::vector<std::thread> threads;
  for(size_t i=0; i<N; ++i) {
    threads.emplace_back([&, i](){
      while(stop == false) {
        notifier.prepare_wait(i);
        num_prewaiters.fetch_add(1, std::memory_order_relaxed);
        if(stop == true) {
          notifier.cancel_wait(i);
          return;
        }
        notifier.commit_wait(i);
        num_wakeups.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // wait until all N threads enter the 2PC loop
  while(num_prewaiters != N);
  REQUIRE(num_wakeups == 0);
  
  size_t expected_num_prewaiters = N;
  size_t expected_num_wakeups = 0;

  for(size_t r=1; r<=R; ++r) {
    notifier.notify_all();
    expected_num_prewaiters += N;
    expected_num_wakeups += N;
    while(num_prewaiters != expected_num_prewaiters);  // wait until the notify_all takes effect
    REQUIRE(num_wakeups == expected_num_wakeups);
  }

  // now request stop
  stop = true;
  notifier.notify_all();

  for(auto& thread : threads) {
    thread.join();
  }
}

TEST_CASE("NonblockingNotifier.notify_all.1thread" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.notify_all.2threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.notify_all.3threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(3);
}

TEST_CASE("NonblockingNotifier.notify_all.4threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.notify_all.5threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.notify_all.6threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(6);
}

TEST_CASE("NonblockingNotifier.notify_all.7threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(7);
}

TEST_CASE("NonblockingNotifier.notify_all.8threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.notify_all.15threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.notify_all.31threads" * doctest::timeout(300)) {
  notify_all<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_n
// ----------------------------------------------------------------------------

template <typename T>
void notify_n(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<bool> stop(false);
  std::atomic<size_t> num_prewaiters(0);
  std::atomic<size_t> num_wakeups(0);
  size_t R = 100*(N+1);

  std::vector<std::thread> threads;
  for(size_t i=0; i<N; ++i) {
    threads.emplace_back([&, i](){
      while(stop == false) {
        notifier.prepare_wait(i);
        num_prewaiters.fetch_add(1, std::memory_order_relaxed);
        if(stop == true) {
          notifier.cancel_wait(i);
          return;
        }
        notifier.commit_wait(i);
        num_wakeups.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // wait until all N threads enter the 2PC loop
  while(num_prewaiters != N);
  REQUIRE(num_wakeups == 0);

  size_t expected_num_prewaiters = N;
  size_t expected_num_wakeups = 0;

  for(size_t r=1; r<=R; ++r) {
    notifier.notify_n(r%N);
    expected_num_prewaiters += (r%N);
    expected_num_wakeups += (r%N);
    while(num_prewaiters != expected_num_prewaiters);  // wait until the notify_n takes effect
    REQUIRE(num_wakeups == expected_num_wakeups);
  }

  // now request stop
  stop = true;
  notifier.notify_n(N);

  for(auto& thread : threads) {
    thread.join();
  }
}

TEST_CASE("NonblockingNotifier.notify_n.1thread" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.notify_n.2threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.notify_n.3threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(3);
}

TEST_CASE("NonblockingNotifier.notify_n.4threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.notify_n.5threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.notify_n.6threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(6);
}

TEST_CASE("NonblockingNotifier.notify_n.7threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(7);
}

TEST_CASE("NonblockingNotifier.notify_n.8threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.notify_n.15threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.notify_n.31threads" * doctest::timeout(300)) {
  notify_n<tf::NonblockingNotifier>(31);
}


// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_all_before_commit_rounds
//   - In each round, all threads call prepare_wait()
//   - Main thread calls notify_all() while they are between prepare and commit
//   - Then threads commit_wait(); they must NOT block (no lost wakeup)
//   - At end: completed == R*N and num_waiters() == 0
// ----------------------------------------------------------------------------

// consider notify_one and notify_n variants as well
// try mixing the three func together
// try permutatutions within the functions

template <typename T>
void notify_all_before_commit_rounds(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if(N >= 31) R = 1 * (N + 1);   

  std::atomic<size_t> armed(0);
  std::atomic<size_t> completed(0);
  std::atomic<size_t> round(0);
  std::atomic<bool> stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for(size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {

      size_t local_round = 0;

      while(!stop.load(std::memory_order_relaxed)) {

        while(round.load(std::memory_order_acquire) == local_round &&!stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if(stop.load(std::memory_order_relaxed)) {
          break;
        }

        notifier.prepare_wait(i);
        armed.fetch_add(1, std::memory_order_relaxed);

        while(armed.load(std::memory_order_acquire) < (local_round + 1) * N && !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        notifier.commit_wait(i);
        completed.fetch_add(1, std::memory_order_relaxed);

        local_round++;
      }
    });
  }

  for(size_t r = 0; r < R; ++r) {
    round.fetch_add(1, std::memory_order_release);

    while(armed.load(std::memory_order_acquire) != (r + 1) * N) {
      std::this_thread::yield();
    }

    notifier.notify_all();
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for(auto& t : threads) t.join();

  REQUIRE(completed.load() == R * N);
  REQUIRE(notifier.num_waiters() == 0);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.1threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(1);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.2threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(2);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.4threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(4);
}
TEST_CASE("NonblockingNotifier.notify_before_commit_rounds.5threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(5);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.8threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(8);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.15threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(15);
}
TEST_CASE("NonblockingNotifier.notify_all_before_commit_rounds.31threads" * doctest::timeout(300)) {
  notify_all_before_commit_rounds<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_one_before_commit_rounds
//   - When all threads are in prepare_wait() and commit_wait(), one notify must ensure:
//     - atleast one thread wakes up immediately 
//     - it does not park
//     - it does not wait for another notify
// ----------------------------------------------------------------------------

template <typename T>
void notify_one_before_commit_rounds(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if(N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> armed(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool> stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for(size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while(!stop.load(std::memory_order_relaxed)) {

        while(round.load(std::memory_order_acquire) <= local_round &&
              !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if(stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        armed.fetch_add(1, std::memory_order_release);

        // Everyone commits; some may park, that is allowed.
        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  for(size_t r = 0; r < R; ++r) {

    armed.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    // Wait until all threads are between prepare and commit
    while(armed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    notifier.notify_one();

    while(committed.load() == 0) std::this_thread::yield();  

    notifier.notify_all();

    while(committed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for(auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}


TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.1threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(1);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.2threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(2);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.4threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(4);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.5threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(5);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.8threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(8);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.15threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(15);
}
TEST_CASE("NonblockingNotifier.notify_one_before_commit_rounds.31threads" * doctest::timeout(300)) {
  notify_one_before_commit_rounds<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: notify_n_before_commit_rounds
//   - In each round, all threads call prepare_wait() and pause in the window
//     between prepare_wait and commit_wait.
//   - Main calls notify_n(k) in that window.
//   - At least k threads must be able to finish commit_wait without any
//     additional notify (i.e., without parking forever).
//   - Then main calls notify_all() to release any remaining waiters so the
//     round can complete.
// ----------------------------------------------------------------------------


template <typename T>
void notify_n_before_commit_rounds(size_t N, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if(N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> armed(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool> stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for(size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while(!stop.load(std::memory_order_relaxed)) {

        while(round.load(std::memory_order_acquire) <= local_round &&
              !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if(stop.load(std::memory_order_relaxed)){
          break;
        }

        notifier.prepare_wait(i);
        armed.fetch_add(1, std::memory_order_release);

        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  std::mt19937 rng(seed);

  for(size_t r = 0; r < R; ++r) {
    armed.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r+1, std::memory_order_release);

    while(armed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    size_t k;
    if(N == 1) {
      k = 1;
    }
    else {
      std::uniform_int_distribution<size_t> dist(1, N);
      k = dist(rng);

      if((r % 7) == 0) k = 1;      
      if((r % 11) == 0) k = N;     
      if((r % 13) == 0) k = N/2;  
      if(k == 0) k = 1;
    }

    notifier.notify_n(k);
    while(committed.load(std::memory_order_acquire) < k) {
      std::this_thread::yield();
    }

    notifier.notify_all();

    while(committed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for(auto& t : threads) {
    t.join();
  }

  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.1threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(1);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.2threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(2);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.4threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(4);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.5threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(5);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.8threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(8);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.15threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(15);
}
TEST_CASE("NonblockingNotifier.notify_n_before_commit_rounds.31threads" * doctest::timeout(300)) {
  notify_n_before_commit_rounds<tf::NonblockingNotifier>(31);
}
