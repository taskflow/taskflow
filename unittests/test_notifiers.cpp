#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// ----------------------------------------------------------------------------
// NonblockingNotifier: no_missing_notify_all
//   - In each round, all threads call prepare_wait()
//   - Main thread calls notify_all() while they are between prepare and commit
//   - Then threads commit_wait(); they must NOT block (no lost wakeup)
//   - At end: completed == R*N and num_waiters() == 0
// ----------------------------------------------------------------------------

// consider notify_one and notify_n variants as well
// try mixing the three func together
// try permutatutions within the functions

template <typename T>
void no_missing_notify_all(size_t N) {

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

TEST_CASE("NonblockingNotifier.no_missing_notify_all.1threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_all.2threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_all.4threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.notify_before_commit.5threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_all.8threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_all.15threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_all.31threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: no_missing_notify_one
//   - When all threads are in prepare_wait() and commit_wait(), one notify must ensure:
//     - atleast one thread wakes up immediately 
//     - it does not park
//     - it does not wait for another notify
// ----------------------------------------------------------------------------

template <typename T>
void no_missing_notify_one(size_t N) {

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

TEST_CASE("NonblockingNotifier.no_missing_notify_one.1threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.2threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.4threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.5threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.8threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.15threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_one.31threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// NonblockingNotifier: no_missing_notify_n
//   - In each round, all threads call prepare_wait() and pause in the window
//     between prepare_wait and commit_wait.
//   - Main calls notify_n(k) in that window.
//   - At least k threads must be able to finish commit_wait without any
//     additional notify (i.e., without parking forever).
//   - Then main calls notify_all() to release any remaining waiters so the
//     round can complete.
// ----------------------------------------------------------------------------

template <typename T>
void no_missing_notify_n(size_t N, uint32_t seed = 12345) {

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
  std::uniform_int_distribution<size_t> dist(0, N);

  for(size_t r = 0; r < R; ++r) {
    armed.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r+1, std::memory_order_release);

    while(armed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    size_t k = dist(rng);

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

TEST_CASE("NonblockingNotifier.no_missing_notify_n.1threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.2threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.4threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.5threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.8threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.15threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_n.31threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::NonblockingNotifier>(31);
}

// ----------------------------------------------------------------------------
// mixed test of notify_one, notify_n, and notify_all
// ----------------------------------------------------------------------------

template <typename T>
void no_missing_notify_x(size_t N, uint32_t seed = 12345) {

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
  std::uniform_int_distribution<size_t> dist(0, N);

  for(size_t r = 0; r < R; ++r) {
    armed.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r+1, std::memory_order_release);

    while(armed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    size_t k = dist(rng);

    if(k == 1) {
      notifier.notify_one();
    }
    else if(k == N) {
      notifier.notify_all();
    }
    else {
      notifier.notify_n(k);
    }


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

TEST_CASE("NonblockingNotifier.no_missing_notify_x.1threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.2threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.4threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(4);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.5threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.8threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.15threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(15);
}

TEST_CASE("NonblockingNotifier.no_missing_notify_x.31threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::NonblockingNotifier>(31);
}

