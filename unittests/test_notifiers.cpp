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

  std::atomic<size_t> prepared(0);
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
        prepared.fetch_add(1, std::memory_order_relaxed);

        while(prepared.load(std::memory_order_acquire) < (local_round + 1) * N && !stop.load(std::memory_order_relaxed)) {
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

    while(prepared.load(std::memory_order_acquire) != (r + 1) * N) {
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
  std::atomic<size_t> prepared(0);
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
        prepared.fetch_add(1, std::memory_order_release);

        // Everyone commits; some may park, that is allowed.
        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  for(size_t r = 0; r < R; ++r) {

    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    // Wait until all threads are between prepare and commit
    while(prepared.load(std::memory_order_acquire) != N) {
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
  std::atomic<size_t> prepared(0);
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
        prepared.fetch_add(1, std::memory_order_release);

        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, N);

  for(size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r+1, std::memory_order_release);

    while(prepared.load(std::memory_order_acquire) != N) {
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
  std::atomic<size_t> prepared(0);
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
        prepared.fetch_add(1, std::memory_order_release);

        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, N);

  for(size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r+1, std::memory_order_release);

    while(prepared.load(std::memory_order_acquire) != N) {
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
// ----------------------------------------------------------------------------
// NonblockingNotifier: no_missing_cancel_wait
//   - Verifies that threads can "change their mind" about sleeping after
//     calling prepare_wait().
//   - In each round, every thread:
//      1. Calls prepare_wait(i) to register intent to sleep.
//      2. Waits for the main thread to publish a per-thread predicate.
//      3. If predicate is true, calls cancel_wait(i) and proceeds without
//         becoming a waiter.
//      4. Otherwise calls commit_wait(i) (may block until a notify).
//   - The test ensures:
//      * cancel_wait properly cleans up internal state (no ghost waiters)
//      * commit_wait + notifications complete the round
//      * num_waiters() returns to 0 at the end
// ----------------------------------------------------------------------------

template <typename T>
void no_missing_cancel_wait(size_t N, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if(N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> canceled(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool> stop(false);
  std::atomic<bool> go(false);  // workers spin on this so they do not read has_work[i] too early

  std::vector<std::atomic<bool>> has_work(N);
  for(size_t i = 0; i < N; ++i) {
    has_work[i].store(false, std::memory_order_relaxed);
  }

  std::vector<std::thread> threads;
  threads.reserve(N);

  for(size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {

      size_t local_round = 0;

      while(!stop.load(std::memory_order_relaxed)) {

        // Wait for main to advance the round
        while(round.load(std::memory_order_acquire) <= local_round &&
              !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if(stop.load(std::memory_order_relaxed)) {
          break;
        }

        // Register intent to sleep
        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        // Wait until main publishes per-thread predicate
        while(!go.load(std::memory_order_acquire) &&
              !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if(stop.load(std::memory_order_relaxed)) {
          break;
        }

        // Either cancel or commit
        if(has_work[i].load(std::memory_order_acquire)) {
          notifier.cancel_wait(i);
          canceled.fetch_add(1, std::memory_order_release);
        }
        else {
          notifier.commit_wait(i);
          committed.fetch_add(1, std::memory_order_release);
        }

        local_round++;
      }
    });
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dice(0, 1);

  for(size_t r = 0; r < R; ++r) {

    // Reset per-round counters/gates
    prepared.store(0, std::memory_order_relaxed);
    canceled.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);
    go.store(false, std::memory_order_release);

    // Reset predicates
    for(size_t i = 0; i < N; ++i) {
      has_work[i].store(false, std::memory_order_relaxed);
    }

    // Start round
    round.store(r + 1, std::memory_order_release);

    // Wait until every worker has reached prepare_wait(i)
    while(prepared.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    // publish which threads should cancel
    size_t expected_cancels = 0;
    for(size_t i = 0; i < N; ++i) {
      if(dice(rng)) {
        has_work[i].store(true, std::memory_order_release);
        expected_cancels++;
      }
    }
    size_t expected_commits = N - expected_cancels;

    // Let workers proceed to cancel/commit
    go.store(true, std::memory_order_release);

    // Wait until all cancelers have canceled (committers may still be blocked)
    while(canceled.load(std::memory_order_acquire) != expected_cancels) {
      std::this_thread::yield();
    }

    // Wait until all committers are truly in the waiter set (or there are none)
    while(notifier.num_waiters() != expected_commits) {
      std::this_thread::yield();
    }

    if(expected_commits > 0) {
      std::uniform_int_distribution<size_t> pickX(0, expected_commits);
      size_t X = pickX(rng);

      size_t first = expected_commits - X;   // in [0, expected_commits]
      notifier.notify_n(first);

      for(size_t j = 0; j < X; ++j) {
        notifier.notify_one();
      }
    }

    // ensure the round can always complete
    notifier.notify_all();

    // Wait until every worker either canceled or returned from commit_wait
    while((canceled.load(std::memory_order_acquire) +
           committed.load(std::memory_order_acquire)) != N) {
      std::this_thread::yield();
    }

    // After all workers finished, there should be no lingering waiters
    REQUIRE(notifier.num_waiters() == 0);
  }

  // Shutdown
  stop.store(true, std::memory_order_release);
  go.store(true, std::memory_order_release);
  notifier.notify_all();

  for(auto& t : threads) {
    t.join();
  }

  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.1threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(1);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.2threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(2);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.4threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(4);
}
TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.5threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(5);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.8threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(8);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.16threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(16);
}

TEST_CASE("NonblockingNotifier.no_missing_cancel_wait.31threads"* doctest::timeout(300)){
  no_missing_cancel_wait<tf::NonblockingNotifier>(31);
}



// TODO: 02/12 by TW - try to refactor your functions using compile-time switch
// so we can remove the redundancy in the test cases and make it easier to maintain. Something like:
enum class NotificationType { ONE, N, ALL };

// ----------------------------------------------------------------------------
// no_missing_notifications (compile-time switch)
// - Workers: prepare_wait(i) -> (maybe cancel_wait(i)) -> commit_wait(i).
// - M notifier threads hammer one of: notify_one / notify_n / notify_all.
// - End check: progress continues and num_waiters() == 0 each round and at end.
// ----------------------------------------------------------------------------
template <typename T, NotificationType NT>
void no_missing_notifications(size_t N, size_t M = 4, uint32_t seed = 12345){

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if(N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> canceled(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool> stop(false);
  std::atomic<bool> go(false);

  std::vector<std::atomic<bool>> has_work(N);
  for(size_t i = 0; i < N; ++i){
    has_work[i].store(false, std::memory_order_relaxed);
  }

  // Worker threads
  std::vector<std::thread> workers;
  workers.reserve(N);

  for(size_t i = 0; i < N; ++i){
    workers.emplace_back([&, i](){
      size_t local_round = 0;

      while(!stop.load(std::memory_order_relaxed)){

        while(round.load(std::memory_order_acquire) <= local_round &&
              !stop.load(std::memory_order_relaxed)){
          std::this_thread::yield();
        }
        if(stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        while(!go.load(std::memory_order_acquire) &&
              !stop.load(std::memory_order_relaxed)){
          std::this_thread::yield();
        }
        if(stop.load(std::memory_order_relaxed)) break;

        if(has_work[i].load(std::memory_order_acquire)){
          notifier.cancel_wait(i);
          canceled.fetch_add(1, std::memory_order_release);
        }
        else{
          notifier.commit_wait(i);
          committed.fetch_add(1, std::memory_order_release);
        }

        local_round++;
      }
    });
  }

  // Notifier threads
  std::vector<std::thread> notifiers;
  notifiers.reserve(M);

  if constexpr (NT == NotificationType::ONE){
    for(size_t t = 0; t < M; ++t){
      notifiers.emplace_back([&](){
        while(!stop.load(std::memory_order_relaxed)){
          notifier.notify_one();
          std::this_thread::yield();
        }
      });
    }
  }
  else if constexpr (NT == NotificationType::N){
    for(size_t t = 0; t < M; ++t){
      notifiers.emplace_back([&, t](){
        std::mt19937 trng(seed + static_cast<uint32_t>(t + 1));
        std::uniform_int_distribution<size_t> dist(0, N);

        while(!stop.load(std::memory_order_relaxed)){
          notifier.notify_n(dist(trng));
          std::this_thread::yield();
        }
      });
    }
  }
  else if constexpr (NT == NotificationType::ALL){
    for(size_t t = 0; t < M; ++t){
      notifiers.emplace_back([&, t](){
        std::mt19937 trng(seed + static_cast<uint32_t>(777 + t));
        std::uniform_int_distribution<int> burst(1, 8);

        while(!stop.load(std::memory_order_relaxed)){
          int b = burst(trng);
          for(int i = 0; i < b; ++i){
            notifier.notify_all();
          }
          std::this_thread::yield();
        }
      });
    }
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dice(0, 1);

  for(size_t r = 0; r < R; ++r){

    prepared.store(0, std::memory_order_relaxed);
    canceled.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);
    go.store(false, std::memory_order_release);

    for(size_t i = 0; i < N; ++i){
      has_work[i].store(false, std::memory_order_relaxed);
    }

    round.store(r + 1, std::memory_order_release);

    while(prepared.load(std::memory_order_acquire) != N){
      std::this_thread::yield();
    }

    for(size_t i = 0; i < N; ++i){
      if(dice(rng)){
        has_work[i].store(true, std::memory_order_release);
      }
    }

    go.store(true, std::memory_order_release);

    while((canceled.load(std::memory_order_acquire) +
           committed.load(std::memory_order_acquire)) != N){
      std::this_thread::yield();
    }

    REQUIRE(notifier.num_waiters() == 0);
  }

  // Shutdown
  stop.store(true, std::memory_order_release);
  go.store(true, std::memory_order_release);
  notifier.notify_all();

  for(auto& t : workers) t.join();
  for(auto& t : notifiers) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ONE
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.1threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(1);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.2threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(2);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.4threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(4);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.5threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(5);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.8threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(8);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.16threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(16);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ones.31threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ONE>(31);
}

// N
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.1threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(1);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.2threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(2);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.4threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(4);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.5threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(5);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.8threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(8);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.16threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(16);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_ns.31threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::N>(31);
}

// ALL
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.1threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(1);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.2threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(2);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.4threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(4);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.5threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(5);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.8threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(8);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.16threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(16);
}
TEST_CASE("NonblockingNotifier.no_missing_notify_alls.31threads"* doctest::timeout(300)){
  no_missing_notifications<tf::NonblockingNotifier, NotificationType::ALL>(31);
}


// ============================================================================
// AtomicNotifier: reuse the same test helpers
// ============================================================================

// ----------------------------------------------------------------------------
// AtomicNotifier: no_missing_notify_all
// ----------------------------------------------------------------------------
TEST_CASE("AtomicNotifier.no_missing_notify_all.1threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.2threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.4threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.5threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.8threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.15threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(15);
}
TEST_CASE("AtomicNotifier.no_missing_notify_all.31threads" * doctest::timeout(300)) {
  no_missing_notify_all<tf::AtomicNotifier>(31);
}

// ----------------------------------------------------------------------------
// AtomicNotifier: no_missing_notify_one
// ----------------------------------------------------------------------------
TEST_CASE("AtomicNotifier.no_missing_notify_one.1threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.2threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.4threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.5threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.8threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.15threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(15);
}
TEST_CASE("AtomicNotifier.no_missing_notify_one.31threads" * doctest::timeout(300)) {
  no_missing_notify_one<tf::AtomicNotifier>(31);
}

// ----------------------------------------------------------------------------
// AtomicNotifier: no_missing_notify_n
// ----------------------------------------------------------------------------
TEST_CASE("AtomicNotifier.no_missing_notify_n.1threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.2threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.4threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.5threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.8threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.15threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(15);
}
TEST_CASE("AtomicNotifier.no_missing_notify_n.31threads" * doctest::timeout(300)) {
  no_missing_notify_n<tf::AtomicNotifier>(31);
}

// ----------------------------------------------------------------------------
// AtomicNotifier: mixed notify_one/notify_n/notify_all
// ----------------------------------------------------------------------------
TEST_CASE("AtomicNotifier.no_missing_notify_x.1threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.2threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.4threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.5threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.8threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.15threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(15);
}
TEST_CASE("AtomicNotifier.no_missing_notify_x.31threads" * doctest::timeout(300)) {
  no_missing_notify_x<tf::AtomicNotifier>(31);
}

// ----------------------------------------------------------------------------
// AtomicNotifier: no_missing_cancel_wait
// ----------------------------------------------------------------------------
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.1threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(1);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.2threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(2);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.4threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(4);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.5threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(5);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.8threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(8);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.16threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(16);
}
TEST_CASE("AtomicNotifier.no_missing_cancel_wait.31threads" * doctest::timeout(300)) {
  no_missing_cancel_wait<tf::AtomicNotifier>(31);
}

// ----------------------------------------------------------------------------
// AtomicNotifier: compile-time switch stress (no_missing_notifications)
// ----------------------------------------------------------------------------

// ONE
TEST_CASE("AtomicNotifier.no_missing_notify_ones.1threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.2threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.4threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.5threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.8threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.16threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(16);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ones.31threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ONE>(31);
}

// N
TEST_CASE("AtomicNotifier.no_missing_notify_ns.1threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.2threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.4threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.5threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.8threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.16threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(16);
}
TEST_CASE("AtomicNotifier.no_missing_notify_ns.31threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::N>(31);
}

// ALL
TEST_CASE("AtomicNotifier.no_missing_notify_alls.1threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(1);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.2threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(2);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.4threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(4);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.5threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(5);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.8threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(8);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.16threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(16);
}
TEST_CASE("AtomicNotifier.no_missing_notify_alls.31threads" * doctest::timeout(300)) {
  no_missing_notifications<tf::AtomicNotifier, NotificationType::ALL>(31);
}