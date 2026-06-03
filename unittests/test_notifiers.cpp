#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

#include <atomic>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

// ============================================================================
// Shared helpers
// ============================================================================

static inline void tiny_jitter(std::mt19937& rng) {
  std::uniform_int_distribution<int> pick(0, 9);
  int x = pick(rng);
  if (x < 4) {
    std::this_thread::yield();
  } else if (x < 8) {
    volatile int sink = 0;
    int spins = 20 + (pick(rng) * 30);
    for (int i = 0; i < spins; ++i) sink += i;
    (void)sink;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(200));
  }
}

enum class NotificationType { ONE, N, ALL };

// ============================================================================
// no_missing_notify_all
//   Each round: all N threads prepare_wait(), pause in the window between
//   prepare and commit until all threads are ready, then commit_wait().
//   Main calls notify_all() in that window. All commit_wait() calls must
//   return without blocking (no lost wakeup).
//   End check: completed == R*N, num_waiters() == 0.
// ============================================================================

template <typename T>
void no_missing_notify_all(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> prepared(0);
  std::atomic<size_t> completed(0);
  std::atomic<size_t> round(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) == local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_relaxed);

        while (prepared.load(std::memory_order_acquire) < (local_round + 1) * N &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        notifier.commit_wait(i);
        completed.fetch_add(1, std::memory_order_relaxed);

        local_round++;
      }
    });
  }

  for (size_t r = 0; r < R; ++r) {
    round.fetch_add(1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != (r + 1) * N) {
      std::this_thread::yield();
    }

    notifier.notify_all();
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : threads) t.join();

  REQUIRE(completed.load() == R * N);
  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// no_missing_notify_one
//   Each round: all N threads prepare_wait() then commit_wait().
//   Main calls notify_one() and verifies at least 1 thread unblocks,
//   then notify_all() drains the rest. num_waiters() == 0 at end.
// ============================================================================

template <typename T>
void no_missing_notify_one(size_t N) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) <= local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        notifier.commit_wait(i);
        committed.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  for (size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    notifier.notify_one();

    // At least one thread must unblock from commit_wait.
    while (committed.load() == 0) std::this_thread::yield();

    // Drain any remaining waiters.
    notifier.notify_all();

    while (committed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// no_missing_notify_n
//   Each round: all N threads prepare then commit_wait. Main calls notify_n(k)
//   where k is random in [0, N], verifies at least k unblock, then notify_all
//   drains the rest. num_waiters() == 0 at end.
// ============================================================================

template <typename T>
void no_missing_notify_n(size_t N, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) <= local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

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

  for (size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    size_t k = dist(rng);
    notifier.notify_n(k);

    while (committed.load(std::memory_order_acquire) < k) {
      std::this_thread::yield();
    }

    notifier.notify_all();

    while (committed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// no_missing_notify_x  (mixed: notify_one / notify_n / notify_all)
//   Same structure as no_missing_notify_n but randomly picks notify variant
//   each round. Verifies all variants can coexist without losing wakeups.
// ============================================================================

template <typename T>
void no_missing_notify_x(size_t N, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) <= local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

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

  for (size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    size_t k = dist(rng);
    if (k == 1) {
      notifier.notify_one();
    } else if (k == N) {
      notifier.notify_all();
    } else {
      notifier.notify_n(k);
    }

    while (committed.load(std::memory_order_acquire) < k) {
      std::this_thread::yield();
    }

    notifier.notify_all();

    while (committed.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// no_missing_cancel_wait
//   Each round: every thread calls prepare_wait(). Main randomly assigns
//   per-thread has_work flag. Threads with has_work cancel, rest commit.
//   Verifies cancel_wait cleans up state (no ghost waiters) and that
//   commit_wait + notifications complete the round. num_waiters() == 0
//   after every round and at end.
// ============================================================================

template <typename T>
void no_missing_cancel_wait(size_t N, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> canceled(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool>   stop(false);
  std::atomic<bool>   go(false);

  std::vector<std::atomic<bool>> has_work(N);
  for (size_t i = 0; i < N; ++i) has_work[i].store(false, std::memory_order_relaxed);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) <= local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        while (!go.load(std::memory_order_acquire) &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

        if (has_work[i].load(std::memory_order_acquire)) {
          notifier.cancel_wait(i);
          canceled.fetch_add(1, std::memory_order_release);
        } else {
          notifier.commit_wait(i);
          committed.fetch_add(1, std::memory_order_release);
        }

        local_round++;
      }
    });
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dice(0, 1);

  for (size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    canceled.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);
    go.store(false, std::memory_order_release);

    for (size_t i = 0; i < N; ++i) has_work[i].store(false, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) std::this_thread::yield();

    size_t expected_cancels = 0;
    for (size_t i = 0; i < N; ++i) {
      if (dice(rng)) {
        has_work[i].store(true, std::memory_order_release);
        ++expected_cancels;
      }
    }
    size_t expected_commits = N - expected_cancels;

    go.store(true, std::memory_order_release);

    while (canceled.load(std::memory_order_acquire) != expected_cancels) std::this_thread::yield();

    while (notifier.num_waiters() != expected_commits) std::this_thread::yield();

    if (expected_commits > 0) {
      std::uniform_int_distribution<size_t> pickX(0, expected_commits);
      size_t X    = pickX(rng);
      size_t first = expected_commits - X;
      notifier.notify_n(first);
      for (size_t j = 0; j < X; ++j) notifier.notify_one();
    }

    // Ensure the round always completes.
    notifier.notify_all();

    while ((canceled.load(std::memory_order_acquire) +
            committed.load(std::memory_order_acquire)) != N) {
      std::this_thread::yield();
    }

    REQUIRE(notifier.num_waiters() == 0);
  }

  stop.store(true, std::memory_order_release);
  go.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// no_missing_notifications  (compile-time switch: ONE / N / ALL)
//   M concurrent notifier threads continuously hammer one notify variant.
//   N worker threads follow prepare -> cancel_or_commit protocol.
//   Round ends when every worker has either canceled or returned from commit.
//   num_waiters() == 0 after every round.
// ============================================================================

template <typename T, NotificationType NT>
void no_missing_notifications(size_t N, size_t M = 4, uint32_t seed = 12345) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  size_t R = 20 * (N + 1);
  if (N >= 31) R = 1 * (N + 1);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> canceled(0);
  std::atomic<size_t> committed(0);
  std::atomic<bool>   stop(false);
  std::atomic<bool>   go(false);

  std::vector<std::atomic<bool>> has_work(N);
  for (size_t i = 0; i < N; ++i) has_work[i].store(false, std::memory_order_relaxed);

  std::vector<std::thread> workers;
  workers.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    workers.emplace_back([&, i]() {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) <= local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed)) break;

        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        while (!go.load(std::memory_order_acquire) &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed)) break;

        if (has_work[i].load(std::memory_order_acquire)) {
          notifier.cancel_wait(i);
          canceled.fetch_add(1, std::memory_order_release);
        } else {
          notifier.commit_wait(i);
          committed.fetch_add(1, std::memory_order_release);
        }

        local_round++;
      }
    });
  }

  std::vector<std::thread> notifiers;
  notifiers.reserve(M);

  if constexpr (NT == NotificationType::ONE) {
    for (size_t t = 0; t < M; ++t) {
      notifiers.emplace_back([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
          notifier.notify_one();
          std::this_thread::yield();
        }
      });
    }
  } else if constexpr (NT == NotificationType::N) {
    for (size_t t = 0; t < M; ++t) {
      notifiers.emplace_back([&, t]() {
        std::mt19937 trng(seed + static_cast<uint32_t>(t + 1));
        std::uniform_int_distribution<size_t> dist(0, N);
        while (!stop.load(std::memory_order_relaxed)) {
          notifier.notify_n(dist(trng));
          std::this_thread::yield();
        }
      });
    }
  } else if constexpr (NT == NotificationType::ALL) {
    for (size_t t = 0; t < M; ++t) {
      notifiers.emplace_back([&, t]() {
        std::mt19937 trng(seed + static_cast<uint32_t>(777 + t));
        std::uniform_int_distribution<int> burst(1, 8);
        while (!stop.load(std::memory_order_relaxed)) {
          int b = burst(trng);
          for (int i = 0; i < b; ++i) notifier.notify_all();
          std::this_thread::yield();
        }
      });
    }
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dice(0, 1);

  for (size_t r = 0; r < R; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    canceled.store(0, std::memory_order_relaxed);
    committed.store(0, std::memory_order_relaxed);
    go.store(false, std::memory_order_release);

    for (size_t i = 0; i < N; ++i) has_work[i].store(false, std::memory_order_relaxed);

    round.store(r + 1, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) std::this_thread::yield();

    for (size_t i = 0; i < N; ++i) {
      if (dice(rng)) has_work[i].store(true, std::memory_order_release);
    }

    go.store(true, std::memory_order_release);

    while ((canceled.load(std::memory_order_acquire) +
            committed.load(std::memory_order_acquire)) != N) {
      std::this_thread::yield();
    }

    REQUIRE(notifier.num_waiters() == 0);
  }

  stop.store(true, std::memory_order_release);
  go.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& w : workers)   w.join();
  for (auto& n : notifiers) n.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// fuzz_stress_notifier
//   Condition-variable style predicate (signal_epoch). Workers follow the
//   full two-phase wait protocol with randomized delays and partial
//   participation. M notifier threads continuously advance the epoch and
//   mix all three notify variants. Validates:
//     - Every commit_wait returns (no stuck thread).
//     - num_waiters() == 0 after shutdown.
//     - prepares > 0 (slow path exercised).
//     - fast_path > 0 (predicate check before commit works).
// ============================================================================

template <typename T>
void fuzz_stress_notifier(size_t N, size_t M_notifiers, size_t rounds, uint32_t seed) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<bool>     stop{false};
  std::atomic<uint64_t> signal_epoch{0};

  std::atomic<uint64_t> prepares{0};
  std::atomic<uint64_t> cancels{0};
  std::atomic<uint64_t> commits_entered{0};
  std::atomic<uint64_t> commits_returned{0};
  std::atomic<uint64_t> fast_path{0};

  std::vector<std::thread> workers;
  workers.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    workers.emplace_back([&, i] {
      std::mt19937 rng(seed ^ (0x9e3779b9u + (uint32_t)i * 101u));
      uint64_t local = signal_epoch.load(std::memory_order_relaxed);
      std::uniform_int_distribution<int> coin(0, 99);

      for (size_t it = 0; it < rounds && !stop.load(std::memory_order_relaxed); ++it) {

        // Partial participation: sometimes do "work" without waiting.
        if (coin(rng) < 15) { tiny_jitter(rng); continue; }

        uint64_t cur = signal_epoch.load(std::memory_order_acquire);
        if (cur != local) {
          local = cur;
          fast_path.fetch_add(1, std::memory_order_relaxed);
          tiny_jitter(rng);
          continue;
        }

        // Two-phase wait protocol.
        tiny_jitter(rng);
        notifier.prepare_wait(i);
        prepares.fetch_add(1, std::memory_order_relaxed);

        tiny_jitter(rng);

        cur = signal_epoch.load(std::memory_order_acquire);
        if (cur != local) {
          notifier.cancel_wait(i);
          cancels.fetch_add(1, std::memory_order_relaxed);
          local = cur;
          tiny_jitter(rng);
          continue;
        }

        commits_entered.fetch_add(1, std::memory_order_relaxed);
        notifier.commit_wait(i);
        commits_returned.fetch_add(1, std::memory_order_relaxed);

        local = signal_epoch.load(std::memory_order_acquire);
        tiny_jitter(rng);
      }
    });
  }

  std::vector<std::thread> notifiers;
  notifiers.reserve(M_notifiers);

  for (size_t t = 0; t < M_notifiers; ++t) {
    notifiers.emplace_back([&, t] {
      std::mt19937 rng(seed + (uint32_t)(777u + t * 17u));
      std::uniform_int_distribution<int> which(0, 99);

      while (!stop.load(std::memory_order_relaxed)) {
        tiny_jitter(rng);
        int w = which(rng);

        // Occasionally send "empty" notifies (no predicate change) — must be harmless.
        if (w < 15) {
          int kind = which(rng) % 3;
          if (kind == 0)      notifier.notify_one();
          else if (kind == 1) notifier.notify_n((size_t)(which(rng) % (int)(N + 1)));
          else                notifier.notify_all();
          continue;
        }

        // Normal: advance predicate then notify (condition-variable style).
        signal_epoch.fetch_add(1, std::memory_order_release);

        if (w < 60)      notifier.notify_one();
        else if (w < 85) notifier.notify_n((size_t)(which(rng) % (int)(N + 1)));
        else             notifier.notify_all();
      }
    });
  }

  for (auto& w : workers) w.join();

  stop.store(true, std::memory_order_release);
  notifier.notify_all();
  for (auto& n : notifiers) n.join();

  REQUIRE(notifier.num_waiters() == 0);
  REQUIRE(commits_returned.load() == commits_entered.load());
  REQUIRE(prepares.load() > 0);
  REQUIRE(fast_path.load() > 0);
}

// ============================================================================
// notify_n_releases_committed
//   Forces ALL N threads to become committed waiters (num_waiters() == N),
//   then verifies notify_n(k) alone releases at least min(k, N) of them.
//   notify_all() drains the rest so the round completes.
//   This is a targeted regression for notify_n boundary semantics and for
//   spurious early-exit from commit_wait due to an epoch field bug.
// ============================================================================

template <typename T>
void notify_n_releases_committed(size_t N, size_t k, size_t rounds, uint32_t seed) {
  (void)seed;

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared(0);
  std::atomic<size_t> committed_done(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> workers;
  workers.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    workers.emplace_back([&, i] {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        while (round.load(std::memory_order_acquire) == local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }

        if (stop.load(std::memory_order_relaxed)) break;

        // No cancel path: we want all threads to become committed waiters.
        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);

        notifier.commit_wait(i);
        committed_done.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  for (size_t r = 1; r <= rounds; ++r) {
    prepared.store(0, std::memory_order_relaxed);
    committed_done.store(0, std::memory_order_relaxed);

    round.store(r, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) std::this_thread::yield();

    // Wait until ALL threads are truly committed waiters.
    while (notifier.num_waiters() != N) std::this_thread::yield();

    size_t target = (k < N) ? k : N;
    notifier.notify_n(k);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (committed_done.load(std::memory_order_acquire) < target &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    REQUIRE(committed_done.load(std::memory_order_acquire) >= target);

    // Drain remaining waiters.
    notifier.notify_all();

    auto deadline2 = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (committed_done.load(std::memory_order_acquire) != N &&
           std::chrono::steady_clock::now() < deadline2) {
      notifier.notify_all();
      std::this_thread::yield();
    }

    REQUIRE(committed_done.load(std::memory_order_acquire) == N);
    REQUIRE(notifier.num_waiters() == 0);
  }

  stop.store(true, std::memory_order_release);
  notifier.notify_all();

  for (auto& t : workers) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// stress_test_notifier
//   Validates:
//   1. notify_n(k) wakes at least k threads if k <= N.
//   2. notify_all() clears all remaining waiters.
//   3. num_waiters() correctly reflects the epoch-waiter state.
//   4. The epoch mechanism handles wakeups without spurious immediate re-sleeps.
// ============================================================================

template <typename T>
void stress_test_notifier(size_t N, size_t k, size_t rounds) {

  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<size_t> round(0);
  std::atomic<size_t> prepared_count(0);
  std::atomic<size_t> wake_count(0);
  std::atomic<bool>   stop(false);

  std::vector<std::thread> workers;
  workers.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    workers.emplace_back([&, i] {
      size_t local_round = 0;

      while (!stop.load(std::memory_order_relaxed)) {

        // Include !stop in the condition so the inner spin exits cleanly
        // when the main sets stop=true (not only when round advances).
        while (round.load(std::memory_order_acquire) == local_round &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed)) return;

        notifier.prepare_wait(i);
        prepared_count.fetch_add(1, std::memory_order_release);

        // Two-phase stop check: prepare_wait's seq_cst fence guarantees
        // that stop=true is visible here if main stored it before notify_all().
        // Without this, notify_all() could fire before prepare_wait and the
        // worker would block in commit_wait with no future wakeup.
        if (stop.load(std::memory_order_relaxed)) {
          notifier.cancel_wait(i);
          return;
        }

        notifier.commit_wait(i);
        wake_count.fetch_add(1, std::memory_order_release);

        local_round++;
      }
    });
  }

  for (size_t r = 1; r <= rounds; ++r) {
    prepared_count.store(0);
    wake_count.store(0);

    round.store(r, std::memory_order_release);

    while (prepared_count.load(std::memory_order_acquire) != N) {
      std::this_thread::yield();
    }

    while (notifier.num_waiters() != N) {
      std::this_thread::yield();
    }

    size_t target = std::min(k, N);
    notifier.notify_n(k);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (wake_count.load(std::memory_order_acquire) < target) {
      if (std::chrono::steady_clock::now() > deadline) break;
      std::this_thread::yield();
    }

    REQUIRE(wake_count.load(std::memory_order_acquire) >= target);

    notifier.notify_all();

    auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (wake_count.load(std::memory_order_acquire) < N) {
      if (std::chrono::steady_clock::now() > drain_deadline) {
        notifier.notify_all();
      }
      std::this_thread::yield();
    }

    REQUIRE(wake_count.load(std::memory_order_acquire) == N);
    REQUIRE(notifier.num_waiters() == 0);
  }

  stop.store(true);
  round.fetch_add(1);
  notifier.notify_all();

  for (auto& t : workers) {
    if (t.joinable()) t.join();
  }

  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// notify_before_commit
//   notify_all() fires BEFORE threads call commit_wait(). Threads must NOT
//   block (no lost wakeup). Exercises the window between prepare_wait() and
//   commit_wait() where the notification arrives.
// ============================================================================

template <typename T>
void notify_before_commit(size_t N) {

  T notifier(N);

  std::atomic<size_t> prep_count{0};
  std::atomic<size_t> committed{0};
  std::atomic<bool>   notified{false};

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      notifier.prepare_wait(i);
      prep_count.fetch_add(1, std::memory_order_release);

      // Spin until the main thread has already called notify_all().
      while (!notified.load(std::memory_order_acquire)) std::this_thread::yield();

      notifier.commit_wait(i);
      committed.fetch_add(1, std::memory_order_release);
    });
  }

  while (prep_count.load(std::memory_order_acquire) != N) std::this_thread::yield();

  // Notify before any thread calls commit_wait.
  notifier.notify_all();
  notified.store(true, std::memory_order_release);

  for (auto& t : threads) t.join();

  REQUIRE(committed.load() == N);
  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// notify_n_zero_is_noop
//   After all N threads are fully committed (num_waiters() == N),
//   notify_n(0) must not wake any of them. Immediately after the call
//   num_waiters() must still be N. Then notify_all() drains cleanly.
//   Catches implementations that treat 0 as "notify all" or misread the
//   loop bound.
// ============================================================================

template <typename T>
void notify_n_zero_is_noop(size_t N) {
  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<size_t> prepared{0};
  std::atomic<size_t> committed{0};

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i] {
      notifier.prepare_wait(i);
      prepared.fetch_add(1, std::memory_order_release);
      notifier.commit_wait(i);
      committed.fetch_add(1, std::memory_order_release);
    });
  }

  while (prepared.load(std::memory_order_acquire) != N) std::this_thread::yield();
  while (notifier.num_waiters() != N) std::this_thread::yield();

  notifier.notify_n(0);

  // No thread should have woken up.
  REQUIRE(notifier.num_waiters() == N);
  REQUIRE(committed.load(std::memory_order_acquire) == 0);

  notifier.notify_all();
  for (auto& t : threads) t.join();

  REQUIRE(committed.load() == N);
  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// notify_n_does_not_over_release
//   After notify_n(k) releases exactly min(k,N) committed waiters, the
//   remaining N−k must still be parked (num_waiters() == N−k).
//   Drain them with notify_all() and verify total == N.
//   Catches off-by-one in notify_n's loop or a loop that iterates too many
//   times.
// ============================================================================

template <typename T>
void notify_n_does_not_over_release(size_t N, size_t k, size_t rounds) {
  T notifier(N);
  REQUIRE(notifier.size() == N);

  std::atomic<size_t> round{0};
  std::atomic<size_t> prepared{0};
  std::atomic<size_t> committed_done{0};
  std::atomic<bool>   stop{false};

  std::vector<std::thread> workers;
  workers.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    workers.emplace_back([&, i] {
      size_t local = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        while (round.load(std::memory_order_acquire) == local &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed)) break;
        notifier.prepare_wait(i);
        prepared.fetch_add(1, std::memory_order_release);
        notifier.commit_wait(i);
        committed_done.fetch_add(1, std::memory_order_release);
        local++;
      }
    });
  }

  size_t target = (k < N) ? k : N;

  for (size_t r = 1; r <= rounds; ++r) {
    prepared.store(0);
    committed_done.store(0);

    round.store(r, std::memory_order_release);

    while (prepared.load(std::memory_order_acquire) != N) std::this_thread::yield();
    while (notifier.num_waiters() != N) std::this_thread::yield();

    notifier.notify_n(k);

    // Wait for at least target wakeups.
    auto d1 = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (committed_done.load(std::memory_order_acquire) < target &&
           std::chrono::steady_clock::now() < d1) {
      std::this_thread::yield();
    }
    REQUIRE(committed_done.load(std::memory_order_acquire) >= target);

    // Remaining waiters must still be parked (no over-release).
    size_t remaining = N - committed_done.load(std::memory_order_acquire);
    REQUIRE(notifier.num_waiters() == remaining);

    notifier.notify_all();

    auto d2 = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (committed_done.load(std::memory_order_acquire) != N &&
           std::chrono::steady_clock::now() < d2) {
      notifier.notify_all();
      std::this_thread::yield();
    }
    REQUIRE(committed_done.load(std::memory_order_acquire) == N);
    REQUIRE(notifier.num_waiters() == 0);
  }

  stop.store(true);
  notifier.notify_all();
  for (auto& w : workers) w.join();
  REQUIRE(notifier.num_waiters() == 0);
}

// ============================================================================
// NonblockingNotifier-specific helpers (non-template; test semantics that
// differ from AtomicNotifier).
// ============================================================================

// num_waiters() must count only committed (kWaiting) threads, not pre-waiters.
static void num_waiters_counts_only_committed() {
  tf::NonblockingNotifier notifier(4);

  REQUIRE(notifier.num_waiters() == 0);

  // Single-threaded prepare → num_waiters must stay 0 (pre-waiter, not committed).
  notifier.prepare_wait(0);
  REQUIRE(notifier.num_waiters() == 0);
  notifier.cancel_wait(0);
  REQUIRE(notifier.num_waiters() == 0);

  // Two sequential prepare+cancel cycles; num_waiters must remain 0.
  notifier.prepare_wait(1);
  notifier.cancel_wait(1);
  notifier.prepare_wait(2);
  notifier.cancel_wait(2);
  REQUIRE(notifier.num_waiters() == 0);

  // Now let a thread actually commit: it must show up as 1 waiter.
  std::atomic<bool> parked{false};
  std::thread t([&] {
    notifier.prepare_wait(3);
    notifier.commit_wait(3);
    parked.store(true, std::memory_order_release);
  });

  while (notifier.num_waiters() != 1) std::this_thread::yield();
  REQUIRE(notifier.num_waiters() == 1);

  notifier.notify_all();
  t.join();
  REQUIRE(notifier.num_waiters() == 0);
}

// Many sequential prepare+cancel cycles; verify the notifier is still usable afterwards.
// For NonblockingNotifier, cancels advance the EPOCH counter — this stresses 32-bit overflow.
// For AtomicNotifier,  cancels adjust the waiter count — this stresses count accuracy.
template <typename T>
void rapid_cancel_integrity(size_t cycles) {
  T notifier(2);

  for (size_t i = 0; i < cycles; ++i) {
    notifier.prepare_wait(0);
    notifier.cancel_wait(0);
  }
  REQUIRE(notifier.num_waiters() == 0);

  std::atomic<size_t> committed{0};
  std::thread t([&] {
    notifier.prepare_wait(0);
    notifier.commit_wait(0);
    committed.fetch_add(1, std::memory_order_release);
  });

  while (notifier.num_waiters() < 1) std::this_thread::yield();
  notifier.notify_all();
  t.join();

  REQUIRE(committed.load() == 1);
  REQUIRE(notifier.num_waiters() == 0);
}

// Many sequential prepare+cancel cycles to exercise 32-bit epoch wraparound.
static void rapid_cancel_epoch_integrity(size_t cycles) {
  tf::NonblockingNotifier notifier(2);

  // Saturate the epoch counter.
  for (size_t i = 0; i < cycles; ++i) {
    notifier.prepare_wait(0);
    notifier.cancel_wait(0);
  }
  REQUIRE(notifier.num_waiters() == 0);

  // Verify the notifier is still usable after epoch saturation.
  std::atomic<size_t> committed{0};
  std::thread t([&] {
    notifier.prepare_wait(0);
    notifier.commit_wait(0);
    committed.fetch_add(1, std::memory_order_release);
  });

  while (notifier.num_waiters() != 1) std::this_thread::yield();
  notifier.notify_all();
  t.join();

  REQUIRE(committed.load() == 1);
  REQUIRE(notifier.num_waiters() == 0);
}

// notify_one() fires in the pre-waiter window (after prepare_wait, before commit_wait).
// Works for both NonblockingNotifier (num_waiters stays 0 during pre-wait) and
// AtomicNotifier (num_waiters == 1 immediately after prepare_wait).
template <typename T>
void notify_one_at_prewaiter_window(size_t rounds) {
  T notifier(1);

  for (size_t r = 0; r < rounds; ++r) {
    std::atomic<bool> prepared{false};
    std::atomic<bool> go{false};
    std::atomic<bool> done{false};

    std::thread t([&] {
      notifier.prepare_wait(0);
      prepared.store(true, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      notifier.commit_wait(0);
      done.store(true, std::memory_order_release);
    });

    while (!prepared.load(std::memory_order_acquire)) std::this_thread::yield();

    // Fire notify_one while the thread is in the pre-wait window.
    notifier.notify_one();
    go.store(true, std::memory_order_release);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }

    REQUIRE(done.load(std::memory_order_acquire));
    t.join();
    REQUIRE(notifier.num_waiters() == 0);
  }
}

// ============================================================================
// REGISTER_CORE_TESTS(T, P)
//   Registers the shared correctness suite for notifier type T.
//   T   – notifier class (tf::NonblockingNotifier or tf::AtomicNotifier).
//   P   – string literal prefix, e.g. "NonblockingNotifier".
//   C adjacent-string-literal concatenation produces the final test names,
//   e.g. P ".no_missing_notify_all.1threads" → "NonblockingNotifier.no_missing_notify_all.1threads".
// ============================================================================

// clang-format off
#define REGISTER_CORE_TESTS(T, P)                                                                           \
  /* --- no_missing_notify_all --- */                                                                        \
  TEST_CASE(P ".no_missing_notify_all.1threads"  * doctest::timeout(300)) { no_missing_notify_all<T>(1);  } \
  TEST_CASE(P ".no_missing_notify_all.2threads"  * doctest::timeout(300)) { no_missing_notify_all<T>(2);  } \
  TEST_CASE(P ".no_missing_notify_all.4threads"  * doctest::timeout(300)) { no_missing_notify_all<T>(4);  } \
  TEST_CASE(P ".no_missing_notify_all.5threads"  * doctest::timeout(300)) { no_missing_notify_all<T>(5);  } \
  TEST_CASE(P ".no_missing_notify_all.8threads"  * doctest::timeout(300)) { no_missing_notify_all<T>(8);  } \
  TEST_CASE(P ".no_missing_notify_all.15threads" * doctest::timeout(300)) { no_missing_notify_all<T>(15); } \
  TEST_CASE(P ".no_missing_notify_all.31threads" * doctest::timeout(300)) { no_missing_notify_all<T>(31); } \
  /* --- no_missing_notify_one --- */                                                                        \
  TEST_CASE(P ".no_missing_notify_one.1threads"  * doctest::timeout(300)) { no_missing_notify_one<T>(1);  } \
  TEST_CASE(P ".no_missing_notify_one.2threads"  * doctest::timeout(300)) { no_missing_notify_one<T>(2);  } \
  TEST_CASE(P ".no_missing_notify_one.4threads"  * doctest::timeout(300)) { no_missing_notify_one<T>(4);  } \
  TEST_CASE(P ".no_missing_notify_one.5threads"  * doctest::timeout(300)) { no_missing_notify_one<T>(5);  } \
  TEST_CASE(P ".no_missing_notify_one.8threads"  * doctest::timeout(300)) { no_missing_notify_one<T>(8);  } \
  TEST_CASE(P ".no_missing_notify_one.15threads" * doctest::timeout(300)) { no_missing_notify_one<T>(15); } \
  TEST_CASE(P ".no_missing_notify_one.31threads" * doctest::timeout(300)) { no_missing_notify_one<T>(31); } \
  /* --- no_missing_notify_n --- */                                                                          \
  TEST_CASE(P ".no_missing_notify_n.1threads"  * doctest::timeout(300)) { no_missing_notify_n<T>(1);  }     \
  TEST_CASE(P ".no_missing_notify_n.2threads"  * doctest::timeout(300)) { no_missing_notify_n<T>(2);  }     \
  TEST_CASE(P ".no_missing_notify_n.4threads"  * doctest::timeout(300)) { no_missing_notify_n<T>(4);  }     \
  TEST_CASE(P ".no_missing_notify_n.5threads"  * doctest::timeout(300)) { no_missing_notify_n<T>(5);  }     \
  TEST_CASE(P ".no_missing_notify_n.8threads"  * doctest::timeout(300)) { no_missing_notify_n<T>(8);  }     \
  TEST_CASE(P ".no_missing_notify_n.15threads" * doctest::timeout(300)) { no_missing_notify_n<T>(15); }     \
  TEST_CASE(P ".no_missing_notify_n.31threads" * doctest::timeout(300)) { no_missing_notify_n<T>(31); }     \
  /* --- no_missing_notify_x (mixed) --- */                                                                  \
  TEST_CASE(P ".no_missing_notify_x.1threads"  * doctest::timeout(300)) { no_missing_notify_x<T>(1);  }     \
  TEST_CASE(P ".no_missing_notify_x.2threads"  * doctest::timeout(300)) { no_missing_notify_x<T>(2);  }     \
  TEST_CASE(P ".no_missing_notify_x.4threads"  * doctest::timeout(300)) { no_missing_notify_x<T>(4);  }     \
  TEST_CASE(P ".no_missing_notify_x.5threads"  * doctest::timeout(300)) { no_missing_notify_x<T>(5);  }     \
  TEST_CASE(P ".no_missing_notify_x.8threads"  * doctest::timeout(300)) { no_missing_notify_x<T>(8);  }     \
  TEST_CASE(P ".no_missing_notify_x.15threads" * doctest::timeout(300)) { no_missing_notify_x<T>(15); }     \
  TEST_CASE(P ".no_missing_notify_x.31threads" * doctest::timeout(300)) { no_missing_notify_x<T>(31); }     \
  /* --- no_missing_cancel_wait --- */                                                                        \
  TEST_CASE(P ".no_missing_cancel_wait.1threads"  * doctest::timeout(300)) { no_missing_cancel_wait<T>(1);  } \
  TEST_CASE(P ".no_missing_cancel_wait.2threads"  * doctest::timeout(300)) { no_missing_cancel_wait<T>(2);  } \
  TEST_CASE(P ".no_missing_cancel_wait.4threads"  * doctest::timeout(300)) { no_missing_cancel_wait<T>(4);  } \
  TEST_CASE(P ".no_missing_cancel_wait.5threads"  * doctest::timeout(300)) { no_missing_cancel_wait<T>(5);  } \
  TEST_CASE(P ".no_missing_cancel_wait.8threads"  * doctest::timeout(300)) { no_missing_cancel_wait<T>(8);  } \
  TEST_CASE(P ".no_missing_cancel_wait.16threads" * doctest::timeout(300)) { no_missing_cancel_wait<T>(16); } \
  TEST_CASE(P ".no_missing_cancel_wait.31threads" * doctest::timeout(300)) { no_missing_cancel_wait<T>(31); } \
  /* --- no_missing_notifications: ONE --- */                                                                  \
  TEST_CASE(P ".no_missing_notify_ones.1threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(1);  } \
  TEST_CASE(P ".no_missing_notify_ones.2threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(2);  } \
  TEST_CASE(P ".no_missing_notify_ones.4threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(4);  } \
  TEST_CASE(P ".no_missing_notify_ones.5threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(5);  } \
  TEST_CASE(P ".no_missing_notify_ones.8threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(8);  } \
  TEST_CASE(P ".no_missing_notify_ones.16threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(16); } \
  TEST_CASE(P ".no_missing_notify_ones.31threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ONE>(31); } \
  /* --- no_missing_notifications: N --- */                                                                    \
  TEST_CASE(P ".no_missing_notify_ns.1threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(1);  } \
  TEST_CASE(P ".no_missing_notify_ns.2threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(2);  } \
  TEST_CASE(P ".no_missing_notify_ns.4threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(4);  } \
  TEST_CASE(P ".no_missing_notify_ns.5threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(5);  } \
  TEST_CASE(P ".no_missing_notify_ns.8threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(8);  } \
  TEST_CASE(P ".no_missing_notify_ns.16threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(16); } \
  TEST_CASE(P ".no_missing_notify_ns.31threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::N>(31); } \
  /* --- no_missing_notifications: ALL --- */                                                                  \
  TEST_CASE(P ".no_missing_notify_alls.1threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(1);  } \
  TEST_CASE(P ".no_missing_notify_alls.2threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(2);  } \
  TEST_CASE(P ".no_missing_notify_alls.4threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(4);  } \
  TEST_CASE(P ".no_missing_notify_alls.5threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(5);  } \
  TEST_CASE(P ".no_missing_notify_alls.8threads"  * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(8);  } \
  TEST_CASE(P ".no_missing_notify_alls.16threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(16); } \
  TEST_CASE(P ".no_missing_notify_alls.31threads" * doctest::timeout(300)) { no_missing_notifications<T, NotificationType::ALL>(31); } \
  /* --- fuzz_stress --- */                                                                                    \
  TEST_CASE(P ".fuzz_stress.1threads")                    { fuzz_stress_notifier<T>(1,   1, 20000, 1);    }   \
  TEST_CASE(P ".fuzz_stress.2threads")                    { fuzz_stress_notifier<T>(2,   2, 12000, 2);    }   \
  TEST_CASE(P ".fuzz_stress.3threads")                    { fuzz_stress_notifier<T>(3,   2,  9000, 3);    }   \
  TEST_CASE(P ".fuzz_stress.7threads")                    { fuzz_stress_notifier<T>(7,   3,  6000, 77);   }   \
  TEST_CASE(P ".fuzz_stress.13threads")                   { fuzz_stress_notifier<T>(13,  4,  3500, 1313); }   \
  TEST_CASE(P ".fuzz_stress.16threads")                   { fuzz_stress_notifier<T>(16,  4,  3000, 1616); }   \
  TEST_CASE(P ".fuzz_stress.29threads")                   { fuzz_stress_notifier<T>(29,  5,  2000, 2929); }   \
  TEST_CASE(P ".fuzz_stress.32threads")                   { fuzz_stress_notifier<T>(32,  6,  1500, 3232); }   \
  TEST_CASE(P ".fuzz_stress.64threads")                   { fuzz_stress_notifier<T>(64,  8,   800, 6464); }   \
  TEST_CASE(P ".fuzz_stress.more_notifiers_than_workers") { fuzz_stress_notifier<T>(8,  16,  3000, 8888); }   \
  /* --- notify_n_releases_committed (common subset) --- */                                                    \
  TEST_CASE(P ".notify_n_releases_committed.N8.k1"   * doctest::timeout(300)) { notify_n_releases_committed<T>(8,   1, 200, 1); } \
  TEST_CASE(P ".notify_n_releases_committed.N8.k3"   * doctest::timeout(300)) { notify_n_releases_committed<T>(8,   3, 200, 2); } \
  TEST_CASE(P ".notify_n_releases_committed.N8.k16"  * doctest::timeout(300)) { notify_n_releases_committed<T>(8,  16, 200, 5); } \
  TEST_CASE(P ".notify_n_releases_committed.N31.k15" * doctest::timeout(300)) { notify_n_releases_committed<T>(31, 15, 120, 3); } \
  TEST_CASE(P ".notify_n_releases_committed.N31.k31" * doctest::timeout(300)) { notify_n_releases_committed<T>(31, 31, 120, 4); }
// clang-format on


// ============================================================================
// TEST CASES: NonblockingNotifier
// ============================================================================

REGISTER_CORE_TESTS(tf::NonblockingNotifier, "NonblockingNotifier")

// --- invariants (NonblockingNotifier counts only committed threads) ---

TEST_CASE("NonblockingNotifier.invariants.size") {
  tf::NonblockingNotifier n1(1);   REQUIRE(n1.size() == 1);
  tf::NonblockingNotifier n4(4);   REQUIRE(n4.size() == 4);
  tf::NonblockingNotifier n31(31); REQUIRE(n31.size() == 31);
}

TEST_CASE("NonblockingNotifier.invariants.capacity") {
  tf::NonblockingNotifier notifier(8);
  REQUIRE(notifier.capacity() == (1u << tf::NonblockingNotifier::STACK_BITS));
}

TEST_CASE("NonblockingNotifier.invariants.initial_num_waiters_is_zero") {
  tf::NonblockingNotifier notifier(8);
  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("NonblockingNotifier.invariants.notify_on_empty_is_noop") {
  tf::NonblockingNotifier notifier(4);
  notifier.notify_one();
  notifier.notify_all();
  notifier.notify_n(0);
  notifier.notify_n(2);
  notifier.notify_n(4);
  notifier.notify_n(5);
  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("NonblockingNotifier.invariants.prepare_does_not_increment_num_waiters") {
  // Pre-waiters must NOT show up in num_waiters(); only committed threads do.
  tf::NonblockingNotifier notifier(4);
  notifier.prepare_wait(0);
  REQUIRE(notifier.num_waiters() == 0);
  notifier.cancel_wait(0);
  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("NonblockingNotifier.invariants.num_waiters_counts_only_committed") {
  num_waiters_counts_only_committed();
}

// --- notify_before_commit ---

TEST_CASE("NonblockingNotifier.notify_before_commit.1thread"  * doctest::timeout(60)) { notify_before_commit<tf::NonblockingNotifier>(1);  }
TEST_CASE("NonblockingNotifier.notify_before_commit.2threads" * doctest::timeout(60)) { notify_before_commit<tf::NonblockingNotifier>(2);  }
TEST_CASE("NonblockingNotifier.notify_before_commit.4threads" * doctest::timeout(60)) { notify_before_commit<tf::NonblockingNotifier>(4);  }
TEST_CASE("NonblockingNotifier.notify_before_commit.8threads" * doctest::timeout(60)) { notify_before_commit<tf::NonblockingNotifier>(8);  }
TEST_CASE("NonblockingNotifier.notify_before_commit.15threads"* doctest::timeout(60)) { notify_before_commit<tf::NonblockingNotifier>(15); }

// --- cancel_only (no ghost waiters) ---

TEST_CASE("NonblockingNotifier.cancel_only_no_ghost_waiters.1thread" * doctest::timeout(60)) {
  tf::NonblockingNotifier notifier(1);
  REQUIRE(notifier.num_waiters() == 0);
  for (int i = 0; i < 2000; ++i) {
    notifier.prepare_wait(0);
    REQUIRE(notifier.num_waiters() == 0);
    notifier.cancel_wait(0);
    REQUIRE(notifier.num_waiters() == 0);
  }
}

TEST_CASE("NonblockingNotifier.cancel_only_no_ghost_waiters.8threads" * doctest::timeout(60)) {
  const size_t N = 8;
  tf::NonblockingNotifier notifier(N);

  std::atomic<size_t> round{0};
  std::atomic<bool>   stop{false};
  std::vector<std::thread> threads;
  threads.reserve(N);

  // Each thread does sequential prepare+cancel rounds (ordering respected via
  // round counter so tickets resolve in order).
  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i] {
      size_t local = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        while (round.load(std::memory_order_acquire) == local &&
               !stop.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed)) break;
        notifier.prepare_wait(i);
        notifier.cancel_wait(i);
        local++;
      }
    });
  }

  for (size_t r = 1; r <= 300; ++r) {
    round.store(r, std::memory_order_release);
    // Allow all threads time to complete the round before moving on.
    std::this_thread::yield();
  }

  stop.store(true, std::memory_order_release);
  round.store(round.load() + 1, std::memory_order_release);
  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// --- Basic stress ---

TEST_CASE("NonblockingNotifier.Basic" * doctest::timeout(10)) {
  SUBCASE("OneAtATime") { stress_test_notifier<tf::NonblockingNotifier>(4,  1, 10); }
  SUBCASE("HalfBurst")  { stress_test_notifier<tf::NonblockingNotifier>(8,  4,  5); }
  SUBCASE("FullBurst")  { stress_test_notifier<tf::NonblockingNotifier>(12, 12, 5); }
  SUBCASE("OverSaturate"){ stress_test_notifier<tf::NonblockingNotifier>(4, 10,  5); }
}

// --- notify_n boundary (k == N) ---

TEST_CASE("NonblockingNotifier.notify_n_exact_size.4threads"  * doctest::timeout(60)) { notify_n_releases_committed<tf::NonblockingNotifier>(4,  4,  50, 43); }
TEST_CASE("NonblockingNotifier.notify_n_exact_size.8threads"  * doctest::timeout(60)) { notify_n_releases_committed<tf::NonblockingNotifier>(8,  8,  50, 42); }
TEST_CASE("NonblockingNotifier.notify_n_exact_size.15threads" * doctest::timeout(60)) { notify_n_releases_committed<tf::NonblockingNotifier>(15, 15, 30, 44); }

// --- notify_n(0) is a strict noop ---

TEST_CASE("NonblockingNotifier.notify_n_zero_is_noop.4threads" * doctest::timeout(60)) { notify_n_zero_is_noop<tf::NonblockingNotifier>(4); }
TEST_CASE("NonblockingNotifier.notify_n_zero_is_noop.8threads" * doctest::timeout(60)) { notify_n_zero_is_noop<tf::NonblockingNotifier>(8); }

// --- notify_n does not over-release ---

TEST_CASE("NonblockingNotifier.notify_n_no_over_release.N8.k1"   * doctest::timeout(120)) { notify_n_does_not_over_release<tf::NonblockingNotifier>(8,   1, 100); }
TEST_CASE("NonblockingNotifier.notify_n_no_over_release.N8.k3"   * doctest::timeout(120)) { notify_n_does_not_over_release<tf::NonblockingNotifier>(8,   3, 100); }
TEST_CASE("NonblockingNotifier.notify_n_no_over_release.N15.k7"  * doctest::timeout(120)) { notify_n_does_not_over_release<tf::NonblockingNotifier>(15,  7,  60); }
TEST_CASE("NonblockingNotifier.notify_n_no_over_release.N31.k15" * doctest::timeout(120)) { notify_n_does_not_over_release<tf::NonblockingNotifier>(31, 15,  40); }

// --- rapid_cancel_integrity (generic) and rapid_cancel_epoch_integrity (epoch-overflow) ---

TEST_CASE("NonblockingNotifier.rapid_cancel_integrity.1k_cycles"   * doctest::timeout(30)) { rapid_cancel_integrity<tf::NonblockingNotifier>(1000);   }
TEST_CASE("NonblockingNotifier.rapid_cancel_integrity.100k_cycles" * doctest::timeout(60)) { rapid_cancel_integrity<tf::NonblockingNotifier>(100000); }
TEST_CASE("NonblockingNotifier.rapid_cancel_epoch_integrity.1k_cycles"   * doctest::timeout(30)) { rapid_cancel_epoch_integrity(1000);   }
TEST_CASE("NonblockingNotifier.rapid_cancel_epoch_integrity.100k_cycles" * doctest::timeout(60)) { rapid_cancel_epoch_integrity(100000); }

// --- notify_one fires in pre-waiter window ---

TEST_CASE("NonblockingNotifier.notify_one_at_prewaiter_window.50rounds"  * doctest::timeout(60)) { notify_one_at_prewaiter_window<tf::NonblockingNotifier>(50);  }
TEST_CASE("NonblockingNotifier.notify_one_at_prewaiter_window.200rounds" * doctest::timeout(60)) { notify_one_at_prewaiter_window<tf::NonblockingNotifier>(200); }


// ============================================================================
// TEST CASES: AtomicNotifier
// ============================================================================

REGISTER_CORE_TESTS(tf::AtomicNotifier, "AtomicNotifier")

// --- notify_n boundary: k==N and k>N ---

TEST_CASE("AtomicNotifier.notify_n_releases_committed.N8.k8"   * doctest::timeout(300)) { notify_n_releases_committed<tf::AtomicNotifier>(8,  8,  200, 3); }
TEST_CASE("AtomicNotifier.notify_n_releases_committed.N31.k32" * doctest::timeout(300)) { notify_n_releases_committed<tf::AtomicNotifier>(31, 32, 120, 7); }

// --- Basic stress ---

TEST_CASE("AtomicNotifier.Basic" * doctest::timeout(10)) {
  SUBCASE("OneAtATime") { stress_test_notifier<tf::AtomicNotifier>(4,  1, 10); }
  SUBCASE("HalfBurst")  { stress_test_notifier<tf::AtomicNotifier>(8,  4,  5); }
  SUBCASE("FullBurst")  { stress_test_notifier<tf::AtomicNotifier>(12, 12, 5); }
  SUBCASE("OverSaturate"){ stress_test_notifier<tf::AtomicNotifier>(4, 10,  5); }
}

// --- invariants ---

TEST_CASE("AtomicNotifier.invariants.size") {
  tf::AtomicNotifier n1(1);   REQUIRE(n1.size() == 1);
  tf::AtomicNotifier n4(4);   REQUIRE(n4.size() == 4);
  tf::AtomicNotifier n31(31); REQUIRE(n31.size() == 31);
}

TEST_CASE("AtomicNotifier.invariants.initial_num_waiters_is_zero") {
  tf::AtomicNotifier notifier(8);
  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("AtomicNotifier.invariants.notify_on_empty_is_noop") {
  tf::AtomicNotifier notifier(4);
  notifier.notify_one();
  notifier.notify_all();
  notifier.notify_n(0);
  notifier.notify_n(2);
  notifier.notify_n(4);
  notifier.notify_n(5);  // n > size() — exercises notify_n boundary
  REQUIRE(notifier.num_waiters() == 0);
}

TEST_CASE("AtomicNotifier.invariants.prepare_cancel_restores_state") {
  tf::AtomicNotifier notifier(4);

  notifier.prepare_wait(0);
  REQUIRE(notifier.num_waiters() == 1);
  notifier.cancel_wait(0);
  REQUIRE(notifier.num_waiters() == 0);

  notifier.prepare_wait(1);
  notifier.prepare_wait(2);
  REQUIRE(notifier.num_waiters() == 2);
  notifier.cancel_wait(2);
  notifier.cancel_wait(1);
  REQUIRE(notifier.num_waiters() == 0);
}

// --- notify_before_commit ---

TEST_CASE("AtomicNotifier.notify_before_commit.5threads" * doctest::timeout(60)) { notify_before_commit<tf::AtomicNotifier>(5); }
TEST_CASE("AtomicNotifier.notify_before_commit.8threads" * doctest::timeout(60)) { notify_before_commit<tf::AtomicNotifier>(8); }

// --- cancel_only (no ghost waiters) ---

TEST_CASE("AtomicNotifier.cancel_only_no_ghost_waiters.1thread" * doctest::timeout(60)) {
  tf::AtomicNotifier notifier(1);
  REQUIRE(notifier.num_waiters() == 0);
  for (int i = 0; i < 10000; ++i) {
    notifier.prepare_wait(0);
    notifier.cancel_wait(0);
    REQUIRE(notifier.num_waiters() == 0);
  }
}

TEST_CASE("AtomicNotifier.cancel_only_no_ghost_waiters.8threads" * doctest::timeout(60)) {
  const size_t N = 8;
  tf::AtomicNotifier notifier(N);

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back([&, i] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int r = 0; r < 500; ++r) {
        notifier.prepare_wait(i);
        notifier.cancel_wait(i);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& t : threads) t.join();

  REQUIRE(notifier.num_waiters() == 0);
}

// --- bug2: notify_n boundary (k == N must invoke the notify_all path) ---

TEST_CASE("AtomicNotifier.bug2_notify_n_exact_size.4threads"  * doctest::timeout(60)) { notify_n_releases_committed<tf::AtomicNotifier>(4,  4,  50, 43); }
TEST_CASE("AtomicNotifier.bug2_notify_n_exact_size.8threads"  * doctest::timeout(60)) { notify_n_releases_committed<tf::AtomicNotifier>(8,  8,  50, 42); }
TEST_CASE("AtomicNotifier.bug2_notify_n_exact_size.15threads" * doctest::timeout(60)) { notify_n_releases_committed<tf::AtomicNotifier>(15, 15, 30, 44); }

// --- bug3: epoch field (prepare_wait must store the epoch field, not the waiter count) ---

TEST_CASE("AtomicNotifier.bug3_epoch_field.N2" * doctest::timeout(30)) { notify_n_releases_committed<tf::AtomicNotifier>(2, 1, 20, 99);  }
TEST_CASE("AtomicNotifier.bug3_epoch_field.N4" * doctest::timeout(30)) { notify_n_releases_committed<tf::AtomicNotifier>(4, 2, 20, 100); }
TEST_CASE("AtomicNotifier.bug3_epoch_field.N8" * doctest::timeout(30)) { notify_n_releases_committed<tf::AtomicNotifier>(8, 4, 20, 101); }

// --- notify_n(0) is a strict noop ---

TEST_CASE("AtomicNotifier.notify_n_zero_is_noop.4threads" * doctest::timeout(60)) { notify_n_zero_is_noop<tf::AtomicNotifier>(4); }
TEST_CASE("AtomicNotifier.notify_n_zero_is_noop.8threads" * doctest::timeout(60)) { notify_n_zero_is_noop<tf::AtomicNotifier>(8); }

// --- notify_one fires in pre-waiter window ---
// For AtomicNotifier, num_waiters() increments immediately at prepare_wait (unlike
// NonblockingNotifier where it only counts committed/parked threads).

TEST_CASE("AtomicNotifier.notify_one_at_prewaiter_window.50rounds"  * doctest::timeout(60)) { notify_one_at_prewaiter_window<tf::AtomicNotifier>(50);  }
TEST_CASE("AtomicNotifier.notify_one_at_prewaiter_window.200rounds" * doctest::timeout(60)) { notify_one_at_prewaiter_window<tf::AtomicNotifier>(200); }

// --- rapid_cancel_integrity ---
// Stresses that many prepare+cancel cycles keep the waiter count accurate and
// leave the notifier fully functional.

TEST_CASE("AtomicNotifier.rapid_cancel_integrity.1k_cycles"   * doctest::timeout(30)) { rapid_cancel_integrity<tf::AtomicNotifier>(1000);   }
TEST_CASE("AtomicNotifier.rapid_cancel_integrity.100k_cycles" * doctest::timeout(60)) { rapid_cancel_integrity<tf::AtomicNotifier>(100000); }