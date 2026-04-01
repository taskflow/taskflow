#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/core/freelist.hpp>

// ----------------------------------------------------------------------------
// Test node types
// ----------------------------------------------------------------------------

struct Node {
  Node* _parent {nullptr};
  int   value   {0};
};

struct Item {
  Item* _next_free {nullptr};
  int   id         {0};
};

using NodeStack = tf::AtomicIntrusiveStack<Node*, &Node::_parent>;
using ItemStack = tf::AtomicIntrusiveStack<Item*, &Item::_next_free>;

static std::vector<Node*> make_nodes(int N, int base_value = 0) {
  std::vector<Node*> v(N);
  for(int i = 0; i < N; ++i) v[i] = new Node{nullptr, base_value + i};
  return v;
}

static void free_nodes(std::vector<Node*>& v) {
  for(auto* n : v) delete n;
  v.clear();
}

// ============================================================================
// Single-threaded correctness
// ============================================================================

TEST_CASE("AtomicIntrusiveStack.empty" * doctest::timeout(10)) {
  NodeStack stack;
  REQUIRE(stack.empty() == true);
  Node n;
  stack.push(&n);
  REQUIRE(stack.empty() == false);
  stack.pop();
  REQUIRE(stack.empty() == true);
}

TEST_CASE("AtomicIntrusiveStack.pop_empty" * doctest::timeout(10)) {
  NodeStack stack;
  REQUIRE(stack.pop() == nullptr);
  REQUIRE(stack.pop() == nullptr);
}

TEST_CASE("AtomicIntrusiveStack.push_pop_lifo" * doctest::timeout(10)) {
  NodeStack stack;
  auto nodes = make_nodes(5);
  for(auto* n : nodes) stack.push(n);
  for(int i = 4; i >= 0; --i) {
    auto* n = stack.pop();
    REQUIRE(n != nullptr);
    REQUIRE(n->value == i);
  }
  REQUIRE(stack.pop() == nullptr);
  free_nodes(nodes);
}

TEST_CASE("AtomicIntrusiveStack.link_cleared_after_pop" * doctest::timeout(10)) {
  NodeStack stack;
  Node a{nullptr, 1}, b{nullptr, 2};
  stack.push(&a);
  stack.push(&b);

  Node* n = stack.pop();
  REQUIRE(n == &b);

  n = stack.pop();
  REQUIRE(n == &a);
}

TEST_CASE("AtomicIntrusiveStack.push_pop_single" * doctest::timeout(10)) {
  NodeStack stack;
  Node n{nullptr, 42};
  stack.push(&n);
  REQUIRE(stack.empty() == false);
  auto* r = stack.pop();
  REQUIRE(r == &n);
  REQUIRE(r->value == 42);
  REQUIRE(stack.empty() == true);
  REQUIRE(stack.pop() == nullptr);
}

TEST_CASE("AtomicIntrusiveStack.custom_member_ptr" * doctest::timeout(10)) {
  ItemStack stack;
  REQUIRE(stack.empty() == true);
  Item a{nullptr, 1}, b{nullptr, 2}, c{nullptr, 3};
  stack.push(&a);
  stack.push(&b);
  stack.push(&c);
  REQUIRE(stack.pop() == &c);
  REQUIRE(stack.pop() == &b);
  REQUIRE(stack.pop() == &a);
  REQUIRE(stack.pop() == nullptr);
}

// ============================================================================
// Concurrent push: M producers, single-threaded drain after join
// ============================================================================

void concurrent_push(size_t M) {
  NodeStack stack;
  const size_t N = 1000;
  std::vector<std::vector<Node*>> thread_nodes(M);
  std::vector<std::thread> threads;

  for(size_t t = 0; t < M; ++t) {
    thread_nodes[t] = make_nodes(N, static_cast<int>(t * N));
    threads.emplace_back([&, t](){
      for(auto* n : thread_nodes[t]) stack.push(n);
    });
  }
  for(auto& th : threads) th.join();

  std::vector<int> values;
  while(auto* n = stack.pop()) {
    values.push_back(n->value);
  }
  REQUIRE(values.size() == M * N);
  std::sort(values.begin(), values.end());
  for(size_t i = 0; i < M * N; ++i) {
    REQUIRE(values[i] == static_cast<int>(i));
  }
  for(auto& v : thread_nodes) free_nodes(v);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_push.1producer" * doctest::timeout(300)) {
  concurrent_push(1);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push.2producers" * doctest::timeout(300)) {
  concurrent_push(2);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push.4producers" * doctest::timeout(300)) {
  concurrent_push(4);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push.8producers" * doctest::timeout(300)) {
  concurrent_push(8);
}

// ============================================================================
// Concurrent pop: pre-populated stack, M consumers drain it
// ============================================================================

void concurrent_pop(size_t M) {
  NodeStack stack;
  const size_t N = 10000;
  auto nodes = make_nodes(N);

  // pre-populate before any consumer starts
  for(auto* n : nodes) stack.push(n);

  std::atomic<size_t> consumed {0};
  std::vector<std::thread> threads;
  std::vector<std::vector<Node*>> stolen(M);

  for(size_t t = 0; t < M; ++t) {
    threads.emplace_back([&, t](){
      while(consumed.load(std::memory_order_acquire) < N) {
        if(auto* n = stack.pop(); n) {
          stolen[t].push_back(n);
          consumed.fetch_add(1, std::memory_order_release);
        }
      }
    });
  }
  for(auto& th : threads) th.join();

  REQUIRE(stack.empty() == true);
  REQUIRE(stack.pop() == nullptr);

  std::vector<int> values;
  for(size_t t = 0; t < M; ++t) {
    for(auto* n : stolen[t]) values.push_back(n->value);
  }
  REQUIRE(values.size() == N);
  std::sort(values.begin(), values.end());
  for(size_t i = 0; i < N; ++i) {
    REQUIRE(values[i] == static_cast<int>(i));
  }
  free_nodes(nodes);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_pop.1consumer" * doctest::timeout(300)) {
  concurrent_pop(1);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_pop.2consumers" * doctest::timeout(300)) {
  concurrent_pop(2);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_pop.4consumers" * doctest::timeout(300)) {
  concurrent_pop(4);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_pop.8consumers" * doctest::timeout(300)) {
  concurrent_pop(8);
}

// ============================================================================
// Concurrent push + pop: M producers and M consumers simultaneously
// push() is safe to call concurrently with pop() since push only writes
// the link field before the CAS, and pop only reads it after a successful CAS
// ============================================================================

void concurrent_push_pop(size_t M) {
  NodeStack stack;
  const size_t N = 5000;
  std::vector<std::vector<Node*>> thread_nodes(M);
  std::atomic<size_t> consumed {0};
  const size_t total = M * N;
  std::vector<std::thread> threads;
  std::vector<std::vector<Node*>> stolen(M);

  // producers
  for(size_t t = 0; t < M; ++t) {
    thread_nodes[t] = make_nodes(N, static_cast<int>(t * N));
    threads.emplace_back([&, t](){
      for(auto* n : thread_nodes[t]) stack.push(n);
    });
  }

  // consumers
  for(size_t t = 0; t < M; ++t) {
    threads.emplace_back([&, t](){
      while(consumed.load(std::memory_order_acquire) < total) {
        if(auto* n = stack.pop(); n) {
          stolen[t].push_back(n);
          consumed.fetch_add(1, std::memory_order_release);
        }
      }
    });
  }

  for(auto& th : threads) th.join();

  // drain any remaining
  while(auto* n = stack.pop()) {
    stolen[0].push_back(n);
    consumed.fetch_add(1, std::memory_order_release);
  }

  std::vector<int> values;
  for(size_t t = 0; t < M; ++t) {
    for(auto* n : stolen[t]) values.push_back(n->value);
  }
  REQUIRE(values.size() == total);
  std::sort(values.begin(), values.end());
  for(size_t i = 0; i < total; ++i) {
    REQUIRE(values[i] == static_cast<int>(i));
  }
  for(auto& v : thread_nodes) free_nodes(v);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_push_pop.1thread" * doctest::timeout(300)) {
  concurrent_push_pop(1);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push_pop.2threads" * doctest::timeout(300)) {
  concurrent_push_pop(2);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push_pop.4threads" * doctest::timeout(300)) {
  concurrent_push_pop(4);
}
TEST_CASE("AtomicIntrusiveStack.concurrent_push_pop.8threads" * doctest::timeout(300)) {
  concurrent_push_pop(8);
}
