#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/core/freelist.hpp>

// ----------------------------------------------------------------------------
// Test node types
// ----------------------------------------------------------------------------

// Basic node using next as the intrusive link — mirrors Taskflow's Node
struct Node {
  Node* next {nullptr};
  int   value {0};
};

// Node using a differently-named field — verifies MemberPtr generality
struct Item {
  Item* next_free {nullptr};
  int   id {0};
};

using NodeStack = tf::AtomicIntrusiveStack<Node*, &Node::next>;
using ItemStack = tf::AtomicIntrusiveStack<Item*, &Item::next_free>;

// ----------------------------------------------------------------------------
// Helper: build a flat array of N heap-allocated nodes
// ----------------------------------------------------------------------------
static std::vector<Node*> make_nodes(int N, int base_value = 0) {
  std::vector<Node*> v(N);
  for(int i = 0; i < N; ++i) {
    v[i] = new Node{nullptr, base_value + i};
  }
  return v;
}

static void free_nodes(std::vector<Node*>& v) {
  for(auto* n : v) delete n;
  v.clear();
}

// ============================================================================
// Basic single-threaded correctness
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
  // pop from empty stack must return nullptr, never crash
  REQUIRE(stack.pop() == nullptr);
  REQUIRE(stack.pop() == nullptr);
}

TEST_CASE("AtomicIntrusiveStack.push_pop_lifo" * doctest::timeout(10)) {
  NodeStack stack;
  auto nodes = make_nodes(5);

  // push 0..4
  for(auto* n : nodes) stack.push(n);

  // pop must be LIFO: 4, 3, 2, 1, 0
  for(int i = 4; i >= 0; --i) {
    auto* n = stack.pop();
    REQUIRE(n != nullptr);
    REQUIRE(n->value == i);
    REQUIRE(n->next == nullptr);  // link cleared after pop
  }
  REQUIRE(stack.pop() == nullptr);
  free_nodes(nodes);
}

TEST_CASE("AtomicIntrusiveStack.link_cleared_after_pop" * doctest::timeout(10)) {
  NodeStack stack;
  Node a{nullptr, 1}, b{nullptr, 2};

  stack.push(&a);
  stack.push(&b);

  // b.next was set to &a by push — must be nullptr after pop
  Node* n = stack.pop();
  REQUIRE(n == &b);
  REQUIRE(n->next == nullptr);

  n = stack.pop();
  REQUIRE(n == &a);
  REQUIRE(n->next == nullptr);
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

// ============================================================================
// bulk_push
// ============================================================================

TEST_CASE("AtomicIntrusiveStack.bulk_push_zero" * doctest::timeout(10)) {
  NodeStack stack;
  Node* arr[1] = {nullptr};
  // bulk_push with N=0 must be a no-op
  stack.bulk_push(arr, 0);
  REQUIRE(stack.empty() == true);
}

TEST_CASE("AtomicIntrusiveStack.bulk_push_order" * doctest::timeout(10)) {
  NodeStack stack;
  auto nodes = make_nodes(5);
  Node* arr[5];
  for(int i = 0; i < 5; ++i) arr[i] = nodes[i];

  stack.bulk_push(arr, 5);

  // bulk_push puts first element on top
  // expected pop order: nodes[0], nodes[1], ..., nodes[4]
  for(int i = 0; i < 5; ++i) {
    auto* n = stack.pop();
    REQUIRE(n != nullptr);
    REQUIRE(n->value == i);
    REQUIRE(n->next == nullptr);
  }
  REQUIRE(stack.pop() == nullptr);
  free_nodes(nodes);
}

TEST_CASE("AtomicIntrusiveStack.bulk_push_single" * doctest::timeout(10)) {
  NodeStack stack;
  Node n{nullptr, 7};
  Node* arr[1] = {&n};

  stack.bulk_push(arr, 1);
  REQUIRE(stack.pop() == &n);
  REQUIRE(stack.pop() == nullptr);
}

TEST_CASE("AtomicIntrusiveStack.bulk_push_then_push" * doctest::timeout(10)) {
  NodeStack stack;
  auto nodes = make_nodes(3);
  Node* arr[3] = {nodes[0], nodes[1], nodes[2]};

  stack.bulk_push(arr, 3);

  Node extra{nullptr, 99};
  stack.push(&extra);

  // extra was pushed last so it's on top
  REQUIRE(stack.pop() == &extra);
  REQUIRE(stack.pop() == nodes[0]);
  REQUIRE(stack.pop() == nodes[1]);
  REQUIRE(stack.pop() == nodes[2]);
  REQUIRE(stack.pop() == nullptr);
  free_nodes(nodes);
}

// ============================================================================
// Different member pointer (Item::next_free)
// ============================================================================

TEST_CASE("AtomicIntrusiveStack.custom_member_ptr" * doctest::timeout(10)) {
  ItemStack stack;
  REQUIRE(stack.empty() == true);

  Item a{nullptr, 1}, b{nullptr, 2}, c{nullptr, 3};
  stack.push(&a);
  stack.push(&b);
  stack.push(&c);

  auto* r = stack.pop();
  REQUIRE(r == &c);
  REQUIRE(r->next_free == nullptr);

  r = stack.pop();
  REQUIRE(r == &b);
  REQUIRE(r->next_free == nullptr);

  r = stack.pop();
  REQUIRE(r == &a);
  REQUIRE(r->next_free == nullptr);

  REQUIRE(stack.pop() == nullptr);
}

// ============================================================================
// Concurrent push: M producers push N nodes each, owner pops all
// ============================================================================

void concurrent_push(size_t M) {
  NodeStack stack;
  const size_t N = 1000;

  // each producer thread pushes N nodes
  std::vector<std::vector<Node*>> thread_nodes(M);
  std::vector<std::thread> threads;

  for(size_t t = 0; t < M; ++t) {
    thread_nodes[t] = make_nodes(N, static_cast<int>(t * N));
    threads.emplace_back([&, t](){
      for(auto* n : thread_nodes[t]) {
        stack.push(n);
      }
    });
  }
  for(auto& th : threads) th.join();

  // pop all nodes and verify we got exactly M*N distinct nodes
  std::vector<int> values;
  while(auto* n = stack.pop()) {
    REQUIRE(n->next == nullptr);  // link always cleared
    values.push_back(n->value);
  }
  REQUIRE(values.size() == M * N);

  // verify all values are present (no duplicates, no missing)
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
// Concurrent pop: 1 producer pushes N nodes, M consumers pop them all
// ============================================================================

void concurrent_pop(size_t M) {
  NodeStack stack;
  const size_t N = 10000;
  auto nodes = make_nodes(N);

  // pre-populate the stack
  for(auto* n : nodes) stack.push(n);

  std::atomic<size_t> consumed {0};
  std::vector<std::thread> threads;
  std::vector<std::vector<Node*>> stolen(M);

  for(size_t t = 0; t < M; ++t) {
    threads.emplace_back([&, t](){
      while(consumed.load(std::memory_order_relaxed) < N) {
        if(auto* n = stack.pop(); n) {
          REQUIRE(n->next == nullptr);
          stolen[t].push_back(n);
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for(auto& th : threads) th.join();

  REQUIRE(stack.empty() == true);
  REQUIRE(stack.pop() == nullptr);

  // merge and verify all N nodes were consumed exactly once
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
// ============================================================================

void concurrent_push_pop(size_t M) {
  NodeStack stack;
  const size_t N = 5000;  // nodes per producer

  std::vector<std::vector<Node*>> thread_nodes(M);
  std::atomic<size_t> consumed {0};
  const size_t total = M * N;

  std::vector<std::thread> threads;
  std::vector<std::vector<Node*>> stolen(M);

  // producers
  for(size_t t = 0; t < M; ++t) {
    thread_nodes[t] = make_nodes(N, static_cast<int>(t * N));
    threads.emplace_back([&, t](){
      for(auto* n : thread_nodes[t]) {
        stack.push(n);
      }
    });
  }

  // consumers
  for(size_t t = 0; t < M; ++t) {
    threads.emplace_back([&, t](){
      while(consumed.load(std::memory_order_relaxed) < total) {
        if(auto* n = stack.pop(); n) {
          REQUIRE(n->next == nullptr);
          stolen[t].push_back(n);
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for(auto& th : threads) th.join();

  // drain any remaining nodes
  while(auto* n = stack.pop()) {
    stolen[0].push_back(n);
    consumed.fetch_add(1, std::memory_order_relaxed);
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

// ============================================================================
// Concurrent bulk_push + pop
// ============================================================================

void concurrent_bulk_push_pop(size_t M) {
  NodeStack stack;
  const size_t N = 1000;  // nodes per bulk push

  std::vector<std::vector<Node*>> thread_nodes(M);
  std::atomic<size_t> consumed {0};
  const size_t total = M * N;

  std::vector<std::thread> threads;
  std::vector<std::vector<Node*>> stolen(M);

  // producers do bulk_push
  for(size_t t = 0; t < M; ++t) {
    thread_nodes[t] = make_nodes(N, static_cast<int>(t * N));
    threads.emplace_back([&, t](){
      stack.bulk_push(thread_nodes[t].data(), N);
    });
  }

  // consumers do pop
  for(size_t t = 0; t < M; ++t) {
    threads.emplace_back([&, t](){
      while(consumed.load(std::memory_order_relaxed) < total) {
        if(auto* n = stack.pop(); n) {
          REQUIRE(n->next == nullptr);
          stolen[t].push_back(n);
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for(auto& th : threads) th.join();

  while(auto* n = stack.pop()) {
    stolen[0].push_back(n);
    consumed.fetch_add(1, std::memory_order_relaxed);
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

TEST_CASE("AtomicIntrusiveStack.concurrent_bulk_push_pop.1thread" * doctest::timeout(300)) {
  concurrent_bulk_push_pop(1);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_bulk_push_pop.2threads" * doctest::timeout(300)) {
  concurrent_bulk_push_pop(2);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_bulk_push_pop.4threads" * doctest::timeout(300)) {
  concurrent_bulk_push_pop(4);
}

TEST_CASE("AtomicIntrusiveStack.concurrent_bulk_push_pop.8threads" * doctest::timeout(300)) {
  concurrent_bulk_push_pop(8);
}
