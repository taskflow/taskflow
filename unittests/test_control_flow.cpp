#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: Conditional Tasking
// --------------------------------------------------------

TEST_CASE("Cond.Types") {

  tf::Taskflow taskflow;

  auto explicit_c = [](){
    return 1;
  };

  auto implicit_c = []() -> int {
    return 2;
  };

  auto explicit_task = taskflow.emplace(explicit_c);
  auto implicit_task = taskflow.emplace(implicit_c);

  static_assert(tf::is_condition_task_v<decltype(explicit_c)>);
  static_assert(tf::is_condition_task_v<decltype(implicit_c)>);

  REQUIRE(explicit_task.type() == tf::TaskType::CONDITION);
  REQUIRE(implicit_task.type() == tf::TaskType::CONDITION);
}


void conditional_spawn(
  std::atomic<int>& counter,
  const int max_depth,
  int depth,
  tf::Subflow& subflow
)  {
  if(depth < max_depth) {
    for(int i=0; i<2; i++) {
      auto A = subflow.emplace([&](){ counter++; });
      auto B = subflow.emplace(
        [&, max_depth, depth=depth+1](tf::Subflow& sf2){
          conditional_spawn(counter, max_depth, depth, sf2);
      });
      auto C = subflow.emplace(
        [&, max_depth, depth=depth+1](tf::Subflow& sf2){
          conditional_spawn(counter, max_depth, depth, sf2);
        }
      );

      auto cond = subflow.emplace([depth](){
        if(depth%2) return 1;
        else return 0;
      }).precede(B, C);
      A.precede(cond);
    }
  }
}

void loop_cond(unsigned w) {

  tf::Executor executor(w);
  tf::Taskflow taskflow;

  int counter = -1;
  int state   = 0;

  auto A = taskflow.emplace([&] () { counter = 0; });
  auto B = taskflow.emplace([&] () mutable {
      REQUIRE((++counter % 100) == (++state % 100));
      return counter < 100 ? 0 : 1;
  });
  auto C = taskflow.emplace(
    [&] () {
      REQUIRE(counter == 100);
      counter = 0;
  });

  A.precede(B);
  B.precede(B, C);

  REQUIRE(A.num_strong_dependents() == 0);
  REQUIRE(A.num_weak_dependents() == 0);
  REQUIRE(A.num_dependents() == 0);

  REQUIRE(B.num_strong_dependents() == 1);
  REQUIRE(B.num_weak_dependents() == 1);
  REQUIRE(B.num_dependents() == 2);

  executor.run(taskflow).wait();
  REQUIRE(counter == 0);
  REQUIRE(state == 100);

  executor.run(taskflow);
  executor.run(taskflow);
  executor.run(taskflow);
  executor.run(taskflow);
  executor.run_n(taskflow, 10);
  executor.wait_for_all();

  REQUIRE(state == 1500);
}

TEST_CASE("LoopCond.1thread" * doctest::timeout(300)) {
  loop_cond(1);
}

TEST_CASE("LoopCond.2threads" * doctest::timeout(300)) {
  loop_cond(2);
}

TEST_CASE("LoopCond.3threads" * doctest::timeout(300)) {
  loop_cond(3);
}

TEST_CASE("LoopCond.4threads" * doctest::timeout(300)) {
  loop_cond(4);
}

// ----------------------------------------------------------------------------
// Testcase: FlipCoinCond
// ----------------------------------------------------------------------------
void flip_coin_cond(unsigned w) {

  tf::Taskflow taskflow;

  size_t rounds = 10000;
  size_t steps = 0;
  size_t total_steps = 0;
  double average_steps = 0.0;

  auto A = taskflow.emplace( [&](){ steps = 0; } );
  auto B = taskflow.emplace( [&](){ ++steps; return std::rand()%2; } );
  auto C = taskflow.emplace( [&](){ return std::rand()%2; } );
  auto D = taskflow.emplace( [&](){ return std::rand()%2; } );
  auto E = taskflow.emplace( [&](){ return std::rand()%2; } );
  auto F = taskflow.emplace( [&](){ return std::rand()%2; } );
  auto G = taskflow.emplace( [&]() mutable {
      //++N;  // a new round
      total_steps += steps;
      //avg = (double)accu/N;
      //std::cout << "round " << N << ": steps=" << steps
      //                           << " accumulated_steps=" << accu
      //                           << " average_steps=" << avg << '\n';
    }
  );

  A.precede(B).name("init");
  B.precede(C, B).name("flip-coin-1");
  C.precede(D, B).name("flip-coin-2");
  D.precede(E, B).name("flip-coin-3");
  E.precede(F, B).name("flip-coin-4");
  F.precede(G, B).name("flip-coin-5");

  //taskflow.dump(std::cout);

  tf::Executor executor(w);

  executor.run_n(taskflow, rounds).wait();

  average_steps = total_steps / (double)rounds;

  REQUIRE(std::fabs(average_steps-32.0)<1.0);

  //taskflow.dump(std::cout);
}

TEST_CASE("FlipCoinCond.1thread" * doctest::timeout(300)) {
  flip_coin_cond(1);
}

TEST_CASE("FlipCoinCond.2threads" * doctest::timeout(300)) {
  flip_coin_cond(2);
}

TEST_CASE("FlipCoinCond.3threads" * doctest::timeout(300)) {
  flip_coin_cond(3);
}

TEST_CASE("FlipCoinCond.4threads" * doctest::timeout(300)) {
  flip_coin_cond(4);
}

// ----------------------------------------------------------------------------
// Testcase: CyclicCondition
// ----------------------------------------------------------------------------
void cyclic_cond(unsigned w) {
  tf::Executor executor(w);

  //      ____________________
  //      |                  |
  //      v                  |
  // S -> A -> Branch -> many branches -> T
  //
  // Make sure each branch will be passed through exactly once
  // and the T (target) node will also be passed

  tf::Taskflow flow;
  auto S = flow.emplace([](){});

  int num_iterations = 0;
  const int total_iteration = 1000;
  auto A = flow.emplace([&](){ num_iterations ++; });
  S.precede(A);

  int sel = 0;
  bool pass_T = false;
  std::vector<bool> pass(total_iteration, false);
  auto T = flow.emplace([&](){
    REQUIRE(num_iterations == total_iteration); pass_T=true; }
  );
  auto branch = flow.emplace([&](){ return sel++; });
  A.precede(branch);
  for(size_t i=0; i<total_iteration; i++) {
    auto t = flow.emplace([&, i](){
      if(num_iterations < total_iteration) {
        REQUIRE(!pass[i]);
        pass[i] = true;
        return 0;
      }
      // The last node will come to here (last iteration)
      REQUIRE(!pass[i]);
      pass[i] = true;
      return 1;
    });
    branch.precede(t);
    t.precede(A);
    t.precede(T);
  }

  executor.run(flow).get();

  REQUIRE(pass_T);
  for(size_t i=0; i<pass.size(); i++) {
    REQUIRE(pass[i]);
  }
}

TEST_CASE("CyclicCond.1thread" * doctest::timeout(300)) {
  cyclic_cond(1);
}

TEST_CASE("CyclicCond.2threads" * doctest::timeout(300)) {
  cyclic_cond(2);
}

TEST_CASE("CyclicCond.3threads" * doctest::timeout(300)) {
  cyclic_cond(3);
}

TEST_CASE("CyclicCond.4threads" * doctest::timeout(300)) {
  cyclic_cond(4);
}

TEST_CASE("CyclicCond.5threads" * doctest::timeout(300)) {
  cyclic_cond(5);
}

TEST_CASE("CyclicCond.6threads" * doctest::timeout(300)) {
  cyclic_cond(6);
}

TEST_CASE("CyclicCond.7threads" * doctest::timeout(300)) {
  cyclic_cond(7);
}

TEST_CASE("CyclicCond.8threads" * doctest::timeout(300)) {
  cyclic_cond(8);
}

// ----------------------------------------------------------------------------
// BTreeCond
// ----------------------------------------------------------------------------
TEST_CASE("BTreeCondition" * doctest::timeout(300)) {
  for(unsigned w=1; w<=8; ++w) {
    for(int l=1; l<12; l++) {
      tf::Taskflow flow;
      std::vector<tf::Task> prev_tasks;
      std::vector<tf::Task> tasks;

      std::atomic<int> counter {0};
      int level = l;

      for(int i=0; i<level; i++) {
        tasks.clear();
        for(int j=0; j< (1<<i); j++) {
          if(i % 2 == 0) {
            tasks.emplace_back(flow.emplace([&](){ counter++; }) );
          }
          else {
            if(j%2) {
              tasks.emplace_back(flow.emplace([](){ return 1; }));
            }
            else {
              tasks.emplace_back(flow.emplace([](){ return 0; }));
            }
          }
        }

        for(size_t j=0; j<prev_tasks.size(); j++) {
          prev_tasks[j].precede(tasks[2*j]    );
          prev_tasks[j].precede(tasks[2*j + 1]);
        }
        tasks.swap(prev_tasks);
      }

      tf::Executor executor(w);
      executor.run(flow).wait();

      REQUIRE(counter == (1<<((level+1)/2)) - 1);
    }
  }
}

//             ---- > B
//             |
//  A -> Cond -
//             |
//             ---- > C

TEST_CASE("DynamicBTreeCondition" * doctest::timeout(300)) {
  for(unsigned w=1; w<=8; ++w) {
    std::atomic<int> counter {0};
    constexpr int max_depth = 6;
    tf::Taskflow flow;
    flow.emplace([&](tf::Subflow& subflow) {
      counter++;
      conditional_spawn(counter, max_depth, 0, subflow); }
    );
    tf::Executor executor(w);
    executor.run_n(flow, 4).get();
    // Each run increments the counter by (2^(max_depth+1) - 1)
    REQUIRE(counter.load() == ((1<<(max_depth+1)) - 1)*4);
  }
}

//        ______
//       |      |
//       v      |
//  S -> A -> cond

void nested_cond(unsigned w) {

  const int outer_loop = 3;
  const int mid_loop = 4;
  const int inner_loop = 5;

  int counter {0};
  tf::Taskflow flow;
  auto S = flow.emplace([](){});
  auto A = flow.emplace([&] (tf::Subflow& subflow) mutable {
    //           ___________
    //          |           |
    //          v           |
    //   S1 -> A1 -> B1 -> cond
    auto S1 = subflow.emplace([](){ });
    auto A1 = subflow.emplace([](){ }).succeed(S1);
    auto B1 = subflow.emplace([&](tf::Subflow& sf){

      //           ___________
      //          |           |
      //          v           |
      //   S2 -> A2 -> B2 -> cond
      //          |
      //          -----> C
      //          -----> D
      //          -----> E

      auto S2 = sf.emplace([](){});
      auto A2 = sf.emplace([](){}).succeed(S2);
      auto B2 = sf.emplace([&](){ counter++; }).succeed(A2);
      sf.emplace([&, repeat=0]() mutable {
        if(repeat ++ < inner_loop)
          return 0;

        repeat = 0;
        return 1;
      }).succeed(B2).precede(A2).name("cond");

      // Those are redundant tasks
      sf.emplace([](){}).succeed(A2).name("C");
      sf.emplace([](){}).succeed(A2).name("D");
      sf.emplace([](){}).succeed(A2).name("E");
    }).succeed(A1);
    subflow.emplace([&, repeat=0]() mutable {
      if(repeat ++ < mid_loop)
        return 0;

      repeat = 0;
      return 1;
    }).succeed(B1).precede(A1).name("cond");

  }).succeed(S);

  flow.emplace(
    [&, repeat=0]() mutable {
      if(repeat ++ < outer_loop) {
        return 0;
      }

      repeat = 0;
      return 1;
    }
  ).succeed(A).precede(A);

  tf::Executor executor(w);
  const int repeat = 10;
  executor.run_n(flow, repeat).get();

  REQUIRE(counter == (inner_loop+1)*(mid_loop+1)*(outer_loop+1)*repeat);
}

TEST_CASE("NestedCond.1thread" * doctest::timeout(300)) {
  nested_cond(1);
}

TEST_CASE("NestedCond.2threads" * doctest::timeout(300)) {
  nested_cond(2);
}

TEST_CASE("NestedCond.3threads" * doctest::timeout(300)) {
  nested_cond(3);
}

TEST_CASE("NestedCond.4threads" * doctest::timeout(300)) {
  nested_cond(4);
}

TEST_CASE("NestedCond.5threads" * doctest::timeout(300)) {
  nested_cond(5);
}

TEST_CASE("NestedCond.6threads" * doctest::timeout(300)) {
  nested_cond(6);
}

TEST_CASE("NestedCond.7threads" * doctest::timeout(300)) {
  nested_cond(7);
}

TEST_CASE("NestedCond.8threads" * doctest::timeout(300)) {
  nested_cond(8);
}

//         ________________
//        |  ___   ______  |
//        | |   | |      | |
//        v v   | v      | |
//   S -> A -> cond1 -> cond2 -> D
//               |
//                ----> B

void cond2cond(unsigned w) {

  const int repeat = 10;
  tf::Taskflow flow;

  int num_visit_A {0};
  int num_visit_C1 {0};
  int num_visit_C2 {0};

  int iteration_C1 {0};
  int iteration_C2 {0};

  auto S = flow.emplace([](){});
  auto A = flow.emplace([&](){ num_visit_A++; }).succeed(S);
  auto cond1 = flow.emplace([&]() mutable {
    num_visit_C1++;
    iteration_C1++;
    if(iteration_C1 == 1) return 0;
    return 1;
  }).succeed(A).precede(A);

  auto cond2 = flow.emplace([&]() mutable {
    num_visit_C2 ++;
    return iteration_C2++;
  }).succeed(cond1).precede(cond1, A);

  flow.emplace([](){ REQUIRE(false); }).succeed(cond1).name("B");
  flow.emplace([&](){
    iteration_C1 = 0;
    iteration_C2 = 0;
  }).succeed(cond2).name("D");

  tf::Executor executor(w);
  executor.run_n(flow, repeat).get();

  REQUIRE(num_visit_A  == 3*repeat);
  REQUIRE(num_visit_C1 == 4*repeat);
  REQUIRE(num_visit_C2 == 3*repeat);

}

TEST_CASE("Cond2Cond.1thread" * doctest::timeout(300)) {
  cond2cond(1);
}

TEST_CASE("Cond2Cond.2threads" * doctest::timeout(300)) {
  cond2cond(2);
}

TEST_CASE("Cond2Cond.3threads" * doctest::timeout(300)) {
  cond2cond(3);
}

TEST_CASE("Cond2Cond.4threads" * doctest::timeout(300)) {
  cond2cond(4);
}

TEST_CASE("Cond2Cond.5threads" * doctest::timeout(300)) {
  cond2cond(5);
}

TEST_CASE("Cond2Cond.6threads" * doctest::timeout(300)) {
  cond2cond(6);
}

TEST_CASE("Cond2Cond.7threads" * doctest::timeout(300)) {
  cond2cond(7);
}

TEST_CASE("Cond2Cond.8threads" * doctest::timeout(300)) {
  cond2cond(8);
}


void hierarchical_condition(unsigned w) {

  tf::Executor executor(w);
  tf::Taskflow tf0("c0");
  tf::Taskflow tf1("c1");
  tf::Taskflow tf2("c2");
  tf::Taskflow tf3("top");

  int c1, c2, c2_repeat;

  auto c1A = tf1.emplace( [&](){ c1=0; } );
  auto c1B = tf1.emplace( [&, state=0] () mutable {
    REQUIRE(state++ % 100 == c1 % 100);
  });
  auto c1C = tf1.emplace( [&](){ return (++c1 < 100) ? 0 : 1; });

  c1A.precede(c1B);
  c1B.precede(c1C);
  c1C.precede(c1B);
  c1A.name("c1A");
  c1B.name("c1B");
  c1C.name("c1C");

  auto c2A = tf2.emplace( [&](){ REQUIRE(c2 == 100); c2 = 0; } );
  auto c2B = tf2.emplace( [&, state=0] () mutable {
      REQUIRE((state++ % 100) == (c2 % 100));
  });
  auto c2C = tf2.emplace( [&](){ return (++c2 < 100) ? 0 : 1; });

  c2A.precede(c2B);
  c2B.precede(c2C);
  c2C.precede(c2B);
  c2A.name("c2A");
  c2B.name("c2B");
  c2C.name("c2C");

  auto init = tf3.emplace([&](){
    c1=c2=c2_repeat=0;
  }).name("init");

  auto loop1 = tf3.emplace([&](){
    return (++c2 < 100) ? 0 : 1;
  }).name("loop1");

  auto loop2 = tf3.emplace([&](){
    c2 = 0;
    return ++c2_repeat < 100 ? 0 : 1;
  }).name("loop2");

  auto sync = tf3.emplace([&](){
    REQUIRE(c2==0);
    REQUIRE(c2_repeat==100);
    c2_repeat = 0;
  }).name("sync");

  auto grab = tf3.emplace([&](){
    REQUIRE(c1 == 100);
    REQUIRE(c2 == 0);
    REQUIRE(c2_repeat == 0);
  }).name("grab");

  auto mod0 = tf3.composed_of(tf0).name("module0");
  auto mod1 = tf3.composed_of(tf1).name("module1");
  auto sbf1 = tf3.emplace([&](tf::Subflow& sbf){
    auto sbf1_1 = sbf.emplace([](){}).name("sbf1_1");
    auto module1 = sbf.composed_of(tf1).name("module1");
    auto sbf1_2 = sbf.emplace([](){}).name("sbf1_2");
    sbf1_1.precede(module1);
    module1.precede(sbf1_2);
    sbf.join();
  }).name("sbf1");
  auto mod2 = tf3.composed_of(tf2).name("module2");

  init.precede(mod0, sbf1, loop1);
  loop1.precede(loop1, mod2);
  loop2.succeed(mod2).precede(loop1, sync);
  mod0.precede(grab);
  sbf1.precede(mod1);
  mod1.precede(grab);
  sync.precede(grab);

  executor.run(tf3);
  executor.run_n(tf3, 10);
  executor.wait_for_all();

  //tf3.dump(std::cout);
}

TEST_CASE("HierCondition.1thread" * doctest::timeout(300)) {
  hierarchical_condition(1);
}

TEST_CASE("HierCondition.2threads" * doctest::timeout(300)) {
  hierarchical_condition(2);
}

TEST_CASE("HierCondition.3threads" * doctest::timeout(300)) {
  hierarchical_condition(3);
}

TEST_CASE("HierCondition.4threads" * doctest::timeout(300)) {
  hierarchical_condition(4);
}

TEST_CASE("HierCondition.5threads" * doctest::timeout(300)) {
  hierarchical_condition(5);
}

TEST_CASE("HierCondition.6threads" * doctest::timeout(300)) {
  hierarchical_condition(6);
}

TEST_CASE("HierCondition.7threads" * doctest::timeout(300)) {
  hierarchical_condition(7);
}

TEST_CASE("HierCondition.8threads" * doctest::timeout(300)) {
  hierarchical_condition(8);
}

// ----------------------------------------------------------------------------
// CondSubflow
// ----------------------------------------------------------------------------

void condition_subflow(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  const size_t I = 1000;

  std::vector<size_t> data(I);

  size_t i;

  auto init = taskflow.emplace([&](){ i = 0; }).name("init");

  auto subflow = taskflow.emplace([&](tf::Subflow& sf){
    sf.emplace([&, i](){
      REQUIRE(i<I);
      data[i] = i*(i+1)/2*123;;
    }).name(std::to_string(i));
    sf.detach();
  }).name("subflow");

  auto cond = taskflow.emplace([&](){
    if(++i < I) return 0;
    return 1;
  }).name("cond");

  auto stop = taskflow.emplace([](){}).name("stop");

  init.precede(subflow);
  subflow.precede(cond);
  cond.precede(subflow, stop);

  executor.run(taskflow).wait();

  REQUIRE(taskflow.num_tasks() == 4 + I);

  for(size_t j=0; j<data.size(); ++j) {
    REQUIRE(data[j] == j*(j+1)/2*123);
    data[j] = 0;
  }

  executor.run_n(taskflow, 1);
  executor.run_n(taskflow, 10);
  executor.run_n(taskflow, 100);

  executor.wait_for_all();

  REQUIRE(taskflow.num_tasks() == 4 + I*100);

  for(size_t j=0; j<data.size(); ++j) {
    REQUIRE(data[j] == j*(j+1)/2*123);
  }

}

TEST_CASE("CondSubflow.1thread") {
  condition_subflow(1);
}

TEST_CASE("CondSubflow.2threads") {
  condition_subflow(2);
}

TEST_CASE("CondSubflow.3threads") {
  condition_subflow(3);
}

TEST_CASE("CondSubflow.4threads") {
  condition_subflow(4);
}

TEST_CASE("CondSubflow.5threads") {
  condition_subflow(5);
}

TEST_CASE("CondSubflow.6threads") {
  condition_subflow(6);
}

TEST_CASE("CondSubflow.7threads") {
  condition_subflow(7);
}

TEST_CASE("CondSubflow.8threads") {
  condition_subflow(8);
}

// ----------------------------------------------------------------------------
// Multi-conditional tasking
// ----------------------------------------------------------------------------

TEST_CASE("MultiCond.Types") {

  tf::Taskflow taskflow;

  auto explicit_mc = [](){
    tf::SmallVector<int> v;
    return v;
  };

  auto implicit_mc = []() -> tf::SmallVector<int> {
    return {1, 2, 3, 9};
  };

  auto explicit_task = taskflow.emplace(explicit_mc);
  auto implicit_task = taskflow.emplace(implicit_mc);

  static_assert(tf::is_multi_condition_task_v<decltype(explicit_mc)>);
  static_assert(tf::is_multi_condition_task_v<decltype(implicit_mc)>);

  REQUIRE(explicit_task.type() == tf::TaskType::CONDITION);
  REQUIRE(implicit_task.type() == tf::TaskType::CONDITION);
}

// ----------------------------------------------------------------------------
// Testcase: Multiple Branches
// ----------------------------------------------------------------------------

void multiple_branches(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto A = taskflow.placeholder();

  for(int i=0; i<100; i++) {
    auto X = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    auto Y = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    X.precede(Y);
    A.precede(X);
  }

  int ans = 0;
  tf::SmallVector<int> conds;

  for(int i=-10; i<=110; i++) {
    if(::rand() % 10 == 0) {
      conds.push_back(i);
      if(0<=i && i<100) {
        ans++;
      }
    }
  }

  A.work([&]() { return conds; });

  executor.run(taskflow).wait();

  REQUIRE(2*ans == counter);
}

TEST_CASE("MultipleBranches.1thread") {
  multiple_branches(1);
}

TEST_CASE("MultipleBranches.2threads") {
  multiple_branches(2);
}

TEST_CASE("MultipleBranches.3threads") {
  multiple_branches(3);
}

TEST_CASE("MultipleBranches.4threads") {
  multiple_branches(4);
}

TEST_CASE("MultipleBranches.5threads") {
  multiple_branches(5);
}

TEST_CASE("MultipleBranches.6threads") {
  multiple_branches(6);
}

TEST_CASE("MultipleBranches.7threads") {
  multiple_branches(7);
}

TEST_CASE("MultipleBranches.8threads") {
  multiple_branches(8);
}

// ----------------------------------------------------------------------------
// Testcase: Multiple Loops
// ----------------------------------------------------------------------------

void multiple_loops(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter{0};

  auto A = taskflow.emplace([](){});
  auto B = taskflow.emplace([&, i=bool{true}, c = int(0)]() mutable -> tf::SmallVector<int> {
    if(i) {
      i = false;
      return {0, 1};
    }
    else {
      counter.fetch_add(1, std::memory_order_relaxed);
      return {++c < 10 ? 0 : -1};
    }
  });
  auto C = taskflow.emplace([&, i=bool{true}, c = int(0)]() mutable -> tf::SmallVector<int> {
    if(i) {
      i = false;
      return {0, 1};
    }
    else {
      counter.fetch_add(1, std::memory_order_relaxed);
      return {++c < 10 ? 0 : -1};
    }
  });
  auto D = taskflow.emplace([&, i=bool{true}, c = int(0)]() mutable -> tf::SmallVector<int> {
    if(i) {
      i = false;
      return {0, 1};
    }
    else {
      counter.fetch_add(1, std::memory_order_relaxed);
      return {++c < 10 ? 0 : -1};
    }
  });
  auto E = taskflow.emplace([&, i=bool{true}, c = int(0)]() mutable -> tf::SmallVector<int> {
    if(i) {
      i = false;
      return {0, 1};
    }
    else {
      counter.fetch_add(1, std::memory_order_relaxed);
      return {++c < 10 ? 0 : -1};
    }
  });

  A.precede(B);
  B.precede(B, C);
  C.precede(C, D);
  D.precede(D, E);
  E.precede(E);

  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);

  REQUIRE(counter == 40);
}

TEST_CASE("MultipleLoops.1thread") {
  multiple_loops(1);
}

TEST_CASE("MultipleLoops.2threads") {
  multiple_loops(2);
}

TEST_CASE("MultipleLoops.3threads") {
  multiple_loops(3);
}

TEST_CASE("MultipleLoops.4threads") {
  multiple_loops(4);
}

TEST_CASE("MultipleLoops.5threads") {
  multiple_loops(5);
}

TEST_CASE("MultipleLoops.6threads") {
  multiple_loops(6);
}

TEST_CASE("MultipleLoops.7threads") {
  multiple_loops(7);
}

TEST_CASE("MultipleLoops.8threads") {
  multiple_loops(8);
}

// ----------------------------------------------------------------------------
// binary tree
// ----------------------------------------------------------------------------

void binary_tree(unsigned w) {

  const int N = 10;

  tf::Taskflow taskflow;
  tf::Executor executor(w);

  std::atomic<int> counter{0};
  std::vector<tf::Task> tasks;

  for(int i=1; i<(1<<N); i++) {
    tasks.emplace_back(taskflow.emplace([&counter]() -> tf::SmallVector<int>{
      counter.fetch_add(1, std::memory_order_relaxed);
      return {0, 1};
    }));
  }

  for(size_t i=0; i<tasks.size(); i++) {
    size_t l = i*2+1;
    size_t r = l + 1;
    if(l < tasks.size()) tasks[i].precede(tasks[l]);
    if(r < tasks.size()) tasks[i].precede(tasks[r]);
  }

  executor.run_n(taskflow, N).wait();

  REQUIRE(((1<<N)-1)*N == counter);
}

TEST_CASE("MultiCondBinaryTree.1thread") {
  binary_tree(1);
}

TEST_CASE("MultiCondBinaryTree.2threads") {
  binary_tree(2);
}

TEST_CASE("MultiCondBinaryTree.3threads") {
  binary_tree(3);
}

TEST_CASE("MultiCondBinaryTree.4threads") {
  binary_tree(4);
}

































