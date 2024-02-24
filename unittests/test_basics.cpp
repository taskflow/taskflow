#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/reduce.hpp>

// --------------------------------------------------------
// Testcase: Type
// --------------------------------------------------------
TEST_CASE("Type" * doctest::timeout(300)) {

  tf::Taskflow taskflow, taskflow2;

  auto t1 = taskflow.emplace([](){});
  auto t2 = taskflow.emplace([](){ return 1; });
  auto t3 = taskflow.emplace([](tf::Subflow&){ });
  auto t4 = taskflow.composed_of(taskflow2);
  auto t5 = taskflow.emplace([](){ return tf::SmallVector{1, 2}; });
  auto t6 = taskflow.emplace([](tf::Runtime&){});
  auto t7 = taskflow.emplace([](tf::Runtime&){ return 1; });
  auto t8 = taskflow.emplace([](tf::Runtime&){ return tf::SmallVector{1, 2}; });

  REQUIRE(t1.type() == tf::TaskType::STATIC);
  REQUIRE(t2.type() == tf::TaskType::CONDITION);
  REQUIRE(t3.type() == tf::TaskType::SUBFLOW);
  REQUIRE(t4.type() == tf::TaskType::MODULE);
  REQUIRE(t5.type() == tf::TaskType::CONDITION);
  REQUIRE(t6.type() == tf::TaskType::STATIC);
  REQUIRE(t7.type() == tf::TaskType::CONDITION);
  REQUIRE(t8.type() == tf::TaskType::CONDITION);
}

// --------------------------------------------------------
// Testcase: Builder
// --------------------------------------------------------
TEST_CASE("Builder" * doctest::timeout(300)) {

  SUBCASE("EmptyFlow") {
    for(unsigned W=1; W<32; ++W) {
      tf::Executor executor(W);
      tf::Taskflow taskflow;
      REQUIRE(taskflow.num_tasks() == 0);
      REQUIRE(taskflow.empty() == true);
      executor.run(taskflow).wait();
    }
  }
  
  const size_t num_tasks = 100;

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
  std::vector<tf::Task> tasks;

  SUBCASE("Placeholder") {

    for(size_t i=0; i<num_tasks; ++i) {
      silent_tasks.emplace_back(taskflow.placeholder().name(std::to_string(i)));
    }

    for(size_t i=0; i<num_tasks; ++i) {
      REQUIRE(silent_tasks[i].name() == std::to_string(i));
      REQUIRE(silent_tasks[i].num_dependents() == 0);
      REQUIRE(silent_tasks[i].num_successors() == 0);
    }

    for(auto& task : silent_tasks) {
      task.work([&counter](){ counter++; });
    }

    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("EmbarrassinglyParallel"){

    for(size_t i=0;i<num_tasks;i++) {
      tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(taskflow.num_tasks() == num_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_tasks() == 100);

    counter = 0;

    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
    }

    REQUIRE(taskflow.num_tasks() == num_tasks * 2);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks * 2);
    REQUIRE(taskflow.num_tasks() == 200);
  }

  SUBCASE("BinarySequence"){
    for(size_t i=0;i<num_tasks;i++){
      if(i%2 == 0){
        tasks.emplace_back(
          taskflow.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;})
        );
      }
      else{
        tasks.emplace_back(
          taskflow.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;})
        );
      }
      if(i>0){
        //tasks[i-1].first.precede(tasks[i].first);
        tasks[i-1].precede(tasks[i]);
      }

      if(i==0) {
        //REQUIRE(tasks[i].first.num_dependents() == 0);
        REQUIRE(tasks[i].num_dependents() == 0);
      }
      else {
        //REQUIRE(tasks[i].first.num_dependents() == 1);
        REQUIRE(tasks[i].num_dependents() == 1);
      }
    }
    executor.run(taskflow).get();
  }

  SUBCASE("LinearCounter"){
    for(size_t i=0;i<num_tasks;i++){
      tasks.emplace_back(
        taskflow.emplace([&counter, i]() {
          REQUIRE(counter == i); counter += 1;}
        )
      );
      if(i>0) {
        tasks[i-1].precede(tasks[i]);
      }
    }
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_tasks() == num_tasks);
  }

  SUBCASE("Broadcast"){
    auto src = taskflow.emplace([&counter]() {counter -= 1;});
    for(size_t i=1; i<num_tasks; i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter]() {REQUIRE(counter == -1);})
      );
      src.precede(silent_tasks.back());
    }
    executor.run(taskflow).get();
    REQUIRE(counter == - 1);
    REQUIRE(taskflow.num_tasks() == num_tasks);
  }

  SUBCASE("Succeed"){
    auto dst = taskflow.emplace([&]() { REQUIRE(counter == num_tasks - 1);});
    for(size_t i=1;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter]() {counter += 1;})
      );
      dst.succeed(silent_tasks.back());
    }
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(taskflow.num_tasks() == num_tasks);
  }

  SUBCASE("MapReduce"){

    auto src = taskflow.emplace([&counter]() {counter = 0;});

    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );

    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter]() {counter += 1;})
      );
      src.precede(silent_tasks.back());
      dst.succeed(silent_tasks.back());
    }
    executor.run(taskflow).get();
    REQUIRE(taskflow.num_tasks() == num_tasks + 2);
  }

  SUBCASE("Linearize"){
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter, i]() {
          REQUIRE(counter == i); counter += 1;}
        )
      );
    }
    taskflow.linearize(silent_tasks);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks);
    REQUIRE(taskflow.num_tasks() == num_tasks);
  }

  SUBCASE("Kite"){
    auto src = taskflow.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter, i]() {
          REQUIRE(counter == i); counter += 1; }
        )
      );
      src.precede(silent_tasks.back());
    }
    taskflow.linearize(silent_tasks);
    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );

    for(auto task : silent_tasks) dst.succeed(task);
    executor.run(taskflow).get();
    REQUIRE(taskflow.num_tasks() == num_tasks + 2);
  }
}

// --------------------------------------------------------
// Testcase: Creation
// --------------------------------------------------------
TEST_CASE("Creation" * doctest::timeout(300)) {

  std::vector<int> dummy(1000, -1);

  auto create_taskflow = [&] () {
    for(int i=0; i<10; ++i) {
      tf::Taskflow tf;
      tf.for_each(dummy.begin(), dummy.end(), [] (int) {});
    }
  };

  SUBCASE("One") {
    create_taskflow();
    REQUIRE(dummy.size() == 1000);
    for(auto item : dummy) {
      REQUIRE(item == -1);
    }
  }

  SUBCASE("Two") {
    std::thread t1(create_taskflow);
    std::thread t2(create_taskflow);
    t1.join();
    t2.join();
    REQUIRE(dummy.size() == 1000);
    for(auto item : dummy) {
      REQUIRE(item == -1);
    }
  }

  SUBCASE("Four") {
    std::thread t1(create_taskflow);
    std::thread t2(create_taskflow);
    std::thread t3(create_taskflow);
    std::thread t4(create_taskflow);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    REQUIRE(dummy.size() == 1000);
    for(auto item : dummy) {
      REQUIRE(item == -1);
    }
  }

}

// --------------------------------------------------------
// Testcase: Removal
// --------------------------------------------------------
TEST_CASE("Removal" * doctest::timeout(300)) {
  
  tf::Taskflow taskflow;
  auto a = taskflow.placeholder().name("a");
  auto b = taskflow.placeholder().name("b");
  auto c = taskflow.placeholder().name("c");
  auto d = taskflow.placeholder().name("d");

  REQUIRE(a.num_successors() == 0);
  REQUIRE(a.num_dependents() == 0);
  REQUIRE(a.num_successors() == 0);
  REQUIRE(a.num_dependents() == 0);

  a.precede(b, c, d);
  REQUIRE(a.num_successors() == 3);
  REQUIRE(b.num_dependents() == 1);
  REQUIRE(c.num_dependents() == 1);
  REQUIRE(d.num_dependents() == 1);

  taskflow.remove_dependency(a, b);
  REQUIRE(a.num_successors() == 2);
  REQUIRE(b.num_dependents() == 0);

  taskflow.remove_dependency(a, c);
  REQUIRE(a.num_successors() == 1);
  REQUIRE(c.num_dependents() == 0);
  
  taskflow.remove_dependency(a, d);
  REQUIRE(a.num_successors() == 0);
  REQUIRE(d.num_dependents() == 0);

  a.precede(b, b, c, c, d, d);
  REQUIRE(a.num_successors() == 6);
  REQUIRE(b.num_dependents() == 2);

  taskflow.remove_dependency(a, b);
  REQUIRE(a.num_successors() == 4);
  REQUIRE(b.num_dependents() == 0);

  taskflow.remove_dependency(a, c);
  REQUIRE(a.num_successors() == 2);
  REQUIRE(b.num_dependents() == 0);
  
  taskflow.remove_dependency(a, d);
  REQUIRE(a.num_successors() == 0);
  REQUIRE(d.num_dependents() == 0);
}


// --------------------------------------------------------
// Testcase: STDFunction
// --------------------------------------------------------
TEST_CASE("STDFunction" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  int counter = 0;

  std::function<void()> func1  = [&] () { ++counter; };
  std::function<int()>  func2  = [&] () { ++counter; return 0; };
  std::function<void()> func3  = [&] () { };
  std::function<void()> func4  = [&] () { ++counter;};

  // scenario 1
  auto A = taskflow.emplace(func1);
  auto B = taskflow.emplace(func2);
  auto C = taskflow.emplace(func3);
  auto D = taskflow.emplace(func4);
  A.precede(B);
  B.precede(C, D);
  executor.run(taskflow).wait();
  REQUIRE(counter == 2);

  // scenario 2
  counter = 0;
  A.work(func1);
  B.work(func2);
  C.work(func4);
  D.work(func3);
  executor.run(taskflow).wait();
  REQUIRE(counter == 3);

  // scenario 3
  taskflow.clear();
  std::tie(A, B, C, D) = taskflow.emplace(
    std::move(func1), std::move(func2), std::move(func3), std::move(func4)
  );
  A.precede(B);
  B.precede(C, D);
  counter = 0;
  executor.run(taskflow).wait();
  REQUIRE(counter == 2);
}

// --------------------------------------------------------
// Testcase: Iterators
// --------------------------------------------------------
TEST_CASE("Iterators" * doctest::timeout(300)) {

  SUBCASE("Order") {
    tf::Taskflow taskflow;

    auto A = taskflow.emplace([](){}).name("A");
    auto B = taskflow.emplace([](){}).name("B");
    auto C = taskflow.emplace([](){}).name("C");
    auto D = taskflow.emplace([](){}).name("D");
    auto E = taskflow.emplace([](){}).name("E");

    A.precede(B, C, D, E);
    E.succeed(B, C, D);

    A.for_each_successor([&, i=0] (tf::Task s) mutable {
      switch(i++) {
        case 0:
          REQUIRE(s == B);
        break;
        case 1:
          REQUIRE(s == C);
        break;
        case 2:
          REQUIRE(s == D);
        break;
        case 3:
          REQUIRE(s == E);
        break;
        default:
        break;
      }
    });

    E.for_each_dependent([&, i=0](tf::Task s) mutable {
      switch(i++) {
        case 0:
          REQUIRE(s == A);
        break;
        case 1:
          REQUIRE(s == B);
        break;
        case 2:
          REQUIRE(s == C);
        break;
        case 3:
          REQUIRE(s == D);
        break;
      }
    });
  }

  SUBCASE("Generic") {
    tf::Taskflow taskflow;

    auto A = taskflow.emplace([](){}).name("A");
    auto B = taskflow.emplace([](){}).name("B");
    auto C = taskflow.emplace([](){}).name("C");
    auto D = taskflow.emplace([](){}).name("D");
    auto E = taskflow.emplace([](){}).name("E");

    std::vector<tf::Task> tasks;

    taskflow.for_each_task([&tasks](tf::Task s){
      tasks.push_back(s);
    });

    REQUIRE(std::find(tasks.begin(), tasks.end(), A) != tasks.end());

    A.precede(B);

    A.for_each_successor([B](tf::Task s){ REQUIRE(s==B); });
    B.for_each_dependent([A](tf::Task s){ REQUIRE(s==A); });

    A.precede(C);
    A.precede(D);
    A.precede(E);
    C.precede(B);
    D.precede(B);
    E.precede(B);

    int counter{0}, a{0}, b{0}, c{0}, d{0}, e{0};
    A.for_each_successor([&](tf::Task s) {
      counter++;
      if(s == A) ++a;
      if(s == B) ++b;
      if(s == C) ++c;
      if(s == D) ++d;
      if(s == E) ++e;
    });
    REQUIRE(counter == A.num_successors());
    REQUIRE(a==0);
    REQUIRE(b==1);
    REQUIRE(c==1);
    REQUIRE(d==1);
    REQUIRE(e==1);

    counter = a = b = c = d = e = 0;
    B.for_each_dependent([&](tf::Task s) {
      counter++;
      if(s == A) ++a;
      if(s == B) ++b;
      if(s == C) ++c;
      if(s == D) ++d;
      if(s == E) ++e;
    });

    REQUIRE(counter == B.num_dependents());
    REQUIRE(a == 1);
    REQUIRE(b == 0);
    REQUIRE(c == 1);
    REQUIRE(d == 1);
    REQUIRE(e == 1);

    A.for_each_successor([](tf::Task s){
      s.name("A");
    });

    REQUIRE(A.name() == "A");
    REQUIRE(B.name() == "A");
    REQUIRE(C.name() == "A");
    REQUIRE(D.name() == "A");
    REQUIRE(E.name() == "A");

    B.for_each_dependent([](tf::Task s){
      s.name("B");
    });

    REQUIRE(A.name() == "B");
    REQUIRE(B.name() == "A");
    REQUIRE(C.name() == "B");
    REQUIRE(D.name() == "B");
    REQUIRE(E.name() == "B");

  }
}

// --------------------------------------------------------
// Testcase: Hash
// --------------------------------------------------------
TEST_CASE("Hash" * doctest::timeout(300)) {

  std::hash<tf::Task> hash;

  // empty hash
  tf::Task t1, t2;

  REQUIRE(hash(t1) == hash(t2));

  tf::Taskflow taskflow;

  t1 = taskflow.emplace([](){});

  REQUIRE(((hash(t1) != hash(t2)) || (hash(t1) == hash(t2) && t1 != t2)));

  t2 = taskflow.emplace([](){});

  REQUIRE(((hash(t1) != hash(t2)) || (hash(t1) == hash(t2) && t1 != t2)));

  t2 = t1;

  REQUIRE(hash(t1) == hash(t2));
}

// --------------------------------------------------------
// Testcase: SequentialRuns
// --------------------------------------------------------
void sequential_runs(unsigned W) {

  using namespace std::chrono_literals;

  size_t num_tasks = 100;

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;

  for(size_t i=0;i<num_tasks;i++){
    silent_tasks.emplace_back(
      taskflow.emplace([&counter]() {counter += 1;})
    );
  }

  SUBCASE("RunOnce"){
    auto fu = executor.run(taskflow);
    REQUIRE(taskflow.num_tasks() == num_tasks);
    fu.get();
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("WaitForAll") {
    executor.run(taskflow);
    executor.wait_for_all();
    REQUIRE(counter == num_tasks);
  }

  SUBCASE("RunWithFuture") {

    std::atomic<size_t> count {0};
    tf::Taskflow f;
    auto A = f.emplace([&](){ count ++; });
    auto B = f.emplace([&](tf::Subflow& subflow){
      count ++;
      auto B1 = subflow.emplace([&](){ count++; });
      auto B2 = subflow.emplace([&](){ count++; });
      auto B3 = subflow.emplace([&](){ count++; });
      B1.precede(B3); B2.precede(B3);
    });
    auto C = f.emplace([&](){ count ++; });
    auto D = f.emplace([&](){ count ++; });

    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    std::list<tf::Future<void>> fu_list;
    for(size_t i=0; i<500; i++) {
      if(i == 499) {
        executor.run(f).get();   // Synchronize the first 500 runs
        executor.run_n(f, 500);  // Run 500 times more
      }
      else if(i % 2) {
        fu_list.push_back(executor.run(f));
      }
      else {
        fu_list.push_back(executor.run(f, [&, i=i](){
          REQUIRE(count == (i+1)*7); })
        );
      }
    }

    executor.wait_for_all();

    for(auto& fu: fu_list) {
      REQUIRE(fu.valid());
      REQUIRE(fu.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    }

    REQUIRE(count == 7000);

  }

  SUBCASE("RunWithChange") {
    std::atomic<size_t> count {0};
    tf::Taskflow f;
    auto A = f.emplace([&](){ count ++; });
    auto B = f.emplace([&](tf::Subflow& subflow){
      count ++;
      auto B1 = subflow.emplace([&](){ count++; });
      auto B2 = subflow.emplace([&](){ count++; });
      auto B3 = subflow.emplace([&](){ count++; });
      B1.precede(B3); B2.precede(B3);
    });
    auto C = f.emplace([&](){ count ++; });
    auto D = f.emplace([&](){ count ++; });

    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    executor.run_n(f, 10).get();
    REQUIRE(count == 70);

    auto E = f.emplace([](){});
    D.precede(E);
    executor.run_n(f, 10).get();
    REQUIRE(count == 140);

    auto F = f.emplace([](){});
    E.precede(F);
    executor.run_n(f, 10);
    executor.wait_for_all();
    REQUIRE(count == 210);

  }

  SUBCASE("RunWithPred") {
    std::atomic<size_t> count {0};
    tf::Taskflow f;
    auto A = f.emplace([&](){ count ++; });
    auto B = f.emplace([&](tf::Subflow& subflow){
      count ++;
      auto B1 = subflow.emplace([&](){ count++; });
      auto B2 = subflow.emplace([&](){ count++; });
      auto B3 = subflow.emplace([&](){ count++; });
      B1.precede(B3); B2.precede(B3);
    });
    auto C = f.emplace([&](){ count ++; });
    auto D = f.emplace([&](){ count ++; });

    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    executor.run_until(f, [run=10]() mutable { return run-- == 0; },
      [&](){
        REQUIRE(count == 70);
        count = 0;
      }
    ).get();


    executor.run_until(f, [run=10]() mutable { return run-- == 0; },
      [&](){
        REQUIRE(count == 70);
        count = 0;
    });

    executor.run_until(f, [run=10]() mutable { return run-- == 0; },
      [&](){
        REQUIRE(count == 70);
        count = 0;
      }
    ).get();

  }

  SUBCASE("MultipleRuns") {
    std::atomic<size_t> count(0);

    tf::Taskflow tf1, tf2, tf3, tf4;

    for(size_t n=0; n<16; ++n) {
      tf1.emplace([&](){count.fetch_add(1, std::memory_order_relaxed);});
    }

    for(size_t n=0; n<1024; ++n) {
      tf2.emplace([&](){count.fetch_add(1, std::memory_order_relaxed);});
    }

    for(size_t n=0; n<32; ++n) {
      tf3.emplace([&](){count.fetch_add(1, std::memory_order_relaxed);});
    }

    for(size_t n=0; n<128; ++n) {
      tf4.emplace([&](){count.fetch_add(1, std::memory_order_relaxed);});
    }

    for(int i=0; i<200; ++i) {
      executor.run(tf1);
      executor.run(tf2);
      executor.run(tf3);
      executor.run(tf4);
    }
    executor.wait_for_all();
    REQUIRE(count == 240000);

  }
}


TEST_CASE("SerialRuns.1thread" * doctest::timeout(300)) {
  sequential_runs(1);
}

TEST_CASE("SerialRuns.2threads" * doctest::timeout(300)) {
  sequential_runs(2);
}

TEST_CASE("SerialRuns.3threads" * doctest::timeout(300)) {
  sequential_runs(3);
}

TEST_CASE("SerialRuns.4threads" * doctest::timeout(300)) {
  sequential_runs(4);
}

TEST_CASE("SerialRuns.5threads" * doctest::timeout(300)) {
  sequential_runs(5);
}

TEST_CASE("SerialRuns.6threads" * doctest::timeout(300)) {
  sequential_runs(6);
}

TEST_CASE("SerialRuns.7threads" * doctest::timeout(300)) {
  sequential_runs(7);
}

TEST_CASE("SerialRuns.8threads" * doctest::timeout(300)) {
  sequential_runs(8);
}

// --------------------------------------------------------
// Testcase:: RunAndWait
// --------------------------------------------------------

TEST_CASE("RunAndWait.Simple") {
  
  // create an executor and a taskflow
  tf::Executor executor(2);
  tf::Taskflow taskflow("Demo");

  REQUIRE_THROWS(executor.corun(taskflow));

  int counter{0};
  
  // taskflow to run by the main taskflow
  tf::Taskflow others;
  tf::Task A = others.emplace([&](){ counter++; });
  tf::Task B = others.emplace([&](){ counter++; });
  A.precede(B);

  // main taskflow
  tf::Task C = taskflow.emplace([&](){
    executor.corun(others);
    REQUIRE(counter == 2);
  });
  tf::Task D = taskflow.emplace([&](){
    executor.corun(others);
    REQUIRE(counter == 4);
  });
  C.precede(D);

  executor.run(taskflow).wait();

  // run others again
  executor.run(others).wait();

  REQUIRE(counter == 6);
}

TEST_CASE("RunAndWait.Complex") {

  const size_t N = 100;
  const size_t T = 1000;
  
  // create an executor and a taskflow
  tf::Executor executor(2);
  tf::Taskflow taskflow;

  std::array<tf::Taskflow, N> taskflows;

  std::atomic<size_t> counter{0};
  
  for(size_t n=0; n<N; n++) {
    for(size_t i=0; i<T; i++) {
      taskflows[n].emplace([&](){ counter++; });
    }
    taskflow.emplace([&executor, &tf=taskflows[n]](){
      executor.corun(tf);
      //executor.run(tf).wait();
    });
  }

  executor.run(taskflow).wait();

  REQUIRE(counter == T*N);
}

// --------------------------------------------------------
// Testcase: WorkerID
// --------------------------------------------------------
void worker_id(unsigned w) {

  tf::Taskflow taskflow;
  tf::Executor executor(w);

  for(int i=0; i<1000; i++) {
    auto A = taskflow.emplace([&](){
      auto id = executor.this_worker_id();
      REQUIRE(id>=0);
      REQUIRE(id< w);
    });

    auto B = taskflow.emplace([&](tf::Subflow& sf){
      auto id = executor.this_worker_id();
      REQUIRE(id>=0);
      REQUIRE(id< w);
      sf.emplace([&](){
        auto sfid = executor.this_worker_id();
        REQUIRE(sfid>=0);
        REQUIRE(sfid< w);
      });
      sf.emplace([&](tf::Subflow&){
        auto sfid = executor.this_worker_id();
        REQUIRE(sfid>=0);
        REQUIRE(sfid< w);
      });
    });

    A.precede(B);
  }

  executor.run_n(taskflow, 100).wait();
}

TEST_CASE("WorkerID.1thread") {
  worker_id(1);
}

TEST_CASE("WorkerID.2threads") {
  worker_id(2);
}

TEST_CASE("WorkerID.3threads") {
  worker_id(3);
}

TEST_CASE("WorkerID.4threads") {
  worker_id(4);
}

TEST_CASE("WorkerID.5threads") {
  worker_id(5);
}

TEST_CASE("WorkerID.6threads") {
  worker_id(6);
}

TEST_CASE("WorkerID.7threads") {
  worker_id(7);
}

TEST_CASE("WorkerID.8threads") {
  worker_id(8);
}

// --------------------------------------------------------
// Testcase: ParallelRuns
// --------------------------------------------------------
void parallel_runs(unsigned w) {

  std::atomic<int> counter;
  std::vector<std::thread> threads;

  auto make_taskflow = [&] (tf::Taskflow& tf) {
    for(int i=0; i<1024; i++) {
      auto A = tf.emplace([&] () {
        counter.fetch_add(1, std::memory_order_relaxed);
      });
      auto B = tf.emplace([&] () {
        counter.fetch_add(1, std::memory_order_relaxed);
      });
      A.precede(B);
    }
  };

  SUBCASE("RunAndWait") {
    tf::Executor executor(w);
    counter = 0;
    for(int t=0; t<32; t++) {
      threads.emplace_back([&] () {
        tf::Taskflow taskflow;
        make_taskflow(taskflow);
        executor.run(taskflow).wait();
      });
    }

    for(auto& t : threads) {
      t.join();
    }
    threads.clear();

    REQUIRE(counter.load() == 32*1024*2);

  }

  SUBCASE("RunAndWaitForAll") {
    tf::Executor executor(w);
    counter = 0;
    std::vector<std::unique_ptr<tf::Taskflow>> taskflows(32);
    std::atomic<size_t> barrier(0);
    for(int t=0; t<32; t++) {
      threads.emplace_back([&, t=t] () {
        taskflows[t] = std::make_unique<tf::Taskflow>();
        make_taskflow(*taskflows[t]);
        executor.run(*taskflows[t]);
        ++barrier;    // make sure all runs are issued
      });
    }

    while(barrier != 32);
    executor.wait_for_all();
    REQUIRE(counter.load() == 32*1024*2);

    for(auto& t : threads) {
      t.join();
    }
    threads.clear();
  }
}


TEST_CASE("ParallelRuns.1thread" * doctest::timeout(300)) {
  parallel_runs(1);
}

TEST_CASE("ParallelRuns.2threads" * doctest::timeout(300)) {
  parallel_runs(2);
}

TEST_CASE("ParallelRuns.3threads" * doctest::timeout(300)) {
  parallel_runs(3);
}

TEST_CASE("ParallelRuns.4threads" * doctest::timeout(300)) {
  parallel_runs(4);
}

TEST_CASE("ParallelRuns.5threads" * doctest::timeout(300)) {
  parallel_runs(5);
}

TEST_CASE("ParallelRuns.6threads" * doctest::timeout(300)) {
  parallel_runs(6);
}

TEST_CASE("ParallelRuns.7threads" * doctest::timeout(300)) {
  parallel_runs(7);
}

TEST_CASE("ParallelRuns.8threads" * doctest::timeout(300)) {
  parallel_runs(8);
}

// --------------------------------------------------------
// Testcase: NestedRuns
// --------------------------------------------------------
void nested_runs(unsigned w) {

  int counter {0};

  struct A {

    tf::Executor executor;
    tf::Taskflow taskflow;

    int& counter;

    A(unsigned w, int& c) : executor{w}, counter{c} { }

    void run()
    {
      taskflow.clear();
      auto A1 = taskflow.emplace([&]() { counter++; });
      auto A2 = taskflow.emplace([&]() { counter++; });
      A1.precede(A2);
      executor.run_n(taskflow, 10).wait();
    }

  };

  struct B {

    tf::Taskflow taskflow;
    tf::Executor executor;

    int& counter;

    A a_sim;

    B(unsigned w, int& c) : executor{w}, counter{c}, a_sim{w, c} { }

    void run()
    {
      taskflow.clear();
      auto B1 = taskflow.emplace([&] () { ++counter; });
      auto B2 = taskflow.emplace([&] () { ++counter; a_sim.run(); });
      B1.precede(B2);
      executor.run_n(taskflow, 100).wait();
    }
  };

  struct C {

    tf::Taskflow taskflow;
    tf::Executor executor;

    int& counter;

    B b_sim;

    C(unsigned w, int& c) : executor{w}, counter{c}, b_sim{w, c} { }

    void run()
    {
      taskflow.clear();
      auto C1 = taskflow.emplace([&] () { ++counter; });
      auto C2 = taskflow.emplace([&] () { ++counter; b_sim.run(); });
      C1.precede(C2);
      executor.run_n(taskflow, 100).wait();
    }
  };

  C c(w, counter);
  c.run();

  REQUIRE(counter == 220200);
}

TEST_CASE("NestedRuns.1thread") {
  nested_runs(1);
}

TEST_CASE("NestedRuns.2threads") {
  nested_runs(2);
}

TEST_CASE("NestedRuns.3threads") {
  nested_runs(3);
}

TEST_CASE("NestedRuns.4threads") {
  nested_runs(4);
}

TEST_CASE("NestedRuns.8threads") {
  nested_runs(8);
}

TEST_CASE("NestedRuns.16threads") {
  nested_runs(16);
}

// --------------------------------------------------------
// Testcase: Observer
// --------------------------------------------------------

void observer(unsigned w) {

  tf::Executor executor(w);

  auto observer = executor.make_observer<tf::ChromeObserver>();

  tf::Taskflow taskflowA;
  std::vector<tf::Task> tasks;
  // Static tasking
  for(auto i=0; i < 64; i ++) {
    tasks.emplace_back(taskflowA.emplace([](){}));
  }

  // Randomly specify dependency
  for(auto i=0; i < 64; i ++) {
    for(auto j=i+1; j < 64; j++) {
      if(rand()%2 == 0) {
        tasks[i].precede(tasks[j]);
      }
    }
  }

  executor.run_n(taskflowA, 16).get();

  REQUIRE(observer->num_tasks() == 64*16);

  observer->clear();
  REQUIRE(observer->num_tasks() == 0);
  tasks.clear();

}

TEST_CASE("Observer.1thread" * doctest::timeout(300)) {
  observer(1);
}

TEST_CASE("Observer.2threads" * doctest::timeout(300)) {
  observer(2);
}

TEST_CASE("Observer.3threads" * doctest::timeout(300)) {
  observer(3);
}

TEST_CASE("Observer.4threads" * doctest::timeout(300)) {
  observer(4);
}


