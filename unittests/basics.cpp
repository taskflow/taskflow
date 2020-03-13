#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: Builder
// --------------------------------------------------------
TEST_CASE("Builder" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::atomic<int> counter {0};
  std::vector<tf::Task> silent_tasks;
  std::vector<tf::Task> tasks;

  SUBCASE("EmptyFlow") {
    for(unsigned W=1; W<32; ++W) {
      tf::Executor executor(W);
      tf::Taskflow taskflow;
      REQUIRE(taskflow.num_tasks() == 0);
      REQUIRE(taskflow.empty() == true);
      executor.run(taskflow).wait();
    }
  }
    
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
      if(i>0){
        taskflow.precede(tasks[i-1], tasks[i]);
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
    }
    taskflow.broadcast(src, silent_tasks);
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
    }
    taskflow.gather(silent_tasks, dst);
    executor.run(taskflow).get();
    REQUIRE(counter == num_tasks - 1);
    REQUIRE(taskflow.num_tasks() == num_tasks);
  }

  SUBCASE("MapReduce"){
    auto src = taskflow.emplace([&counter]() {counter = 0;});
    for(size_t i=0;i<num_tasks;i++){
      silent_tasks.emplace_back(
        taskflow.emplace([&counter]() {counter += 1;})
      );
    }
    taskflow.broadcast(src, silent_tasks);
    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    taskflow.gather(silent_tasks, dst);
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
    }
    taskflow.broadcast(src, silent_tasks);
    taskflow.linearize(silent_tasks);
    auto dst = taskflow.emplace(
      [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
    );
    taskflow.gather(silent_tasks, dst);
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
      tf.parallel_for(dummy.begin(), dummy.end(), [] (int) {});
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
// Testcase: STDFunction
// --------------------------------------------------------
TEST_CASE("STDFunction" * doctest::timeout(300)) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  int counter = 0;

  std::function<void()> func1  = [&] () { ++counter; };
  std::function<int()> func2 = [&] () { ++counter; return 0; };
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

    std::list<std::future<void>> fu_list;
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
        auto E = f.emplace([&](){ count ++; });
        auto F = f.emplace([&](){ count ++; });
        A.precede(E).precede(F);
    });

    executor.run_until(f, [run=10]() mutable { return run-- == 0; }, 
      [&](){
        REQUIRE(count == 90);
        count = 0;
      }
    ).get();
    
  }

  SUBCASE("MultipleRuns") {
    std::atomic<size_t> counter(0);

    tf::Taskflow tf1, tf2, tf3, tf4;

    for(size_t n=0; n<16; ++n) {
      tf1.emplace([&](){counter.fetch_add(1, std::memory_order_relaxed);});
    }
    
    for(size_t n=0; n<1024; ++n) {
      tf2.emplace([&](){counter.fetch_add(1, std::memory_order_relaxed);});
    }
    
    for(size_t n=0; n<32; ++n) {
      tf3.emplace([&](){counter.fetch_add(1, std::memory_order_relaxed);});
    }
    
    for(size_t n=0; n<128; ++n) {
      tf4.emplace([&](){counter.fetch_add(1, std::memory_order_relaxed);});
    }
    
    for(int i=0; i<200; ++i) {
      executor.run(tf1);
      executor.run(tf2);
      executor.run(tf3);
      executor.run(tf4);
    }
    executor.wait_for_all();
    REQUIRE(counter == 240000);
    
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
// Testcase: ParallelFor
// --------------------------------------------------------

void parallel_for(unsigned W) {
  
  using namespace std::chrono_literals;

  const auto mapper = [](unsigned w, size_t num_data, bool group){
    tf::Executor executor(w);
    tf::Taskflow tf;
    std::vector<int> vec(num_data, 0);
    tf.parallel_for(
      vec.begin(), vec.end(), [] (int& v) { v = 64; }, group ? ::rand() : 0
    );
    for(const auto v : vec) {
      REQUIRE(v == 0);
    }
    executor.run(tf);
    executor.wait_for_all();
    for(const auto v : vec) {
      REQUIRE(v == 64);
    }
  };

  const auto reducer = [](unsigned w, size_t num_data, bool group){
    tf::Executor executor(w);
    tf::Taskflow tf;
    std::vector<int> vec(num_data, 0);
    std::atomic<int> sum(0);
    tf.parallel_for(vec.begin(), vec.end(), [&](auto) { ++sum; }, group ? ::rand() : 0);
    REQUIRE(sum == 0);
    executor.run(tf);
    executor.wait_for_all();
    REQUIRE(sum == vec.size());
  };

  // map
  SUBCASE("Map") {
    for(size_t num_data=1; num_data<=59049; num_data *= 3){
      mapper(W, num_data, true);
      mapper(W, num_data, false);
    }
  }

  // reduce
  SUBCASE("Reduce") {
    for(size_t num_data=1; num_data<=59049; num_data *= 3){
      reducer(W, num_data, true);
      reducer(W, num_data, false);
    }
  }
}

TEST_CASE("ParallelFor.1thread" * doctest::timeout(300)) {
  parallel_for(1);
}

TEST_CASE("ParallelFor.2threads" * doctest::timeout(300)) {
  parallel_for(2);
}

TEST_CASE("ParallelFor.3threads" * doctest::timeout(300)) {
  parallel_for(3);
}

TEST_CASE("ParallelFor.4threads" * doctest::timeout(300)) {
  parallel_for(4);
}

TEST_CASE("ParallelFor.5threads" * doctest::timeout(300)) {
  parallel_for(5);
}

TEST_CASE("ParallelFor.6threads" * doctest::timeout(300)) {
  parallel_for(6);
}

TEST_CASE("ParallelFor.7threads" * doctest::timeout(300)) {
  parallel_for(7);
}

TEST_CASE("ParallelFor.8threads" * doctest::timeout(300)) {
  parallel_for(8);
}

// --------------------------------------------------------
// Testcase: ParallelForOnIndex
// --------------------------------------------------------
void parallel_for_index(unsigned w) {
  
  using namespace std::chrono_literals;

  auto exception_test = [] () {

    tf::Taskflow tf;

    // invalid index
    REQUIRE_THROWS(tf.parallel_for(0, 10, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0, 10, -1, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10, 0, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10, 0, 1, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0u, 10u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10u, 0u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10u, 0u, 1u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0f, 10.0f, 0.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0f, 10.0f, -1.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0f, 0.0f, 0.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0f, 0.0f, 1.0f, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 10.0, 0.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 10.0, -1.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0, 0.0, 0.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(10.0, 0.0, 1.0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0, 0, 0, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0u, 0u, 0u, [] (auto) {}));
    REQUIRE_THROWS(tf.parallel_for(0.0, 0.0, 0.0, [] (auto) {}));
    
    // graceful case
    REQUIRE_NOTHROW(tf.parallel_for(0, 0, -1, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0, 0, 1, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0u, 0u, 1u, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0f, 0.0f, -1.0f, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0f, 0.0f, 1.0f, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0, 0.0, -1.0, [] (auto) {}));
    REQUIRE_NOTHROW(tf.parallel_for(0.0, 0.0, 1.0, [] (auto) {}));
  };

  auto positive_integer_step = [] (unsigned w) {
    tf::Executor executor(w);
    for(int beg=-10; beg<=10; ++beg) {
      for(int end=beg; end<=10; ++end) {
        for(int s=1; s<=end-beg; ++s) {
          int n = 0;
          for(int b = beg; b<end; b+=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_integer_step = [] (unsigned w) {
    tf::Executor executor(w);
    for(int beg=10; beg>=-10; --beg) {
      for(int end=beg; end>=-10; --end) {
        for(int s=1; s<=beg-end; ++s) {
          int n = 0;
          for(int b = beg; b>end; b-=s) {
            ++n;
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto positive_floating_step = [] (unsigned w) {
    tf::Executor executor(w);
    for(float beg=-10.0f; beg<=10.0f; ++beg) {
      for(float end=beg; end<=10.0f; ++end) {
        for(float s=1.0f; s<=end-beg; s+=0.1f) {
          int n = 0;
          if(beg < end) {
            for(float b = beg; b < end; b += s) {
              ++n;
            }
          }
          else if(beg > end) {
            for(float b = beg; b > end; b += s) {
              ++n;
            }
          }
          
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  auto negative_floating_step = [] (unsigned w) {
    tf::Executor executor(w);
    for(float beg=10.0f; beg>=-10.0f; --beg) {
      for(float end=beg; end>=-10.0f; --end) {
        for(float s=1.0f; s<=beg-end; s+=0.1f) {
          int n = 0;
          if(beg < end) {
            for(float b = beg; b < end; b += (-s)) {
              ++n;
            }
          }
          else if(beg > end) {
            for(float b = beg; b > end; b += (-s)) {
              ++n;
            }
          }
          tf::Taskflow tf;
          std::atomic<int> counter {0};
          tf.parallel_for(beg, end, -s, [&] (auto) {
            counter.fetch_add(1, std::memory_order_relaxed);
          });
          executor.run(tf);
          executor.wait_for_all();
          REQUIRE(n == counter);
        }
      }
    }
  };
  
  SUBCASE("Exception") {
    exception_test();
  }

  SUBCASE("PositiveIntegerStep") {
    positive_integer_step(w);  
  }
  
  SUBCASE("NegativeIntegerStep") {
    negative_integer_step(w);  
  }
  
  SUBCASE("PositiveFloatingStep") {
    positive_floating_step(w);  
  }
  
  SUBCASE("NegativeFloatingStep") {
    negative_floating_step(w);  
  }
}

TEST_CASE("ParallelForIndex.1thread" * doctest::timeout(300)) {
  parallel_for_index(1);
}

TEST_CASE("ParallelForIndex.2threads" * doctest::timeout(300)) {
  parallel_for_index(2);
}

TEST_CASE("ParallelForIndex.3threads" * doctest::timeout(300)) {
  parallel_for_index(3);
}

TEST_CASE("ParallelForIndex.4threads" * doctest::timeout(300)) {
  parallel_for_index(4);
}

TEST_CASE("ParallelForIndex.5threads" * doctest::timeout(300)) {
  parallel_for_index(5);
}

TEST_CASE("ParallelForIndex.6threads" * doctest::timeout(300)) {
  parallel_for_index(6);
}

TEST_CASE("ParallelForIndex.7threads" * doctest::timeout(300)) {
  parallel_for_index(7);
}

TEST_CASE("ParallelForIndex.8threads" * doctest::timeout(300)) {
  parallel_for_index(8);
}

// --------------------------------------------------------
// Testcase: Reduce
// --------------------------------------------------------
TEST_CASE("Reduce" * doctest::timeout(300)) {

  const auto plus_test = [](unsigned num_workers, auto &&data){
    tf::Executor executor(num_workers);
    tf::Taskflow tf;
    int result {0};
    std::iota(data.begin(), data.end(), 1);
    tf.reduce(data.begin(), data.end(), result, std::plus<int>());
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, std::plus<int>()));
  };

  const auto multiply_test = [](unsigned num_workers, auto &&data){
    tf::Executor executor(num_workers);
    tf::Taskflow tf;
    std::fill(data.begin(), data.end(), 1.0);
    double result {2.0};
    tf.reduce(data.begin(), data.end(), result, std::multiplies<double>());
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 2.0, std::multiplies<double>()));
  };

  const auto max_test = [](unsigned num_workers, auto &&data){
    tf::Executor executor(num_workers);
    tf::Taskflow tf;
    std::iota(data.begin(), data.end(), 1);
    int result {0};
    auto lambda = [](const auto& l, const auto& r){return std::max(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(data.begin(), data.end(), 0, lambda));
  };

  const auto min_test = [](unsigned num_workers, auto &&data){
    tf::Executor executor(num_workers);
    tf::Taskflow tf;
    std::iota(data.begin(), data.end(), 1);
    int result {std::numeric_limits<int>::max()};
    auto lambda = [](const auto& l, const auto& r){return std::min(l, r);};
    tf.reduce(data.begin(), data.end(), result, lambda);
    executor.run(tf).get();
    REQUIRE(result == std::accumulate(
      data.begin(), data.end(), std::numeric_limits<int>::max(), lambda)
    );
  };

  for(unsigned w=1; w<=4; ++w){
    for(size_t j=0; j<=256; j=j*2+1){
      plus_test(w, std::vector<int>(j));
      plus_test(w, std::list<int>(j));

      multiply_test(w, std::vector<double>(j));
      multiply_test(w, std::list<double>(j));

      max_test(w, std::vector<int>(j));
      max_test(w, std::list<int>(j));

      min_test(w, std::vector<int>(j));
      min_test(w, std::list<int>(j));
    }
  }
}

// --------------------------------------------------------
// Testcase: ReduceMin
// --------------------------------------------------------
TEST_CASE("ReduceMin" * doctest::timeout(300)) {

  for(int w=1; w<=4; w++) {
    tf::Executor executor(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      tf::Taskflow tf;
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::max();
      int test = std::numeric_limits<int>::max();
      for(auto& d : data) {
        d = ::rand();
        gold = std::min(gold, d);
      }
      tf.reduce_min(data.begin(), data.end(), test);
      executor.run(tf).get();
      REQUIRE(test == gold);
    }
  }

}

// --------------------------------------------------------
// Testcase: ReduceMax
// --------------------------------------------------------
TEST_CASE("ReduceMax" * doctest::timeout(300)) {

  for(int w=1; w<=4; w++) {
    tf::Executor executor(w);
    for(int i=0; i<=65536; i = (i <= 1024) ? i + 1 : i*2 + 1) {
      tf::Taskflow tf;
      std::vector<int> data(i);
      int gold = std::numeric_limits<int>::min();
      int test = std::numeric_limits<int>::min();
      for(auto& d : data) {
        d = ::rand();
        gold = std::max(gold, d);
      }
      tf.reduce_max(data.begin(), data.end(), test);
      executor.run(tf).get();
      REQUIRE(test == gold);
    }
  }
}

// --------------------------------------------------------
// Testcase: JoinedSubflow
// -------------------------------------------------------- 

void joined_subflow(unsigned W) {

  using namespace std::literals::chrono_literals;
  
  SUBCASE("Trivial") {
    tf::Executor executor(W);
    tf::Taskflow tf;
    
    // empty flow with future
    tf::Task subflow3, subflow3_;
    //std::future<int> fu3, fu3_;
    std::atomic<int> fu3v{0}, fu3v_{0};
    
    // empty flow
    auto subflow1 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.join();
    }).name("subflow1");
    
    // nested empty flow
    auto subflow2 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.emplace([&] (tf::Subflow& fb) {
        fu3v++;
        fb.emplace( [&] (tf::Subflow& fb) {
          fu3v++;
          fb.join();
        }).name("subflow2_1_1");
      }).name("subflow2_1");
    }).name("subflow2");
    
    subflow3 = tf.emplace([&] (tf::Subflow& fb) {

      REQUIRE(fu3v == 4);

      fu3v++;
      fu3v_++;
      
      subflow3_ = fb.emplace([&] (tf::Subflow& fb) {
        REQUIRE(fu3v_ == 3);
        fu3v++;
        fu3v_++;
        //return 200;
        fb.join();
      });
      subflow3_.name("subflow3_");

      // hereafter we use 100us to avoid dangling reference ...
      auto s1 = fb.emplace([&] () { 
        fu3v_++;
        fu3v++;
      }).name("s1");
      
      auto s2 = fb.emplace([&] () {
        fu3v_++;
        fu3v++;
      }).name("s2");
      
      auto s3 = fb.emplace([&] () {
        fu3v++;
        REQUIRE(fu3v_ == 4);
      }).name("s3");

      s1.precede(subflow3_);
      s2.precede(subflow3_);
      subflow3_.precede(s3);

      REQUIRE(fu3v_ == 1);

      //return 100;
    });
    subflow3.name("subflow3");

    // empty flow to test future
    auto subflow4 = tf.emplace([&] () {
      fu3v++;
    }).name("subflow4");

    subflow1.precede(subflow2);
    subflow2.precede(subflow3);
    subflow3.precede(subflow4);

    executor.run(tf).get();
    // End of for loop
  }
  
  // Mixed intra- and inter- operations
  SUBCASE("Complex") {
    tf::Executor executor(W);
    tf::Taskflow tf;

    std::vector<int> data;
    int sum {0};

    auto A = tf.emplace([&data] () {
      for(int i=0; i<10; ++i) {
        data.push_back(1);
      }
    });

    std::atomic<size_t> count {0};

    auto B = tf.emplace([&count, &data, &sum](tf::Subflow& fb){

      //auto [src, tgt] = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());
      auto task_pair = fb.reduce(data.begin(), data.end(), sum, std::plus<int>());
      auto &src = std::get<0>(task_pair);
      auto &tgt = std::get<1>(task_pair);

      fb.emplace([&sum] () { REQUIRE(sum == 0); }).precede(src);

      tgt.precede(fb.emplace([&sum] () { REQUIRE(sum == 10); }));

      for(size_t i=0; i<10; i ++){
        ++count;
      }

      auto n = fb.emplace([&count](tf::Subflow& fb){

        REQUIRE(count == 20);
        ++count;

        auto prev = fb.emplace([&count](){
          REQUIRE(count == 21);
          ++count;
        });

        for(size_t i=0; i<10; i++){
          auto next = fb.emplace([&count, i](){
            REQUIRE(count == 22+i);
            ++count;
          });
          prev.precede(next);
          prev = next;
        }
      });

      for(size_t i=0; i<10; i++){
        fb.emplace([&count](){ ++count; }).precede(n);
      }
    });

    A.precede(B);

    executor.run(tf).get();
    REQUIRE(count == 32);
    REQUIRE(sum == 10);
    
  }
}

TEST_CASE("JoinedSubflow.1thread" * doctest::timeout(300)){
  joined_subflow(1);
}

TEST_CASE("JoinedSubflow.2threads" * doctest::timeout(300)){
  joined_subflow(2);
}

TEST_CASE("JoinedSubflow.3threads" * doctest::timeout(300)){
  joined_subflow(3);
}

TEST_CASE("JoinedSubflow.4threads" * doctest::timeout(300)){
  joined_subflow(4);
}

TEST_CASE("JoinedSubflow.5threads" * doctest::timeout(300)){
  joined_subflow(5);
}

TEST_CASE("JoinedSubflow.6threads" * doctest::timeout(300)){
  joined_subflow(6);
}

TEST_CASE("JoinedSubflow.7threads" * doctest::timeout(300)){
  joined_subflow(7);
}

TEST_CASE("JoinedSubflow.8threads" * doctest::timeout(300)){
  joined_subflow(8);
}

// --------------------------------------------------------
// Testcase: DetachedSubflow
// --------------------------------------------------------

void detached_subflow(unsigned W) {
  
  using namespace std::literals::chrono_literals;

  SUBCASE("Trivial") {
    tf::Executor executor(W);
    tf::Taskflow tf;
    
    // empty flow with future
    tf::Task subflow3, subflow3_;
    std::atomic<int> fu3v{0}, fu3v_{0};
    
    // empty flow
    auto subflow1 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.detach();
    }).name("subflow1");
    
    // nested empty flow
    auto subflow2 = tf.emplace([&] (tf::Subflow& fb) {
      fu3v++;
      fb.emplace([&] (tf::Subflow& fb) {
        fu3v++;
        fb.emplace( [&] (tf::Subflow& fb) {
          fu3v++;
          fb.join();
        }).name("subflow2_1_1");
        fb.detach();
      }).name("subflow2_1");
      fb.detach();
    }).name("subflow2");
    
    subflow3 = tf.emplace([&] (tf::Subflow& fb) {

      REQUIRE((fu3v >= 2 && fu3v <= 4));

      fu3v++;
      fu3v_++;
      
      subflow3_ = fb.emplace([&] (tf::Subflow& fb) {
        REQUIRE(fu3v_ == 3);
        fu3v++;
        fu3v_++;
        fb.join();
      });
      subflow3_.name("subflow3_");

      // hereafter we use 100us to avoid dangling reference ...
      auto s1 = fb.emplace([&] () { 
        fu3v_++;
        fu3v++;
      }).name("s1");
      
      auto s2 = fb.emplace([&] () {
        fu3v_++;
        fu3v++;
      }).name("s2");
      
      auto s3 = fb.emplace([&] () {
        fu3v++;
        REQUIRE(fu3v_ == 4);
      }).name("s3");

      s1.precede(subflow3_);
      s2.precede(subflow3_);
      subflow3_.precede(s3);

      REQUIRE(fu3v_ == 1);

      fb.detach();

      //return 100;
    });
    subflow3.name("subflow3");

    // empty flow to test future
    auto subflow4 = tf.emplace([&] () {
      REQUIRE((fu3v >= 3 && fu3v <= 9));
      fu3v++;
    }).name("subflow4");

    subflow1.precede(subflow2);
    subflow2.precede(subflow3);
    subflow3.precede(subflow4);

    executor.run(tf).get();

    REQUIRE(fu3v  == 10);
    REQUIRE(fu3v_ == 4);
    
  }
}

TEST_CASE("DetachedSubflow.1thread" * doctest::timeout(300)) {
  detached_subflow(1);
}

TEST_CASE("DetachedSubflow.2threads" * doctest::timeout(300)) {
  detached_subflow(2);
}

TEST_CASE("DetachedSubflow.3threads" * doctest::timeout(300)) {
  detached_subflow(3);
}

TEST_CASE("DetachedSubflow.4threads" * doctest::timeout(300)) {
  detached_subflow(4);
}

TEST_CASE("DetachedSubflow.5threads" * doctest::timeout(300)) {
  detached_subflow(5);
}

TEST_CASE("DetachedSubflow.6threads" * doctest::timeout(300)) {
  detached_subflow(6);
}

TEST_CASE("DetachedSubflow.7threads" * doctest::timeout(300)) {
  detached_subflow(7);
}

TEST_CASE("DetachedSubflow.8threads" * doctest::timeout(300)) {
  detached_subflow(8);
}


// --------------------------------------------------------
// Testcase: TreeSubflow
// -------------------------------------------------------- 
void detach_spawn(const int max_depth, std::atomic<int>& counter, int depth, tf::Subflow& subflow)  {
  if(depth < max_depth) {
    counter.fetch_add(1, std::memory_order_relaxed);
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      detach_spawn(max_depth, counter, depth, subflow); }
    );
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      detach_spawn(max_depth, counter, depth, subflow); }
    );
    subflow.detach();
  }
}

void join_spawn(const int max_depth, std::atomic<int>& counter, int depth, tf::Subflow& subflow)  {
  if(depth < max_depth) {
    counter.fetch_add(1, std::memory_order_relaxed);
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      join_spawn(max_depth, counter, depth, subflow); }
    );
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      join_spawn(max_depth, counter, depth, subflow); }
    );
  }
}

void mix_spawn(
  const int max_depth, 
  std::atomic<int>& counter, 
  int depth, tf::Subflow& subflow
)  {
  if(depth < max_depth) {
    auto ret = counter.fetch_add(1, std::memory_order_relaxed);
    if(ret % 2) {
      subflow.detach();
    }
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      mix_spawn(max_depth, counter, depth, subflow); }
    );
    subflow.emplace([&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
      mix_spawn(max_depth, counter, depth, subflow); }
    );
  }
}

TEST_CASE("TreeSubflow" * doctest::timeout(300)) {

  SUBCASE("AllDetach") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){ 
        detach_spawn(max_depth, counter, 0, subflow); 
      });

      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }


  SUBCASE("AllJoin") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){ 
        join_spawn(max_depth, counter, 0, subflow); 
      });
      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }

  SUBCASE("Mix") {
    constexpr int max_depth {10};
    for(int W=1; W<=4; W++) {
      std::atomic<int> counter {0};
      tf::Taskflow tf;
      tf.emplace([&](tf::Subflow& subflow){ 
        mix_spawn(max_depth, counter, 0, subflow); 
      });

      tf::Executor executor(W);
      executor.run(tf).get();
      REQUIRE(counter == (1<<max_depth) - 1);
    }
  }
}


// --------------------------------------------------------
// Testcase: Composition
// --------------------------------------------------------
TEST_CASE("Composition-1" * doctest::timeout(300)) {

  for(unsigned w=1; w<=8; ++w) {

    tf::Executor executor(w);

    tf::Taskflow f0;

    int cnt {0};

    auto A = f0.emplace([&cnt](){ ++cnt; });
    auto B = f0.emplace([&cnt](){ ++cnt; });
    auto C = f0.emplace([&cnt](){ ++cnt; });
    auto D = f0.emplace([&cnt](){ ++cnt; });
    auto E = f0.emplace([&cnt](){ ++cnt; });

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    tf::Taskflow f1;
    
    // module 1
    std::tie(A, B, C, D, E) = f1.emplace(
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; },
      [&cnt] () { ++cnt; }
    );
    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);
    auto m1_1 = f1.composed_of(f0);
    E.precede(m1_1);
    
    executor.run(f1).get();
    REQUIRE(cnt == 10);

    cnt = 0;
    executor.run_n(f1, 100).get();
    REQUIRE(cnt == 10 * 100);

    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);
    
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f1, n).get();
      REQUIRE(cnt == 15*n);
    }

    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f1);
    }
    
    executor.wait_for_all();

    REQUIRE(cnt == 1500);
  }
}

// TESTCASE: composition-2
TEST_CASE("Composition-2" * doctest::timeout(300)) {

  for(unsigned w=1; w<=8; ++w) {

    tf::Executor executor(w);

    int cnt {0};
    
    // level 0 (+5)
    tf::Taskflow f0;

    auto A = f0.emplace([&cnt](){ ++cnt; });
    auto B = f0.emplace([&cnt](){ ++cnt; });
    auto C = f0.emplace([&cnt](){ ++cnt; });
    auto D = f0.emplace([&cnt](){ ++cnt; });
    auto E = f0.emplace([&cnt](){ ++cnt; });

    A.precede(B);
    B.precede(C);
    C.precede(D);
    D.precede(E);

    // level 1 (+10)
    tf::Taskflow f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+20)
    tf::Taskflow f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);
    
    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f2, n).get();
      REQUIRE(cnt == 20*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f2);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 100*20);
  }
}

// TESTCASE: composition-3
TEST_CASE("Composition-3" * doctest::timeout(300)) {
  
  for(unsigned w=1; w<=8; ++w) {
  
    tf::Executor executor(w);

    int cnt {0};
    
    // level 0 (+2)
    tf::Taskflow f0;

    auto A = f0.emplace([&cnt](){ ++cnt; });
    auto B = f0.emplace([&cnt](){ ++cnt; });

    A.precede(B);

    // level 1 (+4)
    tf::Taskflow f1;
    auto m1_1 = f1.composed_of(f0);
    auto m1_2 = f1.composed_of(f0);
    m1_1.precede(m1_2);

    // level 2 (+8)
    tf::Taskflow f2;
    auto m2_1 = f2.composed_of(f1);
    auto m2_2 = f2.composed_of(f1);
    m2_1.precede(m2_2);

    // level 3 (+16)
    tf::Taskflow f3;
    auto m3_1 = f3.composed_of(f2);
    auto m3_2 = f3.composed_of(f2);
    m3_1.precede(m3_2);

    // synchronous run
    for(int n=0; n<100; n++) {
      cnt = 0;
      executor.run_n(f3, n).get();
      REQUIRE(cnt == 16*n);
    }

    // asynchronous run
    cnt = 0;
    for(int n=0; n<100; n++) {
      executor.run(f3);
    }
    executor.wait_for_all();
    REQUIRE(cnt == 16*100);
  }
}

// --------------------------------------------------------
// Testcase: Observer 
// -------------------------------------------------------- 

void observer(unsigned w) {

  tf::Executor executor(w);

  auto observer = executor.make_observer<tf::ExecutorObserver>();    

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

  // Dynamic tasking  
  tf::Taskflow taskflowB;
  std::atomic<int> num_tasks {0};
  // Static tasking 
  for(auto i=0; i < 64; i ++) {
    tasks.emplace_back(taskflowB.emplace([&](tf::Subflow& subflow){
      num_tasks ++;
      auto num_spawn = rand() % 10 + 1;
      // Randomly spawn tasks
      for(auto i=0; i<num_spawn; i++) {
        subflow.emplace([&](){ num_tasks ++; });
      }    
      if(rand() % 2) {
        subflow.detach();
      }
      else {
        // In join mode, this task will be visited twice
        num_tasks ++;
      }
    }));
  }

  // Randomly specify dependency
  for(auto i=0; i < 64; i ++) {
    for(auto j=i+1; j < 64; j++) {
      if(rand()%2 == 0) {
        tasks[i].precede(tasks[j]);
      }
    }
  }

  executor.run_n(taskflowB, 16).get();
  REQUIRE(observer->num_tasks() == num_tasks);
  
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

// --------------------------------------------------------
// Testcase: Conditional Tasking
// -------------------------------------------------------- 
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
        [&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
          conditional_spawn(counter, max_depth, depth, subflow); 
      });
      auto C = subflow.emplace(
        [&, max_depth, depth=depth+1](tf::Subflow& subflow){ 
          conditional_spawn(counter, max_depth, depth, subflow); 
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

  int counter;
  double avg;

  auto A = taskflow.emplace( [&](){ counter = 0; } );
  auto B = taskflow.emplace( [&](){ ++counter; return ::rand()%2; } );
  auto C = taskflow.emplace( [&](){ return ::rand()%2; } );
  auto D = taskflow.emplace( [&](){ return ::rand()%2; } );
  auto E = taskflow.emplace( [&](){ return ::rand()%2; } );
  auto F = taskflow.emplace( [&](){ return ::rand()%2; } );
  auto G = taskflow.emplace( [&, N=0, accu=0.0]() mutable { 
      ++N;  // a new round
      accu += counter;
      avg = accu/N;
      //std::cout << N << ": " << counter << " => avg = " << avg << '\n';
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

  executor.run_n(taskflow, 10000).wait();
  
  REQUIRE(std::fabs(avg-32.0)<1.0);

  taskflow.dump(std::cout);
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
    //         ___________
    //        |           |
    //        v           |
    //   S -> A -> B -> cond 
    auto S = subflow.emplace([](){ });
    auto A = subflow.emplace([](){ }).succeed(S);
    auto B = subflow.emplace([&](tf::Subflow& subflow){ 

      //         ___________
      //        |           |
      //        v           |
      //   S -> A -> B -> cond 
      //        |
      //        -----> C
      //        -----> D
      //        -----> E

      auto S = subflow.emplace([](){});
      auto A = subflow.emplace([](){}).succeed(S);
      auto B = subflow.emplace([&](){ counter++; }).succeed(A);
      subflow.emplace([&, repeat=0]() mutable {
        if(repeat ++ < inner_loop) 
          return 0;
  
        repeat = 0;
        return 1;
      }).succeed(B).precede(A).name("cond");
  
      // Those are redundant tasks
      subflow.emplace([](){}).succeed(A).name("C");
      subflow.emplace([](){}).succeed(A).name("D");
      subflow.emplace([](){}).succeed(A).name("E");
    }).succeed(A);
    subflow.emplace([&, repeat=0]() mutable {
      if(repeat ++ < mid_loop) 
        return 0;
  
      repeat = 0;
      return 1;
    }).succeed(B).precede(A).name("cond");
  
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


