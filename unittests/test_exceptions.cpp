#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

// --------------------------------------------------------
// Testcase: static_task
// --------------------------------------------------------

void static_task(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter(0);

  auto A = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  auto B = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  auto C = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  auto D = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  auto E = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  auto F = taskflow.emplace([&](){ counter.fetch_add(1, std::memory_order_relaxed); });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  E.precede(F);

  REQUIRE_NOTHROW(executor.run(taskflow).get());
  REQUIRE(counter == 6);

  counter = 0;
  C.work([](){ throw std::runtime_error("x"); });
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
  REQUIRE(counter == 2);

  try {
    counter = 0;
    executor.run(taskflow).get();
    REQUIRE(false);
  }
  catch(std::runtime_error& e) {
    REQUIRE(counter == 2);
    REQUIRE(std::strcmp(e.what(), "x") == 0);
  }
}

TEST_CASE("Exception.StaticTask.1thread" * doctest::timeout(300)) {
  static_task(1);
}

TEST_CASE("Exception.StaticTask.2threads" * doctest::timeout(300)) {
  static_task(2);
}

TEST_CASE("Exception.StaticTask.3threads" * doctest::timeout(300)) {
  static_task(3);
}

TEST_CASE("Exception.StaticTask.4threads" * doctest::timeout(300)) {
  static_task(4);
}

// --------------------------------------------------------
// Testcase: condition_task
// --------------------------------------------------------

void condition_task(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter(0);

  auto A = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });
  auto B = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });
  auto C = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });
  auto D = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });
  auto E = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });
  auto F = taskflow.emplace([&](){ 
    counter.fetch_add(1, std::memory_order_relaxed); 
    return 0;
  });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  E.precede(F);

  REQUIRE_NOTHROW(executor.run(taskflow).get());
  REQUIRE(counter == 6);

  counter = 0;
  C.work([](){ throw std::runtime_error("x"); });
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
  REQUIRE(counter == 2);

  try {
    counter = 0;
    executor.run(taskflow).get();
    REQUIRE(false);
  }
  catch(std::runtime_error& e) {
    REQUIRE(counter == 2);
    REQUIRE(std::strcmp(e.what(), "x") == 0);
  }
}

TEST_CASE("Exception.ConditionTask.1thread" * doctest::timeout(300)) {
  condition_task(1);
}

TEST_CASE("Exception.ConditionTask.2threads" * doctest::timeout(300)) {
  condition_task(2);
}

TEST_CASE("Exception.ConditionTask.3threads" * doctest::timeout(300)) {
  condition_task(3);
}

TEST_CASE("Exception.ConditionTask.4threads" * doctest::timeout(300)) {
  condition_task(4);
}

// --------------------------------------------------------
// Testcase: multicondition_task
// --------------------------------------------------------

void multicondition_task(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  std::atomic<size_t> counter(0);

  auto A = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });
  auto B = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });
  auto C = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });
  auto D = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });
  auto E = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });
  auto F = taskflow.emplace([&](){ 
    tf::SmallVector<int> ret = {0};
    counter.fetch_add(1, std::memory_order_relaxed); 
    return ret;
  });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  E.precede(F);

  REQUIRE_NOTHROW(executor.run(taskflow).get());
  REQUIRE(counter == 6);

  counter = 0;
  C.work([](){ throw std::runtime_error("x"); });
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
  REQUIRE(counter == 2);

  try {
    counter = 0;
    executor.run(taskflow).get();
    REQUIRE(false);
  }
  catch(std::runtime_error& e) {
    REQUIRE(counter == 2);
    REQUIRE(std::strcmp(e.what(), "x") == 0);
  }
}

TEST_CASE("Exception.MultiConditionTask.1thread" * doctest::timeout(300)) {
  multicondition_task(1);
}

TEST_CASE("Exception.MultiConditionTask.2threads" * doctest::timeout(300)) {
  multicondition_task(2);
}

TEST_CASE("Exception.MultiConditionTask.3threads" * doctest::timeout(300)) {
  multicondition_task(3);
}

TEST_CASE("Exception.MultiConditionTask.4threads" * doctest::timeout(300)) {
  multicondition_task(4);
}

// ----------------------------------------------------------------------------
// Subflow Task
// ----------------------------------------------------------------------------

void subflow_task(unsigned W) {

  tf::Taskflow taskflow;
  tf::Executor executor(W);
  
  // subflow work throws
  for(int i=0; i<100; i++) {
    taskflow.emplace([](tf::Subflow& sf){
      throw std::runtime_error("x");
      sf.emplace([](){});
      sf.emplace([](){ throw std::runtime_error("z"); });
    });
  }
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);

  taskflow.clear();
  
  // subflow's task throws
  for(int i=0; i<100; i++) {
    taskflow.emplace([](tf::Subflow& sf){
      sf.emplace([](){});
      sf.emplace([](){ throw std::runtime_error("y"); });
    });
  }
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "y", std::runtime_error);
}

TEST_CASE("Exception.SubflowTask.1thread" * doctest::timeout(300)) {
  subflow_task(1);
}

TEST_CASE("Exception.SubflowTask.2threads" * doctest::timeout(300)) {
  subflow_task(2);
}

TEST_CASE("Exception.SubflowTask.3threads" * doctest::timeout(300)) {
  subflow_task(3);
}

TEST_CASE("Exception.SubflowTask.4threads" * doctest::timeout(300)) {
  subflow_task(4);
}

// ----------------------------------------------------------------------------
// Joined Subflow
// ----------------------------------------------------------------------------

void joined_subflow_1(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  taskflow.emplace([&] (tf::Subflow& sf0) {
    for (int i = 0; i < 100; ++i) {
      sf0.emplace([&] (tf::Subflow& sf1) {

        for (int j = 0; j < 2; ++j) {
          sf1.emplace([] () {
            throw std::runtime_error("x");
          }).name(std::string("sf1-child-") + std::to_string(j));
        }

        sf1.join();
        // [NOTE]: We cannot guarantee post_join won't run since
        // the exception also triggers cancellation which in turns 
        // bypasses the two tasks inside sf1. In this case, sf1.join
        // will succeed and set post_join to true.

        //post_join = true;
      }).name(std::string("sf1-") + std::to_string(i));
    }
  }).name("sf0");
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
  //REQUIRE(post_join == false);

}

TEST_CASE("Exception.JoinedSubflow1.1thread" * doctest::timeout(300)) {
  joined_subflow_1(1);
}

TEST_CASE("Exception.JoinedSubflow1.2threads" * doctest::timeout(300)) {
  joined_subflow_1(2);
}

TEST_CASE("Exception.JoinedSubflow1.3threads" * doctest::timeout(300)) {
  joined_subflow_1(3);
}

TEST_CASE("Exception.JoinedSubflow1.4threads" * doctest::timeout(300)) {
  joined_subflow_1(4);
}

// ----------------------------------------------------------------------------
// Joined Subflow 2
// ----------------------------------------------------------------------------

void joined_subflow_2(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::atomic<bool> post_join {false};

  taskflow.emplace([&](tf::Subflow& sf0){
    for (int j = 0; j < 16; ++j) {
      sf0.emplace([] () {
        throw std::runtime_error("x");
      });
    }
    try {
      sf0.join();
      post_join = true;
    } catch(const std::runtime_error& re) {
      REQUIRE(std::strcmp(re.what(), "x") == 0);
    }
  });
  executor.run(taskflow).wait();
  REQUIRE(post_join == false);
}

TEST_CASE("Exception.JoinedSubflow2.1thread" * doctest::timeout(300)) {
  joined_subflow_2(1);
}

TEST_CASE("Exception.JoinedSubflow2.2threads" * doctest::timeout(300)) {
  joined_subflow_2(2);
}

TEST_CASE("Exception.JoinedSubflow2.3threads" * doctest::timeout(300)) {
  joined_subflow_2(3);
}

TEST_CASE("Exception.JoinedSubflow2.4threads" * doctest::timeout(300)) {
  joined_subflow_2(4);
}

// ----------------------------------------------------------------------------
// Joined Subflow Exception 3
// ----------------------------------------------------------------------------

void joined_subflow_3(unsigned N) {

  tf::Executor executor(N);
  tf::Taskflow taskflow;

  size_t num_tasks = 0;
  
  // implicit join
  taskflow.emplace([&](tf::Subflow& sf) {
    tf::Task W = sf.emplace([&](){ ++num_tasks; });
    tf::Task X = sf.emplace([&](){ ++num_tasks; throw std::runtime_error("x"); });
    tf::Task Y = sf.emplace([&](){ ++num_tasks; });
    tf::Task Z = sf.emplace([&](){ ++num_tasks; });
    W.precede(X);
    X.precede(Y);
    Y.precede(Z);
  });
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
  REQUIRE(num_tasks == 2);

  // explicit join
  num_tasks = 0;
  taskflow.clear();
  taskflow.emplace([&](tf::Subflow& sf) {
    tf::Task W = sf.emplace([&](){ ++num_tasks; });
    tf::Task X = sf.emplace([&](){ ++num_tasks; throw std::runtime_error("y"); });
    tf::Task Y = sf.emplace([&](){ ++num_tasks; });
    tf::Task Z = sf.emplace([&](){ ++num_tasks; });
    W.precede(X);
    X.precede(Y);
    Y.precede(Z);
    sf.join();
  });
  
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "y", std::runtime_error);
  REQUIRE(num_tasks == 2);
}

TEST_CASE("Exception.JoinedSubflow3.1thread" * doctest::timeout(300)) {
  joined_subflow_3(1);
}

TEST_CASE("Exception.JoinedSubflow3.2threads" * doctest::timeout(300)) {
  joined_subflow_3(2);
}

TEST_CASE("Exception.JoinedSubflow3.3threads" * doctest::timeout(300)) {
  joined_subflow_3(3);
}

TEST_CASE("Exception.JoinedSubflow3.4threads" * doctest::timeout(300)) {
  joined_subflow_3(4);
}

// ----------------------------------------------------------------------------
// Nested Subflow
// ----------------------------------------------------------------------------

void nested_subflow(unsigned N) {

  tf::Executor executor(N);
  tf::Taskflow taskflow;

  size_t num_tasks = 0;

  // level 1
  taskflow.emplace([&](tf::Subflow& sf1) {
    tf::Task V1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("V1");
    tf::Task W1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("W1");
    
    // level 2
    tf::Task X1 = sf1.emplace([&num_tasks](tf::Subflow& sf2){ 
      ++num_tasks; 

      tf::Task V2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("V2");
      tf::Task W2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("W2");
      
      // level 3
      tf::Task X2 = sf2.emplace([&num_tasks](tf::Subflow& sf3) {
        ++num_tasks;

        tf::Task V3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("V3");
        tf::Task W3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("W3");

        // level 4
        tf::Task X3 = sf3.emplace([&num_tasks](tf::Subflow& sf4){
          ++num_tasks;

          tf::Task V4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("V4");
          tf::Task W4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("W4");
          tf::Task X4 = sf4.emplace([&num_tasks](){ 
            ++num_tasks; 
            throw std::runtime_error("x");
          }).name("X4 (throw)");
          tf::Task Y4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("Y4");
          tf::Task Z4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("Z4");

          V4.precede(W4);
          W4.precede(X4);
          X4.precede(Y4);
          Y4.precede(Z4);
        }).name("sf-4");

        tf::Task Y3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("Y3");
        tf::Task Z3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("Z3");

        V3.precede(W3);
        W3.precede(X3);
        X3.precede(Y3);
        Y3.precede(Z3);
      }).name("sf3");

      tf::Task Y2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("Y2");
      tf::Task Z2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("Z2");

      V2.precede(W2);
      W2.precede(X2);
      X2.precede(Y2);
      Y2.precede(Z2);
    }).name("sf-2");

    tf::Task Y1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("Y1");
    tf::Task Z1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("Z1");

    V1.precede(W1);
    W1.precede(X1);
    X1.precede(Y1);
    Y1.precede(Z1);
  }).name("sf-1");

  REQUIRE_THROWS_WITH_AS(executor.run_n(taskflow, 10).get(), "x", std::runtime_error);
  REQUIRE(num_tasks == 12);
  
  //taskflow.dump(std::cout);

  // corun the nested subflow from an async task
  num_tasks = 0;
  executor.async([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow), "x", std::runtime_error);
  }).get(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an silent async task
  num_tasks = 0;
  executor.silent_async([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow), "x", std::runtime_error);
  });
  executor.wait_for_all(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an async task's runtime
  num_tasks = 0;
  executor.async([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow), "x", std::runtime_error);
  }).get(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an silent-async task's runtime
  num_tasks = 0;
  executor.silent_async([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow), "x", std::runtime_error);
  });
  executor.wait_for_all(); 
  REQUIRE(num_tasks == 12);

}

TEST_CASE("Exception.NestedSubflow.1thread" * doctest::timeout(300)) {
  nested_subflow(1);
}

TEST_CASE("Exception.NestedSubflow.2threads" * doctest::timeout(300)) {
  nested_subflow(2);
}

TEST_CASE("Exception.NestedSubflow.3threads" * doctest::timeout(300)) {
  nested_subflow(3);
}

TEST_CASE("Exception.NestedSubflow.4threads" * doctest::timeout(300)) {
  nested_subflow(4);
}

// ----------------------------------------------------------------------------
// Nested Subflow 2
// ----------------------------------------------------------------------------

void nested_subflow_2(unsigned N) {

  tf::Executor executor(N);
  tf::Taskflow taskflow;

  size_t num_tasks = 0;

  // level 1
  taskflow.emplace([&](tf::Subflow& sf1) {
    tf::Task V1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("V1");
    tf::Task W1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("W1");
    
    // level 2
    tf::Task X1 = sf1.emplace([&num_tasks](tf::Subflow& sf2){ 
      ++num_tasks; 

      tf::Task V2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("V2");
      tf::Task W2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("W2");
      
      // level 3
      tf::Task X2 = sf2.emplace([&num_tasks](tf::Subflow& sf3) {
        ++num_tasks;

        tf::Task V3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("V3");
        tf::Task W3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("W3");

        // level 4
        tf::Task X3 = sf3.emplace([&num_tasks](tf::Subflow& sf4){
          ++num_tasks;

          tf::Task V4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("V4");
          tf::Task W4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("W4");
          tf::Task X4 = sf4.emplace([&num_tasks](){ 
            ++num_tasks; 
            throw std::runtime_error("x");
          }).name("X4 (throw)");
          tf::Task Y4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("Y4");
          tf::Task Z4 = sf4.emplace([&num_tasks](){ ++num_tasks; }).name("Z4");

          V4.precede(W4);
          W4.precede(X4);
          X4.precede(Y4);
          Y4.precede(Z4);
          
          sf4.join();

        }).name("sf-4");

        tf::Task Y3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("Y3");
        tf::Task Z3 = sf3.emplace([&num_tasks](){ ++num_tasks; }).name("Z3");

        V3.precede(W3);
        W3.precede(X3);
        X3.precede(Y3);
        Y3.precede(Z3);

        sf3.join();

      }).name("sf3");

      tf::Task Y2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("Y2");
      tf::Task Z2 = sf2.emplace([&num_tasks](){ ++num_tasks; }).name("Z2");

      V2.precede(W2);
      W2.precede(X2);
      X2.precede(Y2);
      Y2.precede(Z2);

      sf2.join();

    }).name("sf-2");

    tf::Task Y1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("Y1");
    tf::Task Z1 = sf1.emplace([&num_tasks](){ ++num_tasks; }).name("Z1");

    V1.precede(W1);
    W1.precede(X1);
    X1.precede(Y1);
    Y1.precede(Z1);

    sf1.join();

  }).name("sf-1");

  REQUIRE_THROWS_WITH_AS(executor.run_n(taskflow, 10).get(), "x", std::runtime_error);
  REQUIRE(num_tasks == 12);
  
  //taskflow.dump(std::cout);

  // corun the nested subflow from an async task
  num_tasks = 0;
  executor.async([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow), "x", std::runtime_error);
  }).get(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an silent async task
  num_tasks = 0;
  executor.silent_async([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow), "x", std::runtime_error);
  });
  executor.wait_for_all(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an async task's runtime
  num_tasks = 0;
  executor.async([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow), "x", std::runtime_error);
  }).get(); 
  REQUIRE(num_tasks == 12);
  
  // corun the nested subflow from an silent-async task's runtime
  num_tasks = 0;
  executor.silent_async([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow), "x", std::runtime_error);
  });
  executor.wait_for_all(); 
  REQUIRE(num_tasks == 12);

}

TEST_CASE("Exception.NestedSubflow2.1thread" * doctest::timeout(300)) {
  nested_subflow_2(1);
}

TEST_CASE("Exception.NestedSubflow2.2threads" * doctest::timeout(300)) {
  nested_subflow_2(2);
}

TEST_CASE("Exception.NestedSubflow2.3threads" * doctest::timeout(300)) {
  nested_subflow_2(3);
}

TEST_CASE("Exception.NestedSubflow2.4threads" * doctest::timeout(300)) {
  nested_subflow_2(4);
}

// ----------------------------------------------------------------------------
// Executor Corun Exception 1
// ----------------------------------------------------------------------------

void executor_corun_1(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;

  taskflow1.emplace([](){
    throw std::runtime_error("x");
  });

  taskflow2.emplace([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow1), "x", std::runtime_error);
  });

  executor.run(taskflow2).get();
  
  taskflow1.clear();
  taskflow2.clear();

  for(size_t i=0; i<100; i++) {
    taskflow1.emplace([](tf::Subflow& sf){
      for(size_t j=0; j<100; j++) {
        sf.emplace([&](){
          throw std::runtime_error("y");
        });
      }
    });
  }
  
  taskflow2.emplace([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow1), "y", std::runtime_error);
  });

  executor.run(taskflow2).get();
}

TEST_CASE("Exception.ExecutorCorun1.1thread" * doctest::timeout(300)) {
  executor_corun_1(1);
}

TEST_CASE("Exception.ExecutorCorun1.2threads" * doctest::timeout(300)) {
  executor_corun_1(2);
}

TEST_CASE("Exception.ExecutorCorun1.3threads" * doctest::timeout(300)) {
  executor_corun_1(3);
}

TEST_CASE("Exception.ExecutorCorun1.4threads" * doctest::timeout(300)) {
  executor_corun_1(4);
}

// ----------------------------------------------------------------------------
// Executor Corun Exception 2
// ----------------------------------------------------------------------------

void executor_corun_2(unsigned W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  size_t counter = 0;

  auto A = taskflow.emplace([&](){ counter++; });
  auto B = taskflow.emplace([&](){ counter++; });
  auto C = taskflow.emplace([&](){ throw std::runtime_error("x"); });
  auto D = taskflow.emplace([&](){ counter++; });
  auto E = taskflow.emplace([&](){ counter++; });
  auto F = taskflow.emplace([&](){ counter++; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  E.precede(F);
  
  // uncaught corun exception propagates to the topology 
  tf::Taskflow taskflow2;
  taskflow2.emplace([&](){
    executor.corun(taskflow);
  });
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow2).get(), "x", std::runtime_error);
  REQUIRE(counter == 2);

  // catch corun exception directly
  tf::Taskflow taskflow3;
  taskflow3.emplace([&](){
    REQUIRE_THROWS_WITH_AS(executor.corun(taskflow), "x", std::runtime_error);
  });
  executor.run(taskflow3).get();
  REQUIRE(counter == 4);
}

TEST_CASE("Exception.ExecutorCorun2.1thread" * doctest::timeout(300)) {
  executor_corun_2(1);
}

TEST_CASE("Exception.ExecutorCorun2.2threads" * doctest::timeout(300)) {
  executor_corun_2(2);
}

TEST_CASE("Exception.ExecutorCorun2.3threads" * doctest::timeout(300)) {
  executor_corun_2(3);
}

TEST_CASE("Exception.ExecutorCorun2.4threads" * doctest::timeout(300)) {
  executor_corun_2(4);
}

// ----------------------------------------------------------------------------
// runtime_corun
// ----------------------------------------------------------------------------

void runtime_corun_1(unsigned W) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;
  
  taskflow1.emplace([](){
    throw std::runtime_error("x");
  });

  auto task = taskflow2.emplace([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow1), "x", std::runtime_error);
  });
  executor.run(taskflow2).get();
  
  taskflow1.clear();
  for(size_t i=0; i<100; i++) {
    taskflow1.emplace([](tf::Subflow& sf){
      for(size_t j=0; j<100; j++) {
        sf.emplace([&](){
          throw std::runtime_error("x");
        });
      }
    });
  }
  executor.run(taskflow2).get();
  
  // change it to parent propagation
  task.work([&](tf::Runtime& rt){
    rt.corun(taskflow1);
  });
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow2).get(), "x", std::runtime_error);
}

TEST_CASE("Exception.RuntimeCorun1.1thread" * doctest::timeout(300)) {
  runtime_corun_1(1);
}

TEST_CASE("Exception.RuntimeCorun1.2threads" * doctest::timeout(300)) {
  runtime_corun_1(2);
}

TEST_CASE("Exception.RuntimeCorun1.3threads" * doctest::timeout(300)) {
  runtime_corun_1(3);
}

TEST_CASE("Exception.RuntimeCorun1.4threads" * doctest::timeout(300)) {
  runtime_corun_1(4);
}

// ----------------------------------------------------------------------------
// Runtime Corun Exception 2
// ----------------------------------------------------------------------------

void runtime_corun_2(unsigned W) {
  
  tf::Taskflow taskflow;
  tf::Executor executor(W);

  size_t counter = 0;

  auto A = taskflow.emplace([&](){ counter++; });
  auto B = taskflow.emplace([&](){ counter++; });
  auto C = taskflow.emplace([&](){ throw std::runtime_error("x"); });
  auto D = taskflow.emplace([&](){ counter++; });
  auto E = taskflow.emplace([&](){ counter++; });
  auto F = taskflow.emplace([&](){ counter++; });

  A.precede(B);
  B.precede(C);
  C.precede(D);
  D.precede(E);
  E.precede(F);
  
  // uncaught corun exception propagates to the topology 
  tf::Taskflow taskflow2;
  taskflow2.emplace([&](tf::Runtime& rt){
    rt.corun(taskflow);
  });
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow2).get(), "x", std::runtime_error);
  REQUIRE(counter == 2);

  // catch corun exception directly
  tf::Taskflow taskflow3;
  taskflow3.emplace([&](tf::Runtime& rt){
    REQUIRE_THROWS_WITH_AS(rt.corun(taskflow), "x", std::runtime_error);
  });
  executor.run(taskflow3).get();
  REQUIRE(counter == 4);
}

TEST_CASE("Exception.RuntimeCorun2.1thread" * doctest::timeout(300)) {
  runtime_corun_2(1);
}

TEST_CASE("Exception.RuntimeCorun2.2threads" * doctest::timeout(300)) {
  runtime_corun_2(2);
}

TEST_CASE("Exception.RuntimeCorun2.3threads" * doctest::timeout(300)) {
  runtime_corun_2(3);
}

TEST_CASE("Exception.RuntimeCorun2.4threads" * doctest::timeout(300)) {
  runtime_corun_2(4);
}


// ----------------------------------------------------------------------------
// module_task
// ----------------------------------------------------------------------------

void module_task(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow1;
  tf::Taskflow taskflow2;

  taskflow1.emplace([](){
    throw std::runtime_error("x");
  });
  taskflow2.composed_of(taskflow1);
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow2).get(), "x", std::runtime_error);

  //taskflow1.clear();
  //taskflow1.emplace([](tf::Subflow& sf){
  //  sf.emplace([](){
  //    throw std::runtime_error("y");
  //  });
  //});
  //REQUIRE_THROWS_WITH_AS(executor.run(taskflow2).get(), "y", std::runtime_error);
}

TEST_CASE("Exception.ModuleTask.1thread" * doctest::timeout(300)) {
  module_task(1);
}

TEST_CASE("Exception.ModuleTask.2threads" * doctest::timeout(300)) {
  module_task(2);
}

TEST_CASE("Exception.ModuleTask.3threads" * doctest::timeout(300)) {
  module_task(3);
}

TEST_CASE("Exception.ModuleTask.4threads" * doctest::timeout(300)) {
  module_task(4);
}

// ----------------------------------------------------------------------------
// Exception.Async
// ----------------------------------------------------------------------------

void async_task(unsigned W) {

  // executor async
  tf::Executor executor(W);

  auto fu1 = executor.async([](){
    return 1;
  });
  REQUIRE(fu1.get() == 1);
  
  auto fu2 = executor.async([](){
    throw std::runtime_error("x");
  });
  REQUIRE_THROWS_WITH_AS(fu2.get(), "x", std::runtime_error);
  
  // exception is caught without any action
  executor.silent_async([](){
    throw std::runtime_error("y"); 
  });

  executor.wait_for_all();
}

TEST_CASE("Exception.Async.1thread" * doctest::timeout(300)) {
  async_task(1);
}

TEST_CASE("Exception.Async.2threads" * doctest::timeout(300)) {
  async_task(2);
}

TEST_CASE("Exception.Async.3threads" * doctest::timeout(300)) {
  async_task(3);
}

TEST_CASE("Exception.Async.4threads" * doctest::timeout(300)) {
  async_task(4);
}

// ----------------------------------------------------------------------------
// Async Task with Runtime
// ----------------------------------------------------------------------------

void async_with_runtime(unsigned W) {
  
  tf::Executor executor(W);
  std::vector<std::future<void>> futures;

  for(size_t i=0; i<1024; i++) {
    futures.emplace_back(executor.async([](tf::Runtime&){
      throw std::runtime_error("x");
    }));
  }
  
  for(auto& fu : futures) {
    REQUIRE_THROWS_WITH_AS(fu.get(), "x", std::runtime_error);
  }
  
  // silently caught by the task
  executor.silent_async([](tf::Runtime&){
    throw std::runtime_error("x");
  });

  executor.wait_for_all();
}

TEST_CASE("Exception.Async.Runtime.1thread" * doctest::timeout(300)) {
  async_with_runtime(1);
}

TEST_CASE("Exception.Async.Runtime.2threads" * doctest::timeout(300)) {
  async_with_runtime(2);
}

TEST_CASE("Exception.Async.Runtime.3threads" * doctest::timeout(300)) {
  async_with_runtime(3);
}

TEST_CASE("Exception.Async.Runtime.4threads" * doctest::timeout(300)) {
  async_with_runtime(4);
}

// ----------------------------------------------------------------------------
// Dependent Async Task with Runtime
// ----------------------------------------------------------------------------

void dependent_async_with_runtime(unsigned W) {
  
  tf::Executor executor(W);
  std::vector<std::future<void>> futures;

  for(size_t i=0; i<1024; i++) {
    auto [t, f] = executor.dependent_async([](tf::Runtime&){
      throw std::runtime_error("x");
    });
  }
  
  for(auto& fu : futures) {
    REQUIRE_THROWS_WITH_AS(fu.get(), "x", std::runtime_error);
  }
  
  // silently caught by the task
  executor.silent_dependent_async([](tf::Runtime&){
    throw std::runtime_error("x");
  });

  executor.wait_for_all();
}

TEST_CASE("Exception.DependentAsync.Runtime.1thread" * doctest::timeout(300)) {
  dependent_async_with_runtime(1);
}

TEST_CASE("Exception.DependentAsync.Runtime.2threads" * doctest::timeout(300)) {
  dependent_async_with_runtime(2);
}

TEST_CASE("Exception.DependentAsync.Runtime.3threads" * doctest::timeout(300)) {
  dependent_async_with_runtime(3);
}

TEST_CASE("Exception.DependentAsync.Runtime.4threads" * doctest::timeout(300)) {
  dependent_async_with_runtime(4);
}

/*
// ----------------------------------------------------------------------------
// Runtime Async Task
// ----------------------------------------------------------------------------

void runtime_async_task(unsigned W) {

  // executor async
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  int flag = 0;

  // runtime async
  auto A = taskflow.emplace([](tf::Runtime& rt){
    auto fu1 = rt.async([](){ return 1; });
    REQUIRE(fu1.get() == 1);
    auto fu2 = rt.async([](){ throw std::runtime_error("z"); });
    REQUIRE_THROWS_WITH_AS(fu2.get(), "z", std::runtime_error);
  });
  auto B = taskflow.emplace([&](){
    flag = 1;
  });
  executor.run(taskflow).wait();
  REQUIRE(flag == 1);

  // runtime silent async
  flag = 0;
  taskflow.clear();
  A = taskflow.emplace([&](tf::Runtime& rt){
    rt.silent_async([&](){ throw std::runtime_error("a"); });
    REQUIRE_THROWS_WITH_AS(rt.corun(), "a", std::runtime_error); 
    flag = 1;
  });
  B = taskflow.emplace([&](){
    flag = 2;
  });
  A.precede(B);
  executor.run(taskflow).get();
  REQUIRE(flag == 2);
  
  // runtime silent async
  flag = 0;
  taskflow.clear();
  A = taskflow.emplace([&](tf::Runtime& rt){
    rt.silent_async([&](){ throw std::runtime_error("a"); });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    rt.corun();
    flag = 1;  // can't guarantee since rt.silent_async can finish 
               // before corun finishes
  });
  B = taskflow.emplace([&](){
    flag = 2;
  });
  A.precede(B);
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "a", std::runtime_error);
  REQUIRE(flag == 0);
}

TEST_CASE("Exception.RuntimeAsync.2threads" * doctest::timeout(300)) {
  runtime_async_task(2);
}

TEST_CASE("Exception.RuntimeAsync.3threads" * doctest::timeout(300)) {
  runtime_async_task(3);
}

TEST_CASE("Exception.RuntimeAsync.4threads" * doctest::timeout(300)) {
  runtime_async_task(4);
}
*/

// ----------------------------------------------------------------------------
// Exception.ThreadSafety
// ----------------------------------------------------------------------------

void thread_safety(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  for(int i=0; i<1000; i++) {
    taskflow.emplace([&](){ throw std::runtime_error("x"); });
  }

  // thread sanitizer should not report any data race 
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
}

TEST_CASE("Exception.ThreadSafety.1thread" * doctest::timeout(300)) {
  thread_safety(1);
}

TEST_CASE("Exception.ThreadSafety.2threads" * doctest::timeout(300)) {
  thread_safety(2);
}

TEST_CASE("Exception.ThreadSafety.3threads" * doctest::timeout(300)) {
  thread_safety(3);
}

TEST_CASE("Exception.ThreadSafety.4threads" * doctest::timeout(300)) {
  thread_safety(4);
}


// ----------------------------------------------------------------------------
// Semaphores
// ----------------------------------------------------------------------------

void semaphore1(unsigned W) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  tf::Semaphore semaphore(1);
  
  tf::Task A = taskflow.emplace([](){});
  tf::Task B = taskflow.emplace([](){ throw std::runtime_error("exception"); });
  tf::Task C = taskflow.emplace([](){});
  tf::Task D = taskflow.emplace([](){});

  A.precede(B);
  B.precede(C);
  C.precede(D);
  
  A.acquire(semaphore);
  D.release(semaphore);

  REQUIRE(semaphore.value() == 1);
  
  // when B throws the exception, D will not run and thus semaphore is not released
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "exception", std::runtime_error);

  REQUIRE(semaphore.value() == 0);
  
  // reset the semaphore to a clean state before running the taskflow again
  semaphore.reset();
  
  REQUIRE(semaphore.value() == 1);
  
  // run it again
  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "exception", std::runtime_error);
}

TEST_CASE("Exception.Semaphore.1thread" * doctest::timeout(300)) {
  semaphore1(1);
}

TEST_CASE("Exception.Semaphore.2threads" * doctest::timeout(300)) {
  semaphore1(2);
}

TEST_CASE("Exception.Semaphore.3threads" * doctest::timeout(300)) {
  semaphore1(3);
}

TEST_CASE("Exception.Semaphore.4threads" * doctest::timeout(300)) {
  semaphore1(4);
}

