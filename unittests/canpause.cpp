#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

tf::Executor ex;

// --------------------------------------------------------
// Testcase: Type
// --------------------------------------------------------
TEST_CASE("one_task_no_pausable" * doctest::timeout(300)) {

    tf::Taskflow tf, taskflow2;
    tf.emplace([]() {
        return tf::TaskFlowPauseType::NoPause;
        });
    ex.run(tf);
    ex.wait_for_all();
}


//// --------------------------------------------------------
//// Testcase: Type
//// --------------------------------------------------------
TEST_CASE("one_task_pausable_skip_cur_task" * doctest::timeout(300)) {

    tf::Taskflow tf;
    auto[t1,t2] = tf.emplace([]() {
        return tf::TaskFlowPauseType::PauseSkipCurrentTask;
        }, []() {});
    t1.precede(t2);
    auto waittaskflow = ex.run(tf);
    ex.resumeTask(&tf);
    waittaskflow.wait_for(std::chrono::seconds(3));
    REQUIRE(1 == 1);
}

// --------------------------------------------------------
// Testcase: Type
// --------------------------------------------------------
TEST_CASE("one_task_pausable_no_skip_cur_task" * doctest::timeout(300)) {

    int loopindex = 0;
    int loopcount = 3;
    tf::Taskflow tf, taskflow2;
    auto [t1, t2] = tf.emplace([&]() {
        if (loopindex < loopcount)
        {
            ++loopindex;
            return tf::TaskFlowPauseType::PauseContinueCurrentTask;
        }
        else
        {
            return tf::TaskFlowPauseType::NoPause;
        }
        }, []() {});
    t1.precede(t2);
    
    auto waittaskflow = ex.run(tf);
    int resumecount = 0;
    while (waittaskflow.wait_for(std::chrono::seconds(1))!=std::future_status::ready)
    {
        ex.resumeTask(&tf);
        resumecount++;
    }
    REQUIRE(resumecount == 3);
    REQUIRE(loopindex == 3);
}
