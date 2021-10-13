#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

// --------------------------------------------------------
// Testcase: 
// --------------------------------------------------------

//template <typename D>
//void pipeline_1F_1L(unsigned w) {
//  tf::Taskflow taskflow;
//  tf::Executor executor(w);
//  
//  // iterate different data amount (1, 2, 3, 4, 5, ... 1000000)
//  for(size_t N=0; N<10000; N++) {
//  std::vector<D> source(N);
//
//  tf::make_pipeline<D, 1>(
//    tf::Filter{tf::FilterType::PA
//      [](auto& d){
//      REQUIRE(d.input() == nullptr);
//    });
//
//}

TEST_CASE("Pipeline.1F.1L.1W.int" * doctest::timeout(300)) {
//  pipeline_1F_1L<int>(1);
}
