// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "job_trait.hpp"
#include "task_analyzer.hpp"

namespace tf {
namespace dsl {
template <typename... Chains> class TaskDsl {
  using Links = Unique_t<Flatten_t<TypeList<typename Chain<Chains>::type...>>>;
  using Analyzer = typename Links::template exportTo<TaskAnalyzer>;

  using AllJobs = typename Analyzer::AllJobs;
  using JobsCb = typename Map_t<AllJobs, JobCb>::template exportTo<std::tuple>;

  using OneToOneLinkSet = typename Analyzer::OneToOneLinkSet;
  template <typename OneToOneLink> struct OneToOneLinkInstanceType {
    using type = typename OneToOneLink::template InstanceType<JobsCb>;
  };
  using OneToOneLinkInstances =
      typename Map_t<OneToOneLinkSet,
                     OneToOneLinkInstanceType>::template exportTo<std::tuple>;

public:
  constexpr TaskDsl(FlowBuilder &flow_builder) {
    build_jobs_cb(flow_builder, std::make_index_sequence<AllJobs::size>{});
    build_links(std::make_index_sequence<OneToOneLinkSet::size>{});
  }

private:
  template <size_t... Is>
  void build_jobs_cb(FlowBuilder &flow_builder, std::index_sequence<Is...>) {
    auto _ = {0, (std::get<Is>(jobsCb_).build(flow_builder), 0)...};
    (void)_;
  }

  template <size_t... Is> void build_links(std::index_sequence<Is...>) {
    auto _ = {0, (std::get<Is>(links_).build(jobsCb_), 0)...};
    (void)_;
  }

private:
  JobsCb jobsCb_;
  OneToOneLinkInstances links_;
};
}
} // namespace tf

// def_task(TASK_NAME, { return a action lambda })
#define def_task(name, ...)                                                    \
  struct name : tf::dsl::JobSignature {                                        \
    auto operator()() __VA_ARGS__                                              \
  }

// some_task(A, B, C) means SomeJob
#define some_task(...) auto (*)(tf::dsl::SomeJob<__VA_ARGS__>)
// same as some_task
#define fork(...) some_task(__VA_ARGS__)
// same as some_task
#define merge(...) some_task(__VA_ARGS__)
// task(A) means a task A
#define task(Job) auto (*)(Job)
// chain(task(A) -> task(B) -> ...) for build a task chain
#define chain(link) link->void
// taskbuild(...) build a task dsl graph
#define taskbuild(...) tf::dsl::TaskDsl<__VA_ARGS__>
