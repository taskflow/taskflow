// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "job_trait.hpp"
#include "task_analyzer.hpp"

namespace tf {
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

#define __def_task(name, ...)                                                  \
  struct name : tf::JobSignature {                                             \
    auto operator()() __VA_ARGS__                                              \
  }

#define __some_tsk(...) auto (*)(tf::SomeJob<__VA_ARGS__>)
#define __fork(...) __some_tsk(__VA_ARGS__)
#define __merge(...) __some_tsk(__VA_ARGS__)
#define __tsk(Job) auto (*)(Job)
#define __chain(link) link->void
#define __taskbuild(...) tf::TaskDsl<__VA_ARGS__>

} // namespace tf
