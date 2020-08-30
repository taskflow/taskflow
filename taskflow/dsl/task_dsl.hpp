// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "job_trait.hpp"
#include "meta_macro.hpp"
#include "task_analyzer.hpp"

namespace tf {
namespace dsl {
struct EmptyContext {};
template <typename CONTEXT = EmptyContext, typename... Chains> class TaskDsl {
  using Links = Unique_t<Flatten_t<TypeList<typename Chain<Chains>::type...>>>;
  using Analyzer = typename Links::template exportTo<TaskAnalyzer>;

  using AllJobs = typename Analyzer::AllJobs;

  template <typename J> struct JobCbWithContext {
    using type = JobCb<J, CONTEXT>;
  };
  using JobsCb =
      typename Map_t<AllJobs, JobCbWithContext>::template exportTo<std::tuple>;

  using OneToOneLinkSet = typename Analyzer::OneToOneLinkSet;
  template <typename OneToOneLink> struct OneToOneLinkInstanceType {
    using type = typename OneToOneLink::template InstanceType<JobsCb>;
  };
  using OneToOneLinkInstances =
      typename Map_t<OneToOneLinkSet,
                     OneToOneLinkInstanceType>::template exportTo<std::tuple>;

public:
  constexpr TaskDsl(FlowBuilder &flow_builder, const CONTEXT &context = {}) {
    build_jobs_cb(flow_builder, context,
                  std::make_index_sequence<AllJobs::size>{});
    build_links(std::make_index_sequence<OneToOneLinkSet::size>{});
  }

private:
  template <size_t... Is>
  void build_jobs_cb(FlowBuilder &flow_builder, const CONTEXT &context,
                     std::index_sequence<Is...>) {
    auto _ = {0, (std::get<Is>(jobsCb_).build(flow_builder, context), 0)...};
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

template <typename = void, typename... Chains, typename CONTEXT = EmptyContext>
constexpr TaskDsl<CONTEXT, Chains...> taskDsl(FlowBuilder &flow_builder,
                                              CONTEXT &&context = {}) {
  return {flow_builder, context};
}

} // namespace dsl
} // namespace tf

///////////////////////////////////////////////////////////////////////////////
#define chain(n, link) , link->void

///////////////////////////////////////////////////////////////////////////////
// def_task(TASK_NAME, { return a action lambda })
#define def_task(name, ...)                                                    \
  struct name : tf::dsl::JobSignature, tf::dsl::EmptyContext {                 \
    name(const EmptyContext &context) : EmptyContext(context) {}               \
    auto operator()() { return [] __VA_ARGS__; }                               \
  };
#define def_taskc(name, Context, ...)                                          \
  struct name : tf::dsl::JobSignature, Context {                               \
    name(const Context &context) : Context(context) {}                         \
    auto operator()() {                                                        \
      /* copy *this(copy CONTEXT to lambda) */                                 \
      return [*this] __VA_ARGS__;                                              \
    }                                                                          \
  };

// some_task(A, B, C) means SomeJob
#define some_task(...) auto (*)(tf::dsl::SomeJob<__VA_ARGS__>)
// same as some_task
#define fork(...) some_task(__VA_ARGS__)
// same as some_task
#define merge(...) some_task(__VA_ARGS__)
// task(A) means a task A
#define task(Job) auto (*)(Job)
// taskbuild(...) build a task dsl graph
#define taskbuild(...)                                                         \
  tf::dsl::taskDsl<void TF_PASTE(TF_REPEAT_, TF_GET_ARG_COUNT(__VA_ARGS__))(   \
      chain, 0, __VA_ARGS__)>
