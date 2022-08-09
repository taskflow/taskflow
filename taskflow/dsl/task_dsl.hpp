// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "meta_macro.hpp"
#include "task_analyzer.hpp"
#include "task_trait.hpp"

namespace tf {
namespace dsl {
struct EmptyContext {};
template <typename CONTEXT = EmptyContext, typename... Chains> class TaskDsl {
  using Links = Unique_t<Flatten_t<TypeList<typename Chain<Chains>::type...>>>;
  using Analyzer = typename Links::template exportTo<TaskAnalyzer>;

  using AllTasks = typename Analyzer::AllTasks;

  template <typename TASK> struct TaskCbWithContext {
    using type = TaskCb<TASK, CONTEXT>;
  };
  using TasksCB =
      typename Map_t<AllTasks,
                     TaskCbWithContext>::template exportTo<std::tuple>;

  using OneToOneLinkSet = typename Analyzer::OneToOneLinkSet;
  template <typename OneToOneLink> struct OneToOneLinkInstanceType {
    using type = typename OneToOneLink::template InstanceType<TasksCB>;
  };
  using OneToOneLinkInstances =
      typename Map_t<OneToOneLinkSet,
                     OneToOneLinkInstanceType>::template exportTo<std::tuple>;

public:
  constexpr TaskDsl(FlowBuilder &flow_builder, const CONTEXT &context = {}) {
    build_tasks_cb(flow_builder, context,
                   std::make_index_sequence<AllTasks::size>{});
    build_links(std::make_index_sequence<OneToOneLinkSet::size>{});
  }

  template <typename TASK> Task &get_task() {
    constexpr size_t TasksCBSize = std::tuple_size<TasksCB>::value;
    constexpr size_t TaskIndex =
        TupleElementByF_v<TasksCB, IsTask<TASK>::template apply>;
    static_assert(TaskIndex < TasksCBSize, "fatal: not find TaskCb in TasksCB");
    return std::get<TaskIndex>(tasksCb_).task_;
  }

private:
  template <size_t... Is>
  void build_tasks_cb(FlowBuilder &flow_builder, const CONTEXT &context,
                      std::index_sequence<Is...>) {
    auto _ = {0, (std::get<Is>(tasksCb_).build(flow_builder, context), 0)...};
    (void)_;
  }

  template <size_t... Is> void build_links(std::index_sequence<Is...>) {
    auto _ = {0, (std::get<Is>(links_).build(tasksCb_), 0)...};
    (void)_;
  }

private:
  TasksCB tasksCb_;
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
#define TF_CHAIN(link) , link->void
#define TF_CONTEXT_1(name) tf::dsl::EmptyContext
#define TF_CONTEXT_2(name, context) context
#define TF_CAPTURE_THIS_1
#define TF_CAPTURE_THIS_2 *this

///////////////////////////////////////////////////////////////////////////////
// make_task(TASK_NAME, { return a action lambda })
#define make_task(name, ...)                                                    \
  struct TF_GET_FIRST name : tf::dsl::TaskSignature,                           \
                             TF_PASTE(TF_CONTEXT_, TF_GET_ARG_COUNT name)      \
                                 name {                                        \
    using _ContextType = TF_PASTE(TF_CONTEXT_, TF_GET_ARG_COUNT name) name;    \
    TF_GET_FIRST name(const _ContextType &context) : _ContextType(context) {}  \
    auto operator()() {                                                        \
      return [TF_PASTE(TF_CAPTURE_THIS_, TF_GET_ARG_COUNT name)] __VA_ARGS__;  \
    }                                                                          \
  }

// some_tasks(A, B, C) means SomeTask
#define some_tasks(...) auto (*)(tf::dsl::SomeTask<__VA_ARGS__>)
// same as some_tasks
#define fork_tasks(...) some_tasks(__VA_ARGS__)
// same as some_tasks
#define merge_tasks(...) some_tasks(__VA_ARGS__)
// task(A) means a task A
#define task(Task) auto (*)(Task)
// taskbuild(...) build a task dsl graph
#define build_taskflow(...) tf::dsl::taskDsl<void TF_MAP(TF_CHAIN, __VA_ARGS__)>

