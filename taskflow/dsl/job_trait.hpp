// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "../core/task.hpp"
#include "type_list.hpp"
#include <type_traits>

namespace tf {
struct JobSignature {};

template <typename J> struct JobCb {
  using type = JobCb<J>;
  using JobType = J;
  void build(FlowBuilder &build) { job_ = build.emplace(JobType{}()); }
  Task job_;
};

template <typename... J> struct SomeJob { using JobList = TypeList<J...>; };

template <typename J, typename = void> struct JobTrait;

// a job self
template <typename J>
struct JobTrait<J, void_t<std::is_base_of<JobSignature, J>>> {
  using JobList = TypeList<J>;
};

template <typename... J> struct JobTrait<SomeJob<J...>> {
  using JobList = typename SomeJob<J...>::JobList;
};

} // namespace tf
