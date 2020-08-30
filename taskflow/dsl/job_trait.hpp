// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "../core/task.hpp"
#include "type_list.hpp"
#include <type_traits>

namespace tf {
namespace dsl {
struct JobSignature {};

template <typename J, typename CONTEXT> struct JobCb {
  using JobType = J;
  void build(FlowBuilder &build, const CONTEXT &context) {
    job_ = build.emplace(JobType{context}());
  }

  Task job_;
};

template <typename J, typename = void> struct JobTrait;

template <typename... J> struct SomeJob {
  using JobList =
      Unique_t<Flatten_t<TypeList<typename JobTrait<J>::JobList...>>>;
};

// a job self
template <typename J>
struct JobTrait<J, std::enable_if_t<std::is_base_of<JobSignature, J>::value>> {
  using JobList = TypeList<J>;
};

template <typename... J> struct JobTrait<SomeJob<J...>> {
  using JobList = typename SomeJob<J...>::JobList;
};
} // namespace dsl
} // namespace tf
