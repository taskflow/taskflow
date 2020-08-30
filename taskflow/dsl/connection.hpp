// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "job_trait.hpp"
#include "tuple_utils.hpp"
#include "type_list.hpp"

namespace tf {
namespace dsl {
template <typename F, typename T> class Connection {
  using FROMs = typename JobTrait<F>::JobList;
  using TOs = typename JobTrait<T>::JobList;

public:
  using FromJobList = Unique_t<Flatten_t<FROMs>>;
  using ToJobList = Unique_t<Flatten_t<TOs>>;
};

template <typename T, typename OUT = TypeList<>> struct Chain;

template <typename F, typename OUT> struct Chain<auto (*)(F)->void, OUT> {
  using From = F;
  using type = OUT;
};

template <typename F, typename T, typename OUT>
struct Chain<auto (*)(F)->T, OUT> {
private:
  using To = typename Chain<T, OUT>::From;

public:
  using From = F;
  using type = typename Chain<
      T, typename OUT::template appendTo<Connection<From, To>>>::type;
};

template <typename FROM, typename TO> struct OneToOneLink {
  template <typename JobsCB> struct InstanceType {
    template <typename J> struct IsJob {
      template <typename JobCb> struct apply {
        constexpr static bool value =
            std::is_same<typename JobCb::JobType, J>::value;
      };
    };

    constexpr void build(JobsCB &jobsCb) {
      constexpr size_t JobsCBSize = std::tuple_size<JobsCB>::value;
      constexpr size_t FromJobIndex =
          TupleElementByF_v<JobsCB, IsJob<FROM>::template apply>;
      constexpr size_t ToJobIndex =
          TupleElementByF_v<JobsCB, IsJob<TO>::template apply>;
      static_assert(FromJobIndex < JobsCBSize && ToJobIndex < JobsCBSize,
                    "fatal: not find JobCb in JobsCB");
      std::get<FromJobIndex>(jobsCb).job_.precede(
          std::get<ToJobIndex>(jobsCb).job_);
    }
  };
};
} // namespace dsl
}; // namespace tf
