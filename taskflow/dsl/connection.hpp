// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "type_list.hpp"
#include "tuple_utils.hpp"
#include "job_trait.hpp"

namespace tf {
template<typename T>
class Connection;

template<typename F, typename T>
class Connection<auto(F) -> T> {
    using FROMs = typename JobTrait<F>::JobList;
    using TOs = typename JobTrait<T>::JobList;
public:
    using FromJobList = Unique_t<Flatten_t<FROMs>>;
    using ToJobList = Unique_t<Flatten_t<TOs>>;
};

template<typename FROM, typename TO>
struct OneToOneLink {
    template<typename JobsCB>
    struct InstanceType {
        template<typename J>
        struct IsJob {
            template<typename JobCb> struct apply
            { constexpr static bool value = std::is_same<typename JobCb::JobType, J>::value; };
        };

        constexpr void build(JobsCB& jobsCb) {
            constexpr size_t JobsCBSize = std::tuple_size<JobsCB>::value;
            constexpr size_t FromJobIndex = TupleElementByF_v<JobsCB, IsJob<FROM>::template apply>;
            constexpr size_t ToJobIndex = TupleElementByF_v<JobsCB, IsJob<TO>::template apply>;
            static_assert(FromJobIndex < JobsCBSize && ToJobIndex < JobsCBSize, "fatal: not find JobCb in JobsCB");
            std::get<FromJobIndex>(jobsCb).job_.precede(std::get<ToJobIndex>(jobsCb).job_);
        }
    };
};
};
