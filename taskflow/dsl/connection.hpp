// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "../core/flow_builder.hpp"
#include "task_trait.hpp"
#include "tuple_utils.hpp"
#include "type_list.hpp"

namespace tf {
namespace dsl {
template <typename F, typename T> class Connection {
  using FROMs = typename TaskTrait<F>::TaskList;
  using TOs = typename TaskTrait<T>::TaskList;

public:
  using FromTaskList = Unique_t<Flatten_t<FROMs>>;
  using ToTaskList = Unique_t<Flatten_t<TOs>>;
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
  template <typename TasksCB> struct InstanceType {
    constexpr void build(TasksCB &tasksCb) {
      constexpr size_t TasksCBSize = std::tuple_size<TasksCB>::value;
      constexpr size_t FromTaskIndex =
          TupleElementByF_v<TasksCB, IsTask<FROM>::template apply>;
      constexpr size_t ToTaskIndex =
          TupleElementByF_v<TasksCB, IsTask<TO>::template apply>;
      static_assert(FromTaskIndex < TasksCBSize && ToTaskIndex < TasksCBSize,
                    "fatal: not find TaskCb in TasksCB");
      std::get<FromTaskIndex>(tasksCb).task_.precede(
          std::get<ToTaskIndex>(tasksCb).task_);
    }
  };
};
} // namespace dsl
}; // namespace tf
