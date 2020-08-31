// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include "connection.hpp"
#include "type_list.hpp"
#include <type_traits>

namespace tf {
namespace dsl {
template <typename... Links> class TaskAnalyzer {
  template <typename FROMs, typename TOs, typename = void>
  struct BuildOneToOneLink;

  template <typename... Fs, typename Ts>
  struct BuildOneToOneLink<TypeList<Fs...>, Ts> {
    using type = Concat_t<typename BuildOneToOneLink<Fs, Ts>::type...>;
  };

  template <typename F, typename... Ts>
  struct BuildOneToOneLink<F, TypeList<Ts...>,
                           std::enable_if_t<!IsTypeList_v<F>>> {
    using type = TypeList<OneToOneLink<F, Ts>...>;
  };

  template <typename Link> class OneToOneLinkSetF {
    using FromTaskList = typename Link::FromTaskList;
    using ToTaskList = typename Link::ToTaskList;

  public:
    using type = typename BuildOneToOneLink<FromTaskList, ToTaskList>::type;
  };

public:
  using AllTasks = Unique_t<
      Concat_t<typename Links::FromTaskList..., typename Links::ToTaskList...>>;
  using OneToOneLinkSet =
      Unique_t<Flatten_t<Map_t<TypeList<Links...>, OneToOneLinkSetF>>>;
};

} // namespace dsl
} // namespace tf
