// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include <cstddef>

namespace tf {
namespace dsl {
template <typename...> using void_t = void;

template <typename... Ts> struct TypeList {
  using type = TypeList<Ts...>;
  static constexpr size_t size = 0;

  template <typename... T> struct append { using type = TypeList<T...>; };
  template <typename... T> using appendTo = typename append<T...>::type;

  template <typename T> using prepend = typename TypeList<T>::type;

  template <template <typename...> class T> using exportTo = T<Ts...>;
};

template <typename Head, typename... Tails> struct TypeList<Head, Tails...> {
  using type = TypeList<Head, Tails...>;
  using head = Head;
  using tails = TypeList<Tails...>;
  static constexpr size_t size = sizeof...(Tails) + 1;

  template <typename... Ts> struct append {
    using type = TypeList<Head, Tails..., Ts...>;
  };
  template <typename... Ts> using appendTo = typename append<Ts...>::type;

  template <typename T>
  using prepend = typename TypeList<T, Head, Tails...>::type;

  template <template <typename...> class T> using exportTo = T<Head, Tails...>;
};

template <typename IN> struct IsTypeList {
  constexpr static bool value = false;
};

template <typename IN> constexpr bool IsTypeList_v = IsTypeList<IN>::value;

template <typename... Ts> struct IsTypeList<TypeList<Ts...>> {
  constexpr static bool value = true;
};

template <typename... IN> struct Concat;

template <typename... IN> using Concat_t = typename Concat<IN...>::type;

template <> struct Concat<> { using type = TypeList<>; };
template <typename IN> struct Concat<IN> { using type = IN; };

template <typename IN, typename IN2> struct Concat<IN, IN2> {
  using type = typename IN2::template exportTo<IN::template append>::type;
};

template <typename IN, typename IN2, typename... Rest>
struct Concat<IN, IN2, Rest...> {
  using type = Concat_t<Concat_t<IN, IN2>, Rest...>;
};

template <typename IN, typename OUT = TypeList<>, typename = void>
struct Flatten {
  using type = OUT;
};

template <typename IN> using Flatten_t = typename Flatten<IN>::type;

template <typename IN, typename OUT>
struct Flatten<IN, OUT, std::enable_if_t<IsTypeList_v<typename IN::head>>> {
  using type =
      typename Flatten<typename IN::tails,
                       Concat_t<OUT, Flatten_t<typename IN::head>>>::type;
};

template <typename IN, typename OUT>
struct Flatten<IN, OUT, std::enable_if_t<!IsTypeList_v<typename IN::head>>> {
  using type = typename Flatten<
      typename IN::tails,
      typename OUT::template appendTo<typename IN::head>>::type;
};

template <typename IN, template <typename> class F> struct Map {
  using type = TypeList<>;
};

template <typename IN, template <typename> class F>
using Map_t = typename Map<IN, F>::type;

template <template <typename> class F, typename... Ts>
struct Map<TypeList<Ts...>, F> {
  using type = TypeList<typename F<Ts>::type...>;
};

template <typename IN, template <typename> class F, typename OUT = TypeList<>,
          typename = void>
struct Filter {
  using type = OUT;
};

template <typename IN, template <typename> class F>
using Filter_t = typename Filter<IN, F>::type;

template <typename IN, template <typename> class F, typename OUT>
class Filter<IN, F, OUT, void_t<typename IN::head>> {
  using H = typename IN::head;

public:
  using type = typename std::conditional_t<
      F<H>::value,
      Filter<typename IN::tails, F, typename OUT::template appendTo<H>>,
      Filter<typename IN::tails, F, OUT>>::type;
};

template <typename IN, typename = void> struct Unique { using type = IN; };

template <typename IN> using Unique_t = typename Unique<IN>::type;

template <typename IN> class Unique<IN, void_t<typename IN::head>> {
  template <typename T> struct IsDifferR {
    template <typename R> struct apply {
      static constexpr bool value = !std::is_same<T, R>::value;
    };
  };

  using tails = Unique_t<typename IN::tails>;
  using eraseHead =
      Filter_t<tails, IsDifferR<typename IN::head>::template apply>;

public:
  using type = typename eraseHead::template prepend<typename IN::head>;
};
} // namespace dsl
} // namespace tf
