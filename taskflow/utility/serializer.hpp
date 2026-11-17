// This file is part of Taskflow project <https://taskflow.github.io>
// Copyright (c) 2018-2023 Dr. Tsung-Wei Huang
// MIT License - see LICENSE file for details

#pragma once

#include <cassert>
#include <array>
#include <deque>
#include <forward_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include <concepts>    // C++20

namespace tf {

// ----------------------------------------------------------------------------
// Type detection traits (partial specializations — unchanged as per spec)
// ----------------------------------------------------------------------------

template <typename T>
struct is_std_basic_string : std::false_type {};
template <typename C, typename T, typename A>
struct is_std_basic_string<std::basic_string<C,T,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_basic_string_v = is_std_basic_string<T>::value;

template <typename T>
struct is_std_vector : std::false_type {};
template <typename T, typename A>
struct is_std_vector<std::vector<T,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename T>
struct is_std_deque : std::false_type {};
template <typename T, typename A>
struct is_std_deque<std::deque<T,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_deque_v = is_std_deque<T>::value;

template <typename T>
struct is_std_map : std::false_type {};
template <typename K, typename V, typename C, typename A>
struct is_std_map<std::map<K,V,C,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_map_v = is_std_map<T>::value;

template <typename T>
struct is_std_unordered_map : std::false_type {};
template <typename K, typename V, typename H, typename E, typename A>
struct is_std_unordered_map<std::unordered_map<K,V,H,E,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_unordered_map_v = is_std_unordered_map<T>::value;

template <typename T>
struct is_std_set : std::false_type {};
template <typename K, typename C, typename A>
struct is_std_set<std::set<K,C,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_set_v = is_std_set<T>::value;

template <typename T>
struct is_std_unordered_set : std::false_type {};
template <typename K, typename H, typename E, typename A>
struct is_std_unordered_set<std::unordered_set<K,H,E,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_unordered_set_v = is_std_unordered_set<T>::value;

template <typename T>
struct is_std_forward_list : std::false_type {};
template <typename T, typename A>
struct is_std_forward_list<std::forward_list<T,A>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_forward_list_v = is_std_forward_list<T>::value;

template <typename T>
struct is_std_variant : std::false_type {};
template <typename... Ts>
struct is_std_variant<std::variant<Ts...>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_variant_v = is_std_variant<T>::value;

template <typename T>
struct is_std_optional : std::false_type {};
template <typename T>
struct is_std_optional<std::optional<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_optional_v = is_std_optional<T>::value;

template <typename T>
struct is_std_tuple : std::false_type {};
template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

template <typename T>
struct is_std_array : std::false_type {};
template <typename T, std::size_t N>
struct is_std_array<std::array<T,N>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;

template <typename T>
struct is_std_duration : std::false_type {};
template <typename Rep, typename Period>
struct is_std_duration<std::chrono::duration<Rep,Period>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_duration_v = is_std_duration<T>::value;

template <typename T>
struct is_std_time_point : std::false_type {};
template <typename Clock, typename Duration>
struct is_std_time_point<std::chrono::time_point<Clock,Duration>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_time_point_v = is_std_time_point<T>::value;

template <typename T>
inline constexpr bool is_std_enum_v = std::is_enum_v<T>;

// is_default_serializable_v: compound trait — unchanged as per spec
template <typename T>
inline constexpr bool is_default_serializable_v =
    std::is_arithmetic_v<T>        ||
    is_std_basic_string_v<T>       ||
    is_std_vector_v<T>             ||
    is_std_deque_v<T>              ||
    is_std_map_v<T>                ||
    is_std_unordered_map_v<T>      ||
    is_std_set_v<T>                ||
    is_std_unordered_set_v<T>      ||
    is_std_forward_list_v<T>       ||
    is_std_variant_v<T>            ||
    is_std_optional_v<T>           ||
    is_std_tuple_v<T>              ||
    is_std_array_v<T>              ||
    is_std_duration_v<T>           ||
    is_std_time_point_v<T>         ||
    is_std_enum_v<T>;

// ----------------------------------------------------------------------------
// C++20 concepts derived from the _v traits above
// ----------------------------------------------------------------------------

/**
 * @brief concept for std::basic_string specializations
 */
template <typename T>
concept StdBasicString = is_std_basic_string_v<std::decay_t<T>>;

/**
 * @brief concept for std::vector specializations
 */
template <typename T>
concept StdVector = is_std_vector_v<std::decay_t<T>>;

/**
 * @brief concept for std::deque specializations
 */
template <typename T>
concept StdDeque = is_std_deque_v<std::decay_t<T>>;

/**
 * @brief concept for std::map specializations
 */
template <typename T>
concept StdMap = is_std_map_v<std::decay_t<T>>;

/**
 * @brief concept for std::unordered_map specializations
 */
template <typename T>
concept StdUnorderedMap = is_std_unordered_map_v<std::decay_t<T>>;

/**
 * @brief concept for std::set specializations
 */
template <typename T>
concept StdSet = is_std_set_v<std::decay_t<T>>;

/**
 * @brief concept for std::unordered_set specializations
 */
template <typename T>
concept StdUnorderedSet = is_std_unordered_set_v<std::decay_t<T>>;

/**
 * @brief concept for std::forward_list specializations
 */
template <typename T>
concept StdForwardList = is_std_forward_list_v<std::decay_t<T>>;

/**
 * @brief concept for std::variant specializations
 */
template <typename T>
concept StdVariant = is_std_variant_v<std::decay_t<T>>;

/**
 * @brief concept for std::optional specializations
 */
template <typename T>
concept StdOptional = is_std_optional_v<std::decay_t<T>>;

/**
 * @brief concept for std::tuple specializations
 */
template <typename T>
concept StdTuple = is_std_tuple_v<std::decay_t<T>>;

/**
 * @brief concept for std::array specializations
 */
template <typename T>
concept StdArray = is_std_array_v<std::decay_t<T>>;

/**
 * @brief concept for std::chrono::duration specializations
 */
template <typename T>
concept StdDuration = is_std_duration_v<std::decay_t<T>>;

/**
 * @brief concept for std::chrono::time_point specializations
 */
template <typename T>
concept StdTimePoint = is_std_time_point_v<std::decay_t<T>>;

/**
 * @brief concept for enum types
 */
template <typename T>
concept StdEnum = is_std_enum_v<std::decay_t<T>>;

// ----------------------------------------------------------------------------
// Helper: index sequence iteration over a tuple
// ----------------------------------------------------------------------------

namespace detail {

template <typename F, typename Tuple, std::size_t... Is>
void for_each_in_tuple(F&& f, Tuple&& t, std::index_sequence<Is...>) {
  (f(std::get<Is>(std::forward<Tuple>(t))), ...);
}

template <typename F, typename Tuple>
void for_each_in_tuple(F&& f, Tuple&& t) {
  for_each_in_tuple(
    std::forward<F>(f),
    std::forward<Tuple>(t),
    std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{}
  );
}

} // namespace detail

// ----------------------------------------------------------------------------
// Serializer
// ----------------------------------------------------------------------------

/**
 * @brief serializer class
 *
 * @tparam Device output device type (must support write())
 * @tparam SizeType integer type used to encode sizes
 */
template <typename Device, typename SizeType = std::streamsize>
class Serializer {

public:

  /**
   * @brief constructs the serializer with a reference to the output device
   */
  explicit Serializer(Device& device) : _device(device) {}

  /**
   * @brief serializes one or more values to the device
   *
   * @tparam Ts value types to serialize
   * @param args values to serialize
   * @return total number of bytes written
   */
  template <typename... Ts>
  SizeType operator()(Ts&&... args) {
    return (_save(std::forward<Ts>(args)) + ... + SizeType{0});
  }

private:

  Device& _device;

  // ------------------------------------------------------------------
  // Dispatch overloads — ordered from most- to least-specific so that
  // the requires clauses form a clean, non-overlapping partition.
  // The negated !is_default_serializable_v<U> guard stays as a
  // requires clause per project conventions.
  // ------------------------------------------------------------------

  // (1) Custom / user-defined serializable: T has a .save(ar) method
  //     and is NOT one of the standard types we handle ourselves.
  template <typename T>
  requires (!is_default_serializable_v<std::decay_t<T>>)
  SizeType _save(T&& t) {
    SizeType sz = 0;
    auto ar = [&](auto&&... args) {
      sz += (*this)(std::forward<decltype(args)>(args)...);
    };
    t.save(ar);
    return sz;
  }

  // (2) Arithmetic (int, float, double, bool, …)
  template <typename T>
  requires std::is_arithmetic_v<std::decay_t<T>>
  SizeType _save(T&& t) {
    _device.write(reinterpret_cast<const char*>(&t), sizeof(std::decay_t<T>));
    return sizeof(std::decay_t<T>);
  }

  // (3) std::basic_string
  template <StdBasicString T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    _device.write(t.data(), t.size() * sizeof(typename std::decay_t<T>::value_type));
    sz += static_cast<SizeType>(t.size() * sizeof(typename std::decay_t<T>::value_type));
    return sz;
  }

  // (4) std::vector — bulk write for arithmetic value_type, element-wise otherwise
  template <StdVector T>
  SizeType _save(T&& t) {
    using U = std::decay_t<T>;
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    if constexpr (std::is_arithmetic_v<typename U::value_type>) {
      _device.write(
        reinterpret_cast<const char*>(t.data()),
        t.size() * sizeof(typename U::value_type)
      );
      sz += static_cast<SizeType>(t.size() * sizeof(typename U::value_type));
    } else {
      for (auto&& item : t) {
        sz += _save(item);
      }
    }
    return sz;
  }

  // (5) std::deque
  template <StdDeque T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    for (auto&& item : t) {
      sz += _save(item);
    }
    return sz;
  }

  // (6) std::map
  template <StdMap T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    for (auto&& [k, v] : t) {
      sz += _save(k);
      sz += _save(v);
    }
    return sz;
  }

  // (7) std::unordered_map
  template <StdUnorderedMap T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    for (auto&& [k, v] : t) {
      sz += _save(k);
      sz += _save(v);
    }
    return sz;
  }

  // (8) std::set
  template <StdSet T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    for (auto&& item : t) {
      sz += _save(item);
    }
    return sz;
  }

  // (9) std::unordered_set
  template <StdUnorderedSet T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.size()));
    for (auto&& item : t) {
      sz += _save(item);
    }
    return sz;
  }

  // (10) std::forward_list — must count elements first (no size())
  template <StdForwardList T>
  SizeType _save(T&& t) {
    SizeType count = static_cast<SizeType>(std::distance(t.begin(), t.end()));
    SizeType sz = _save(count);
    for (auto&& item : t) {
      sz += _save(item);
    }
    return sz;
  }

  // (11) std::variant — save index, then the active alternative
  template <StdVariant T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<SizeType>(t.index()));
    sz += std::visit([&](auto&& v) { return _save(v); }, std::forward<T>(t));
    return sz;
  }

  // (12) std::optional — save has_value flag, then the value if present
  template <StdOptional T>
  SizeType _save(T&& t) {
    SizeType sz = _save(static_cast<bool>(t.has_value()));
    if (t.has_value()) {
      sz += _save(*t);
    }
    return sz;
  }

  // (13) std::tuple — fold over elements
  template <StdTuple T>
  SizeType _save(T&& t) {
    SizeType sz{0};
    detail::for_each_in_tuple([&](auto&& v) { sz += _save(v); }, std::forward<T>(t));
    return sz;
  }

  // (14) std::array — bulk write for arithmetic, element-wise otherwise
  template <StdArray T>
  SizeType _save(T&& t) {
    using U = std::decay_t<T>;
    SizeType sz{0};
    if constexpr (std::is_arithmetic_v<typename U::value_type>) {
      _device.write(
        reinterpret_cast<const char*>(t.data()),
        t.size() * sizeof(typename U::value_type)
      );
      sz = static_cast<SizeType>(t.size() * sizeof(typename U::value_type));
    } else {
      for (auto&& item : t) {
        sz += _save(item);
      }
    }
    return sz;
  }

  // (15) std::chrono::duration — serialize the underlying rep
  template <StdDuration T>
  SizeType _save(T&& t) {
    return _save(t.count());
  }

  // (16) std::chrono::time_point — serialize as duration since epoch
  template <StdTimePoint T>
  SizeType _save(T&& t) {
    return _save(t.time_since_epoch());
  }

  // (17) enum — serialize as underlying integer type
  template <StdEnum T>
  SizeType _save(T&& t) {
    return _save(static_cast<std::underlying_type_t<std::decay_t<T>>>(t));
  }
};

// ----------------------------------------------------------------------------
// Deserializer
// ----------------------------------------------------------------------------

/**
 * @brief deserializer class
 *
 * @tparam Device input device type (must support read())
 * @tparam SizeType integer type used to decode sizes (must match Serializer)
 */
template <typename Device, typename SizeType = std::streamsize>
class Deserializer {

public:

  /**
   * @brief constructs the deserializer with a reference to the input device
   */
  explicit Deserializer(Device& device) : _device(device) {}

  /**
   * @brief deserializes one or more values from the device
   *
   * @tparam Ts value types to deserialize
   * @param args values to populate
   * @return total number of bytes read
   */
  template <typename... Ts>
  SizeType operator()(Ts&&... args) {
    return (_load(std::forward<Ts>(args)) + ... + SizeType{0});
  }

private:

  Device& _device;

  // ------------------------------------------------------------------
  // Dispatch overloads — mirror of Serializer::_save
  // ------------------------------------------------------------------

  // (1) Custom / user-defined deserializable
  template <typename T>
  requires (!is_default_serializable_v<std::decay_t<T>>)
  SizeType _load(T&& t) {
    SizeType sz = 0;
    auto ar = [&](auto&&... args) {
      sz += (*this)(std::forward<decltype(args)>(args)...);
    };
    t.load(ar);
    return sz;
  }

  // (2) Arithmetic
  template <typename T>
  requires std::is_arithmetic_v<std::decay_t<T>>
  SizeType _load(T&& t) {
    _device.read(reinterpret_cast<char*>(&t), sizeof(std::decay_t<T>));
    return sizeof(std::decay_t<T>);
  }

  // (3) std::basic_string
  template <StdBasicString T>
  SizeType _load(T&& t) {
    SizeType n{0};
    SizeType sz = _load(n);
    t.resize(static_cast<std::size_t>(n));
    _device.read(
      reinterpret_cast<char*>(t.data()),
      static_cast<std::size_t>(n) * sizeof(typename std::decay_t<T>::value_type)
    );
    sz += static_cast<SizeType>(n * sizeof(typename std::decay_t<T>::value_type));
    return sz;
  }

  // (4) std::vector
  template <StdVector T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.resize(static_cast<std::size_t>(n));
    if constexpr (std::is_arithmetic_v<typename U::value_type>) {
      _device.read(
        reinterpret_cast<char*>(t.data()),
        static_cast<std::size_t>(n) * sizeof(typename U::value_type)
      );
      sz += static_cast<SizeType>(n * sizeof(typename U::value_type));
    } else {
      for (auto& item : t) {
        sz += _load(item);
      }
    }
    return sz;
  }

  // (5) std::deque
  template <StdDeque T>
  SizeType _load(T&& t) {
    SizeType n{0};
    SizeType sz = _load(n);
    t.resize(static_cast<std::size_t>(n));
    for (auto& item : t) {
      sz += _load(item);
    }
    return sz;
  }

  // (6) std::map
  template <StdMap T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.clear();
    for (SizeType i = 0; i < n; ++i) {
      typename U::key_type k{};
      typename U::mapped_type v{};
      sz += _load(k);
      sz += _load(v);
      t.emplace(std::move(k), std::move(v));
    }
    return sz;
  }

  // (7) std::unordered_map
  template <StdUnorderedMap T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.clear();
    t.reserve(static_cast<std::size_t>(n));
    for (SizeType i = 0; i < n; ++i) {
      typename U::key_type k{};
      typename U::mapped_type v{};
      sz += _load(k);
      sz += _load(v);
      t.emplace(std::move(k), std::move(v));
    }
    return sz;
  }

  // (8) std::set
  template <StdSet T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.clear();
    for (SizeType i = 0; i < n; ++i) {
      typename U::value_type v{};
      sz += _load(v);
      t.emplace(std::move(v));
    }
    return sz;
  }

  // (9) std::unordered_set
  template <StdUnorderedSet T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.clear();
    t.reserve(static_cast<std::size_t>(n));
    for (SizeType i = 0; i < n; ++i) {
      typename U::value_type v{};
      sz += _load(v);
      t.emplace(std::move(v));
    }
    return sz;
  }

  // (10) std::forward_list
  template <StdForwardList T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType n{0};
    SizeType sz = _load(n);
    t.clear();
    // Build in reverse then reverse — O(n) without needing push_back
    for (SizeType i = 0; i < n; ++i) {
      typename U::value_type v{};
      sz += _load(v);
      t.push_front(std::move(v));
    }
    t.reverse();
    return sz;
  }

  // (11) std::variant — load the index, then construct the right alternative
  template <StdVariant T>
  SizeType _load(T&& t) {
    SizeType idx{0};
    SizeType sz = _load(idx);
    sz += _load_variant_at(t, static_cast<std::size_t>(idx),
                           std::make_index_sequence<
                             std::variant_size_v<std::decay_t<T>>>{});
    return sz;
  }

  template <typename V, std::size_t... Is>
  SizeType _load_variant_at(V& v, std::size_t idx, std::index_sequence<Is...>) {
    SizeType sz{0};
    bool matched = false;
    ([&]() {
      if (!matched && idx == Is) {
        matched = true;
        std::variant_alternative_t<Is, std::decay_t<V>> alt{};
        sz = _load(alt);
        v = std::move(alt);
      }
    }(), ...);
    return sz;
  }

  // (12) std::optional
  template <StdOptional T>
  SizeType _load(T&& t) {
    bool has{false};
    SizeType sz = _load(has);
    if (has) {
      typename std::decay_t<T>::value_type v{};
      sz += _load(v);
      t = std::move(v);
    } else {
      t = std::nullopt;
    }
    return sz;
  }

  // (13) std::tuple
  template <StdTuple T>
  SizeType _load(T&& t) {
    SizeType sz{0};
    detail::for_each_in_tuple([&](auto& v) { sz += _load(v); }, std::forward<T>(t));
    return sz;
  }

  // (14) std::array
  template <StdArray T>
  SizeType _load(T&& t) {
    using U = std::decay_t<T>;
    SizeType sz{0};
    if constexpr (std::is_arithmetic_v<typename U::value_type>) {
      _device.read(
        reinterpret_cast<char*>(t.data()),
        t.size() * sizeof(typename U::value_type)
      );
      sz = static_cast<SizeType>(t.size() * sizeof(typename U::value_type));
    } else {
      for (auto& item : t) {
        sz += _load(item);
      }
    }
    return sz;
  }

  // (15) std::chrono::duration
  template <StdDuration T>
  SizeType _load(T&& t) {
    typename std::decay_t<T>::rep rep{};
    SizeType sz = _load(rep);
    t = std::decay_t<T>{rep};
    return sz;
  }

  // (16) std::chrono::time_point
  template <StdTimePoint T>
  SizeType _load(T&& t) {
    typename std::decay_t<T>::duration dur{};
    SizeType sz = _load(dur);
    t = std::decay_t<T>{dur};
    return sz;
  }

  // (17) enum
  template <StdEnum T>
  SizeType _load(T&& t) {
    std::underlying_type_t<std::decay_t<T>> val{};
    SizeType sz = _load(val);
    t = static_cast<std::decay_t<T>>(val);
    return sz;
  }
};

} // namespace tf
