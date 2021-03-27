#pragma once

#include "traits.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// Supported C++ STL type
// ----------------------------------------------------------------------------

// std::basic_string
template <typename T> 
struct is_std_basic_string : std::false_type {};

template <typename... ArgsT> 
struct is_std_basic_string <std::basic_string<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_basic_string_v = is_std_basic_string<T>::value;

// std::array
template <typename T> 
struct is_std_array : std::false_type {};

template <typename T, size_t N> 
struct is_std_array <std::array<T, N>> : std::true_type {};

template <typename T> 
constexpr bool is_std_array_v = is_std_array<T>::value;

// std::vector
template <typename T> 
struct is_std_vector : std::false_type {};

template <typename... ArgsT> 
struct is_std_vector <std::vector<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_vector_v = is_std_vector<T>::value;

// std::deque
template <typename T> 
struct is_std_deque : std::false_type {};

template <typename... ArgsT> 
struct is_std_deque <std::deque<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_deque_v = is_std_deque<T>::value;

// std::list
template <typename T> 
struct is_std_list : std::false_type {};

template <typename... ArgsT> 
struct is_std_list <std::list<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_list_v = is_std_list<T>::value;

// std::forward_list
template <typename T> 
struct is_std_forward_list : std::false_type {};

template <typename... ArgsT> 
struct is_std_forward_list <std::forward_list<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_forward_list_v = is_std_forward_list<T>::value;

// std::map
template <typename T> 
struct is_std_map : std::false_type {};

template <typename... ArgsT> 
struct is_std_map <std::map<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_map_v = is_std_map<T>::value;

// std::unordered_map
template <typename T> 
struct is_std_unordered_map : std::false_type {};

template <typename... ArgsT> 
struct is_std_unordered_map <std::unordered_map<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_unordered_map_v = is_std_unordered_map<T>::value;

// std::set
template <typename T> 
struct is_std_set : std::false_type {};

template <typename... ArgsT> 
struct is_std_set <std::set<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_set_v = is_std_set<T>::value;

// std::unordered_set
template <typename T> 
struct is_std_unordered_set : std::false_type {};

template <typename... ArgsT> 
struct is_std_unordered_set <std::unordered_set<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_unordered_set_v = is_std_unordered_set<T>::value;

// std::variant
template <typename T> 
struct is_std_variant : std::false_type {};

template <typename... ArgsT> 
struct is_std_variant <std::variant<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_variant_v = is_std_variant<T>::value;

// std::optional
template <typename T> 
struct is_std_optional : std::false_type {};

template <typename... ArgsT> 
struct is_std_optional <std::optional<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_optional_v = is_std_optional<T>::value;

// std::unique_ptr
template <typename T> 
struct is_std_unique_ptr : std::false_type {};

template <typename... ArgsT> 
struct is_std_unique_ptr <std::unique_ptr<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_unique_ptr_v = is_std_unique_ptr<T>::value;

// std::shared_ptr
template <typename T> 
struct is_std_shared_ptr : std::false_type {};

template <typename... ArgsT> 
struct is_std_shared_ptr <std::shared_ptr<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_shared_ptr_v = is_std_shared_ptr<T>::value;

// std::duration
template <typename T> struct is_std_duration : std::false_type {};

template <typename... ArgsT> 
struct is_std_duration<std::chrono::duration<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_duration_v = is_std_duration<T>::value;

// std::time_point
template <typename T> 
struct is_std_time_point : std::false_type {};

template <typename... ArgsT> 
struct is_std_time_point<std::chrono::time_point<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_time_point_v = is_std_time_point<T>::value;

// std::tuple
template <typename T> 
struct is_std_tuple : std::false_type {};

template <typename... ArgsT> 
struct is_std_tuple<std::tuple<ArgsT...>> : std::true_type {};

template <typename T> 
constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

//-----------------------------------------------------------------------------
// Type extraction.
//-----------------------------------------------------------------------------

// ExtractType: forward declaration
template <size_t, typename> 
struct ExtractType;

// ExtractType_t: alias interface
template <size_t idx, typename C>
using ExtractType_t = typename ExtractType<idx, C>::type;

// ExtractType: base
template <template <typename...> typename C, typename T, typename... RestT>
struct ExtractType <0, C<T, RestT...>> {
  using type = T;
};

// ExtractType: base
template <typename T>
struct ExtractType <0, T> {
  using type = T;
};

// ExtractType: recursive definition.
template <size_t idx, template <typename...> typename C, typename T, typename... RestT>
struct ExtractType <idx, C<T, RestT...>> : ExtractType<idx-1, C<RestT...>> {
};

// ----------------------------------------------------------------------------
// Size Wrapper
// ----------------------------------------------------------------------------

// Struct: SizeTag
// Class that wraps a given size item which can be customized. 
template <typename T>
class SizeTag {

  public: 
  
    using type = std::conditional_t<std::is_lvalue_reference_v<T>, T, std::decay_t<T>>;
    
    SizeTag(T&& item) : _item(std::forward<T>(item)) {}
    
    SizeTag& operator = (const SizeTag&) = delete;

    inline const T& get() const {return _item;}

    template <typename ArchiverT>
    auto save(ArchiverT & ar) const { return ar(_item); }
    
    template <typename ArchiverT>
    auto load(ArchiverT & ar) { return ar(_item); }

  private:

    type _item;
};

// Function: make_size_tag
template <typename T>
SizeTag<T> make_size_tag(T&& t) {
  return { std::forward<T>(t) };
}

// ----------------------------------------------------------------------------
// Size Wrapper
// ----------------------------------------------------------------------------

// Class: MapItem
template <typename KeyT, typename ValueT>
class MapItem {
  
  public:
  
    using KeyType = std::conditional_t <std::is_lvalue_reference_v<KeyT>, KeyT, std::decay_t<KeyT>>;
    using ValueType = std::conditional_t <std::is_lvalue_reference_v<ValueT>, ValueT, std::decay_t<ValueT>>;

    MapItem(KeyT&& k, ValueT&& v) : _key(std::forward<KeyT>(k)), _value(std::forward<ValueT>(v)) {}
    MapItem& operator = (const MapItem&) = delete;

    inline const KeyT& key() const { return _key; }
    inline const ValueT& value() const { return _value; }

    template <typename ArchiverT>
    auto save(ArchiverT & ar) const { return ar(_key, _value); }
    
    template <typename ArchiverT>
    auto load(ArchiverT & ar) { return ar(_key, _value); }

  private:

    KeyType _key;
    ValueType _value;
};

// Function: make_kv_pair
template <typename KeyT, typename ValueT>
MapItem<KeyT, ValueT> make_kv_pair(KeyT&& k, ValueT&& v) {
  return { std::forward<KeyT>(k), std::forward<ValueT>(v) };
}

// ----------------------------------------------------------------------------
// Serializer Definition
// ----------------------------------------------------------------------------

template <typename T>
constexpr auto is_default_serializable_v = 
  std::is_arithmetic_v<T>    ||
  std::is_enum_v<T>          ||
  is_std_basic_string_v<T>   ||
  is_std_vector_v<T>         ||
  is_std_deque_v<T>          ||
  is_std_list_v<T>           ||
  is_std_forward_list_v<T>   ||
  is_std_map_v<T>            ||
  is_std_unordered_map_v<T>  ||
  is_std_set_v<T>            ||
  is_std_unordered_set_v<T>  ||
  is_std_duration_v<T>       ||
  is_std_time_point_v<T>     ||
  is_std_variant_v<T>        ||
  is_std_optional_v<T>       ||
  is_std_tuple_v<T>          ||
  is_std_array_v<T>;


// Class: Serializer
template <typename Device = std::ostream, typename SizeType = std::streamsize>
class Serializer {

  public:
    
    Serializer(Device& device);
    
    template <typename... T>
    SizeType operator()(T&&... items);
  
  private:

    Device& _device;
    
    template <typename T, 
      std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_basic_string_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_vector_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<
        is_std_deque_v<std::decay_t<T>> ||
        is_std_list_v<std::decay_t<T>>, 
        void
      >* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<
        is_std_forward_list_v<std::decay_t<T>>, 
        void
      >* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<
        is_std_map_v<std::decay_t<T>> ||
        is_std_unordered_map_v<std::decay_t<T>>, 
        void
      >* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<
        is_std_set_v<std::decay_t<T>> ||
        is_std_unordered_set_v<std::decay_t<T>>, 
        void
      >* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<std::is_enum_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);

    template <typename T, 
      std::enable_if_t<is_std_duration_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);

    template <typename T, 
      std::enable_if_t<is_std_time_point_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);

    template <typename T, 
      std::enable_if_t<is_std_optional_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_variant_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_tuple_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_array_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
    
    template <typename T, 
      std::enable_if_t<!is_default_serializable_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _save(T&&);
};

// Constructor
template <typename Device, typename SizeType>
Serializer<Device, SizeType>::Serializer(Device& device) : _device(device) {
}

// Operator ()
template <typename Device, typename SizeType>
template <typename... T>
SizeType Serializer<Device, SizeType>::operator() (T&&... items) {
  return (_save(std::forward<T>(items)) + ...);
}

// arithmetic data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  _device.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof(t));
  return sizeof(t);
}

// std::basic_string
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_basic_string_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  using U = std::decay_t<T>;
  auto sz = _save(make_size_tag(t.size()));
  _device.write(
    reinterpret_cast<const char*>(t.data()), 
    t.size()*sizeof(typename U::value_type)
  );
  return sz + t.size()*sizeof(typename U::value_type);
}

// std::vector
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_vector_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {

  using U = std::decay_t<T>;
    
  auto sz = _save(make_size_tag(t.size()));

  if constexpr (std::is_arithmetic_v<typename U::value_type>) {
    _device.write(
      reinterpret_cast<const char*>(t.data()), 
      t.size() * sizeof(typename U::value_type)
    );
    sz += t.size() * sizeof(typename U::value_type);
  } else {
    for(auto&& item : t) {
      sz += _save(item);
    }
  }

  return sz;
}

// std::list and std::deque
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_deque_v<std::decay_t<T>> ||
                   is_std_list_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  auto sz = _save(make_size_tag(t.size()));
  for(auto&& item : t) {
    sz += _save(item);
  }
  return sz;
}

// std::forward_list
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_forward_list_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  auto sz = _save(make_size_tag(std::distance(t.begin(), t.end())));
  for(auto&& item : t) {
    sz += _save(item);
  }
  return sz;
}

// std::map and std::unordered_map
template <typename Device, typename SizeType>  
template <typename T, std::enable_if_t<
  is_std_map_v<std::decay_t<T>> ||
  is_std_unordered_map_v<std::decay_t<T>>, 
  void
>*>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  auto sz = _save(make_size_tag(t.size()));
  for(auto&& [k, v] : t) {
    sz += _save(make_kv_pair(k, v));
  }
  return sz;
}

// std::set and std::unordered_set
template <typename Device, typename SizeType>  
template <typename T, std::enable_if_t<
  is_std_set_v<std::decay_t<T>> ||
  is_std_unordered_set_v<std::decay_t<T>>, 
  void
>*>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  auto sz = _save(make_size_tag(t.size()));
  for(auto&& item : t) {
    sz += _save(item);
  }
  return sz;
}

// enum data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<std::is_enum_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  using U = std::decay_t<T>;
  return _save(static_cast<std::underlying_type_t<U>>(t));
}

// duration data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_duration_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  return _save(t.count());
}

// time point data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_time_point_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  return _save(t.time_since_epoch());
}

// optional data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_optional_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  if(bool flag = t.has_value(); flag) {
    return _save(flag) + _save(*t);
  }
  else {
    return _save(flag);
  }
}

// variant type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_variant_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  return _save(t.index()) + 
         std::visit([&] (auto&& arg){ return _save(arg);}, t);
}

// tuple type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_tuple_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  return std::apply(
    [&] (auto&&... args) {
      return (_save(std::forward<decltype(args)>(args)) + ... + 0); 
    },
    std::forward<T>(t)
  );
}

// array
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_array_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {

  using U = std::decay_t<T>;

  static_assert(std::tuple_size<U>::value > 0, "Array size can't be zero");

  SizeType sz;

  if constexpr(std::is_arithmetic_v<typename U::value_type>) {
    _device.write(reinterpret_cast<const char*>(t.data()), sizeof(t));
    sz = sizeof(t);
  } 
  else {
    sz = 0;
    for(auto&& item : t) {
      sz += _save(item);
    }
  }

  return sz;
}

// custom save method    
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<!is_default_serializable_v<std::decay_t<T>>, void>*
>
SizeType Serializer<Device, SizeType>::_save(T&& t) {
  return t.save(*this);
}

// ----------------------------------------------------------------------------
// DeSerializer Definition
// ----------------------------------------------------------------------------

template <typename T>
constexpr auto is_default_deserializable_v = 
  std::is_arithmetic_v<T>    ||
  std::is_enum_v<T>          ||
  is_std_basic_string_v<T>   ||
  is_std_vector_v<T>         ||
  is_std_deque_v<T>          ||
  is_std_list_v<T>           ||
  is_std_forward_list_v<T>   ||
  is_std_map_v<T>            ||
  is_std_unordered_map_v<T>  ||
  is_std_set_v<T>            ||
  is_std_unordered_set_v<T>  ||
  is_std_duration_v<T>       ||
  is_std_time_point_v<T>     ||
  is_std_variant_v<T>        ||
  is_std_optional_v<T>       ||
  is_std_tuple_v<T>          ||
  is_std_array_v<T>;

// Class: Deserializer
template <typename Device = std::istream, typename SizeType = std::streamsize>
class Deserializer {

  public:
    
    Deserializer(Device& device);
    
    template <typename... T>
    SizeType operator()(T&&... items);
  
  private:

    Device& _device;
    
    // Function: _variant_helper
    template <
      size_t I = 0, typename... ArgsT, 
      std::enable_if_t<I==sizeof...(ArgsT)>* = nullptr
    >
    SizeType _variant_helper(size_t, std::variant<ArgsT...>&);
    
    // Function: _variant_helper
    template <
      size_t I = 0, typename... ArgsT, 
      std::enable_if_t<I<sizeof...(ArgsT)>* = nullptr
    >
    SizeType _variant_helper(size_t, std::variant<ArgsT...>&);
    
    template <typename T, 
      std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_basic_string_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_vector_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<
        is_std_deque_v<std::decay_t<T>> ||
        is_std_list_v<std::decay_t<T>>  ||
        is_std_forward_list_v<std::decay_t<T>>, 
        void
      >* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_map_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_unordered_map_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_set_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_unordered_set_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<std::is_enum_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);

    template <typename T, 
      std::enable_if_t<is_std_duration_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);

    template <typename T, 
      std::enable_if_t<is_std_time_point_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);

    template <typename T, 
      std::enable_if_t<is_std_optional_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_variant_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_tuple_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<is_std_array_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
    
    template <typename T, 
      std::enable_if_t<!is_default_deserializable_v<std::decay_t<T>>, void>* = nullptr
    >
    SizeType _load(T&&);
};

// Constructor
template <typename Device, typename SizeType>
Deserializer<Device, SizeType>::Deserializer(Device& device) : _device(device) {
}

// Operator ()
template <typename Device, typename SizeType>
template <typename... T>
SizeType Deserializer<Device, SizeType>::operator() (T&&... items) {
  return (_load(std::forward<T>(items)) + ...);
}

// Function: _variant_helper
template <typename Device, typename SizeType>
template <size_t I, typename... ArgsT, std::enable_if_t<I==sizeof...(ArgsT)>*>
SizeType Deserializer<Device, SizeType>::_variant_helper(size_t, std::variant<ArgsT...>&) {
  return 0;
}

// Function: _variant_helper
template <typename Device, typename SizeType>
template <size_t I, typename... ArgsT, std::enable_if_t<I<sizeof...(ArgsT)>*>
SizeType Deserializer<Device, SizeType>::_variant_helper(size_t i, std::variant<ArgsT...>& v) {
  if(i == 0) {
    using type = ExtractType_t<I, std::variant<ArgsT...>>;
    if(v.index() != I) {
      static_assert(
        std::is_default_constructible<type>::value, 
        "Failed to archive variant (type should be default constructible T())"
      );
      v = type();
    }
    return _load(std::get<type>(v));
  }
  return _variant_helper<I+1, ArgsT...>(i-1, v);
}

// arithmetic data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  _device.read(reinterpret_cast<char*>(std::addressof(t)), sizeof(t));
  return sizeof(t);
}

// std::basic_string
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_basic_string_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
  typename U::size_type num_chars;
  auto sz = _load(make_size_tag(num_chars));
  t.resize(num_chars);
  _device.read(reinterpret_cast<char*>(t.data()), num_chars*sizeof(typename U::value_type));
  return sz + num_chars*sizeof(typename U::value_type);
}

// std::vector
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_vector_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {

  using U = std::decay_t<T>;
  
  typename U::size_type num_data;
    
  auto sz = _load(make_size_tag(num_data));

  if constexpr(std::is_arithmetic_v<typename U::value_type>) {
    t.resize(num_data);
    _device.read(reinterpret_cast<char*>(t.data()), num_data * sizeof(typename U::value_type));
    sz += num_data * sizeof(typename U::value_type);
  } 
  else {
    t.resize(num_data);
    for(auto && v : t) {
      sz += _load(v);
    }
  }
  return sz;
}

// std::list and std::deque
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_deque_v<std::decay_t<T>> ||
                   is_std_list_v<std::decay_t<T>>  ||
                   is_std_forward_list_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
    
  typename U::size_type num_data;
  auto sz = _load(make_size_tag(num_data));

  t.resize(num_data);
  for(auto && v : t) {
    sz += _load(v);
  }
  return sz;
}

// std::map 
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_map_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  
  using U = std::decay_t<T>;

  typename U::size_type num_data;
  auto sz = _load(make_size_tag(num_data));
  
  t.clear();
  auto hint = t.begin();
    
  typename U::key_type k;
  typename U::mapped_type v;

  for(size_t i=0; i<num_data; ++i) {
    sz += _load(make_kv_pair(k, v));
    hint = t.emplace_hint(hint, std::move(k), std::move(v));
  }
  return sz;
}

// std::unordered_map
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_unordered_map_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
  typename U::size_type num_data;
  auto sz = _load(make_size_tag(num_data));

  t.clear();
  t.reserve(num_data);

  typename U::key_type k;
  typename U::mapped_type v;

  for(size_t i=0; i<num_data; ++i) {
    sz += _load(make_kv_pair(k, v));
    t.emplace(std::move(k), std::move(v));
  }
  
  return sz;
}

// std::set 
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_set_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  
  using U = std::decay_t<T>;

  typename U::size_type num_data;
  auto sz = _load(make_size_tag(num_data));

  t.clear();
  auto hint = t.begin();
    
  typename U::key_type k;

  for(size_t i=0; i<num_data; ++i) {   
    sz += _load(k);
    hint = t.emplace_hint(hint, std::move(k));
  }   
  return sz;
}

// std::unordered_set
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_unordered_set_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
   
  using U = std::decay_t<T>;
   
  typename U::size_type num_data;
  auto sz = _load(make_size_tag(num_data));

  t.clear();
  t.reserve(num_data);
    
  typename U::key_type k;

  for(size_t i=0; i<num_data; ++i) {   
    sz += _load(k);
    t.emplace(std::move(k));
  }   
  return sz;
}

// enum data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<std::is_enum_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
  std::underlying_type_t<U> k;
  auto sz = _load(k);
  t = static_cast<U>(k);
  return sz;
}

// duration data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_duration_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
  typename U::rep count;
  auto s = _load(count);
  t = U{count};
  return s;
}

// time point data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_time_point_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  using U = std::decay_t<T>;
  typename U::duration elapsed;
  auto s = _load(elapsed);
  t = U{elapsed};
  return s;
}

// optional data type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_optional_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  
  using U = std::decay_t<T>;

  bool has_value;
  auto s = _load(has_value);
  if(has_value) {
    if(!t) {
      t = typename U::value_type();
    }
    s += _load(*t);
  }
  else {
    t.reset(); 
  }
  return s;
}

// variant type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_variant_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  std::decay_t<decltype(t.index())> idx;
  auto s = _load(idx);
  return s + _variant_helper(idx, t);
}

// tuple type
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_tuple_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  return std::apply(
    [&] (auto&&... args) {
      return (_load(std::forward<decltype(args)>(args)) + ... + 0); 
    },
    std::forward<T>(t)
  );
}

// array
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<is_std_array_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {

  using U = std::decay_t<T>;

  static_assert(std::tuple_size<U>::value > 0, "Array size can't be zero");

  SizeType sz;
    
  if constexpr(std::is_arithmetic_v<typename U::value_type>) {
    _device.read(reinterpret_cast<char*>(t.data()), sizeof(t));
    sz = sizeof(t);
  } 
  else {
    sz = 0;
    for(auto && v : t) {
      sz += _load(v);
    }
  }

  return sz;
}

// custom save method    
template <typename Device, typename SizeType>  
template <typename T, 
  std::enable_if_t<!is_default_deserializable_v<std::decay_t<T>>, void>*
>
SizeType Deserializer<Device, SizeType>::_load(T&& t) {
  return t.load(*this);
}

}  // ned of namespace tf -----------------------------------------------------






