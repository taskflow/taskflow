#ifndef FF_TYPETRAITS_H
#define FF_TYPETRAITS_H

#include <type_traits>

namespace ff{

namespace traits {

template <class T, template <class...> class Test>
struct exists{
    template<class U>
    static std::true_type check(Test<U>*);

    template<class U>
    static std::false_type check(...);

    static constexpr bool value = decltype(check<T>(0))::value;
};

template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(serialize<std::pair<char*, size_t>>(std::declval<std::pair<char*, size_t>&>(), std::declval<U*>()))>>>
struct user_serialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(deserialize<std::pair<char*, size_t>>(std::declval<const std::pair<char*, size_t>&>(), std::declval<U*>()))>>>
struct user_deserialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(serializefreetask<char>(std::declval<char*>(), std::declval<U*>()))>>>
struct user_freetask_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(deserializealloctask<std::pair<char*, size_t>>(std::declval<const std::pair<char*, size_t>&>(), std::declval<U*&>()))>>>
struct user_alloctask_test{};

	
template<class U, class = std::enable_if_t<std::is_same_v<std::pair<char*,size_t>, decltype(serializeWrapper<U>(std::declval<U*>(), std::declval<bool&>()))>>>
struct serialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(deserializeWrapper<U>(std::declval<char*>(), std::declval<size_t&>(), std::declval<U*>()))>>>
struct deserialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(freetaskWrapper<U>(std::declval<U*>()))>>>
struct freetask_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(alloctaskWrapper<U>(std::declval<char*>(), std::declval<size_t&>(), std::declval<U*&>()))>>>
struct alloctask_test{};

   
	

/*
    High level traits to use
*/

template<class T>
using is_serializable = exists<T, serialize_test>;

//helper 
template<class T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;


template<class T>
using is_deserializable = exists<T, deserialize_test>;

// helper
template<class T>
inline constexpr bool is_deserializable_v = is_deserializable<T>::value;

template<class T>
using has_freetask = exists<T, freetask_test>;

template<class T>
inline constexpr bool has_freetask_v = has_freetask<T>::value;

template<class T>
using has_alloctask = exists<T, alloctask_test>;

template<class T>
inline constexpr bool has_alloctask_v = has_alloctask<T>::value;

	
}


/*
    Wrapper to user defined serialize and de-serialize functions, in order to exploits user defined functions in other translation units. 
*/
template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_serialize_test>::value>>
std::pair<char*,size_t> serializeWrapper(T*in, bool& datacopied){
    std::pair<char*,size_t> p;
    datacopied = serialize<std::pair<char*, size_t>>(p, in);
    return p;
}

template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_deserialize_test>::value>>
bool deserializeWrapper(char* c, size_t s, T* obj){
    return deserialize<std::pair<char*, size_t>>(std::make_pair(c, s),obj);
}

template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_freetask_test>::value>>
void freetaskWrapper(T*in){
    serializefreetask<char>((char*)in, in);
}

template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_alloctask_test>::value>>
void alloctaskWrapper(char* c, size_t s, T*& p){
    deserializealloctask<std::pair<char*, size_t>>(std::make_pair(c,s), p);
}


}
#endif
