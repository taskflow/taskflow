/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __TBB_iterators_H
#define __TBB_iterators_H

#include <iterator>

#include "tbb_config.h"
#include "tbb_stddef.h"

#if __TBB_CPP11_DECLTYPE_PRESENT

namespace tbb {

template <typename IntType>
class counting_iterator {
public:
    typedef decltype(IntType()-IntType()) difference_type;
    typedef IntType value_type;
    typedef const IntType* pointer;
    typedef const IntType& reference;
    typedef std::random_access_iterator_tag iterator_category;

    explicit counting_iterator(IntType init): my_counter(init) { __TBB_STATIC_ASSERT(std::numeric_limits<IntType>::is_integer, "Integer required."); }

    reference operator*() const { return my_counter; }
    value_type operator[](difference_type i) const { return *(*this + i); }

    difference_type operator-(const counting_iterator& it) const { return my_counter - it.my_counter; }

    counting_iterator& operator+=(difference_type forward) { my_counter += forward; return *this; }
    counting_iterator& operator-=(difference_type backward) { return *this += -backward; }
    counting_iterator& operator++() { return *this += 1; }
    counting_iterator& operator--() { return *this -= 1; }

    counting_iterator operator++(int) {
        counting_iterator it(*this);
        ++(*this);
        return it;
    }
    counting_iterator operator--(int) {
        counting_iterator it(*this);
        --(*this);
        return it;
    }

    counting_iterator operator-(difference_type backward) const { return counting_iterator(my_counter - backward); }
    counting_iterator operator+(difference_type forward) const { return counting_iterator(my_counter + forward); }
    friend counting_iterator operator+(difference_type forward, const counting_iterator it) { return it + forward; }

    bool operator==(const counting_iterator& it) const { return *this - it == 0; }
    bool operator!=(const counting_iterator& it) const { return !(*this == it); }
    bool operator<(const counting_iterator& it) const {return *this - it < 0; }
    bool operator>(const counting_iterator& it) const { return it < *this; }
    bool operator<=(const counting_iterator& it) const { return !(*this > it); }
    bool operator>=(const counting_iterator& it) const { return !(*this < it); }

private:
    IntType my_counter;
};
} //namespace tbb

#endif //__TBB_CPP11_DECLTYPE_PRESENT

#if __TBB_CPP11_PRESENT

#include <type_traits>
#include <tuple>

#include "internal/_template_helpers.h" // index_sequence, make_index_sequence

namespace tbb {
namespace internal {

template<size_t N>
struct tuple_util {
    template<typename TupleType, typename DifferenceType>
    static void increment(TupleType& it, DifferenceType forward) {
        std::get<N-1>(it) += forward;
        tuple_util<N-1>::increment(it, forward);
    }
    template<typename TupleType, typename DifferenceType>
    static bool check_sync(const TupleType& it1, const TupleType& it2, DifferenceType val) {
        if(std::get<N-1>(it1) - std::get<N-1>(it2) != val)
            return false;
        return tuple_util<N-1>::check_sync(it1, it2, val);
    }
};

template<>
struct tuple_util<0> {
    template<typename TupleType, typename DifferenceType>
    static void increment(TupleType&, DifferenceType) {}
    template<typename TupleType, typename DifferenceType>
    static bool check_sync(const TupleType&, const TupleType&, DifferenceType) { return true;}
};

template <typename TupleReturnType>
struct make_references {
    template <typename TupleType, std::size_t... Is>
    TupleReturnType operator()(const TupleType& t, tbb::internal::index_sequence<Is...>) {
        return std::tie((*std::get<Is>(t))...);
    }
};

} //namespace internal

template <typename... Types>
class zip_iterator {
    static const std::size_t num_types = sizeof...(Types);
    typedef typename std::tuple<Types...> it_types;
public:
    typedef typename std::make_signed<std::size_t>::type difference_type;
    typedef std::tuple<typename std::iterator_traits<Types>::value_type...> value_type;
    typedef std::tuple<typename std::iterator_traits<Types>::reference...> reference;
    typedef std::tuple<typename std::iterator_traits<Types>::pointer...> pointer;
    typedef std::random_access_iterator_tag iterator_category;

    explicit zip_iterator(Types... args): my_it(std::make_tuple(args...)) {}

    reference operator*() {
        return tbb::internal::make_references<reference>()(my_it, tbb::internal::make_index_sequence<num_types>());
    }
    reference operator[](difference_type i) const { return *(*this + i); }

    difference_type operator-(const zip_iterator& it) const {
        __TBB_ASSERT(internal::tuple_util<num_types>::check_sync(my_it, it.my_it, std::get<0>(my_it) - std::get<0>(it.my_it)),
            "Components of zip_iterator are not synchronous");
        return std::get<0>(my_it) - std::get<0>(it.my_it);
    }

    zip_iterator& operator+=(difference_type forward) {
        internal::tuple_util<num_types>::increment(my_it, forward);
        return *this;
    }
    zip_iterator& operator-=(difference_type backward) { return *this += -backward; }
    zip_iterator& operator++() { return *this += 1; }
    zip_iterator& operator--() { return *this -= 1; }

    zip_iterator operator++(int) {
        zip_iterator it(*this);
        ++(*this);
        return it;
    }
    zip_iterator operator--(int) {
        zip_iterator it(*this);
        --(*this);
        return it;
    }

    zip_iterator operator-(difference_type backward) const {
        zip_iterator it(*this);
        return it -= backward;
    }
    zip_iterator operator+(difference_type forward) const {
        zip_iterator it(*this);
        return it += forward;
    }
    friend zip_iterator operator+(difference_type forward, const zip_iterator& it) { return it + forward; }

    bool operator==(const zip_iterator& it) const {
        return *this - it == 0;
    }
    bool operator!=(const zip_iterator& it) const { return !(*this == it); }
    bool operator<(const zip_iterator& it) const { return *this - it < 0; }
    bool operator>(const zip_iterator& it) const { return it < *this; }
    bool operator<=(const zip_iterator& it) const { return !(*this > it); }
    bool operator>=(const zip_iterator& it) const { return !(*this < it); }

private:
    it_types my_it;
};

template<typename... T>
zip_iterator<T...> make_zip_iterator(T... args) { return zip_iterator<T...>(args...); }

} //namespace tbb

#endif //__TBB_CPP11_PRESENT

#endif /* __TBB_iterators_H */
