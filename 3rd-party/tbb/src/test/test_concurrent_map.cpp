/*
    Copyright (c) 2019-2020 Intel Corporation

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

#define __TBB_EXTRA_DEBUG 1
#if _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "tbb/tbb_config.h"
#include "harness.h"
#if __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT

#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include "tbb/concurrent_map.h"
#if __TBB_INITIALIZER_LISTS_PRESENT
// These operator== are used implicitly in  test_initializer_list.h.
// For some unknown reason clang is not able to find the if they a declared after the
// inclusion of test_initializer_list.h.
template<typename container_type>
bool equal_containers( container_type const& lhs, container_type const& rhs );
template<typename Key, typename Value>
bool operator==( tbb::concurrent_map<Key, Value> const& lhs, tbb::concurrent_map<Key, Value> const& rhs ) {
    return equal_containers( lhs, rhs );
}
template<typename Key, typename Value>
bool operator==( tbb::concurrent_multimap<Key, Value> const& lhs, tbb::concurrent_multimap<Key, Value> const& rhs ) {
    return equal_containers( lhs, rhs );
}
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */
#include "test_concurrent_ordered_common.h"

typedef tbb::concurrent_map<int, int, std::less<int>, MyAllocator> MyMap;
typedef tbb::concurrent_map<int, int, std::greater<int>, MyAllocator> MyGreaterMap;
typedef tbb::concurrent_map<int, check_type<int>, std::less<int>, MyAllocator> MyCheckedMap;
typedef tbb::concurrent_map<intptr_t, FooWithAssign, std::less<intptr_t>, MyAllocator> MyCheckedStateMap;
typedef tbb::concurrent_multimap<int, int, std::less<int>, MyAllocator> MyMultiMap;
typedef tbb::concurrent_multimap<int, int, std::greater<int>, MyAllocator> MyGreaterMultiMap;
typedef tbb::concurrent_multimap<int, check_type<int>, std::less<int>, MyAllocator> MyCheckedMultiMap;

template <>
struct SpecialTests <MyMap> {
    static void Test( const char *str ) {
        SpecialMapTests<MyMap>(str);
    }
};

template <>
struct SpecialTests <MyMultiMap> {
    static void Test( const char *str ) {
        SpecialMultiMapTests<MyMultiMap>(str);
    }
};

struct co_map_type : ordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_map<element_type, element_type, std::less<element_type>, allocator_type > type;
    };

    typedef FooPairIterator init_iterator_type;
};

struct co_multimap_type : ordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_multimap<element_type, element_type, std::less<element_type>, allocator_type > type;
    };

    typedef FooPairIterator init_iterator_type;
};

template <bool defCtorPresent, typename Key, typename Element, typename Compare, typename Allocator>
void TestMapSpecificMethods( tbb::concurrent_map<Key, Element, Compare, Allocator> &c,
    const typename tbb::concurrent_map<Key, Element, Compare, Allocator>::value_type &value ) {
        TestMapSpecificMethodsImpl<defCtorPresent>(c, value);
    }

struct OrderedMapTypesTester{
    template <bool defCtorPresent, typename ValueType>
    void check( const std::list<ValueType> &lst ) {
        typedef typename ValueType::first_type KeyType;
        typedef typename ValueType::second_type ElemType;
        TypeTester< defCtorPresent, tbb::concurrent_map< KeyType, ElemType>,
                                    tbb::concurrent_map< KeyType, ElemType, std::less<KeyType>, debug_allocator<ValueType> > >( lst );
        TypeTester< defCtorPresent, tbb::concurrent_multimap< KeyType, ElemType>,
                                    tbb::concurrent_multimap< KeyType, ElemType, std::less<KeyType>, debug_allocator<ValueType> > >( lst );
    }
};

void TestTypes() {
    TestMapCommonTypes<OrderedMapTypesTester>();

    #if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT
    // Regression test for a problem with excessive requirements of emplace()
    test_emplace_insert<tbb::concurrent_map< int*, test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    test_emplace_insert<tbb::concurrent_multimap< int*, test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    #endif /*__TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT*/
}

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <template <typename...> typename TMap>
void TestDeductionGuides() {
    std::vector<std::pair<int, int>> v(10, {0, 0});
    TMap map(v.begin(), v.end());
    static_assert(std::is_same_v<decltype(map), TMap<int, int> >, "WRONG\n");
    //print(map);

    std::greater<int> compare;
    std::allocator<int> allocator;
    TMap map2(v.begin(), v.end(), compare);
    static_assert(std::is_same_v<decltype(map2), TMap<int, int, decltype(compare)> >, "WRONG\n");

    TMap map3(v.begin(), v.end(), allocator);
    static_assert(std::is_same_v<decltype(map3), TMap<int, int, std::less<int>, decltype(allocator)> >, "WRONG\n");

    TMap map4(v.begin(), v.end(), compare, allocator);
    static_assert(std::is_same_v<decltype(map4), TMap<int, int, decltype(compare), decltype(allocator)> >, "WRONG\n");

    using pair_t = std::pair<const int, int>;
    auto init = { pair_t{1, 1}, pair_t{2, 2}, pair_t{3, 3} };
    TMap map5(init);
    static_assert(std::is_same_v<decltype(map5), TMap<int, int> >, "WRONG\n");

    TMap map6(init, compare);
    static_assert(std::is_same_v<decltype(map6), TMap<int, int, decltype(compare)> >, "WRONG\n");

    TMap map7(init, allocator);
    static_assert(std::is_same_v<decltype(map7), TMap<int, int, std::less<int>, decltype(allocator)> >, "WRONG\n");

    TMap map8(init, compare, allocator);
    static_assert(std::is_same_v<decltype(map8), TMap<int, int, decltype(compare), decltype(allocator)> >, "WRONG\n");
}
#endif

void test_heterogeneous_functions() {
    check_heterogeneous_functions<tbb::concurrent_map<int, int, transparent_less> >();
    check_heterogeneous_functions<tbb::concurrent_multimap<int, int, transparent_less> >();
}

void multicontainer_specific_test() {
    check_multicontainer_internal_order<tbb::concurrent_multimap<int, int> >();
    check_multicontainer_internal_order<tbb::concurrent_multimap<int, int, std::greater<int> > >();
}

#if !__TBB_SCOPED_ALLOCATOR_BROKEN
#include <scoped_allocator>

template <template<typename...> class Map>
void test_scoped_allocator() {
    using allocator_data_type = allocator_aware_data<std::scoped_allocator_adaptor<tbb::tbb_allocator<int>>>;
    using allocator_type = std::scoped_allocator_adaptor<tbb::tbb_allocator<allocator_data_type>>;
    using map_type = Map<allocator_data_type, allocator_data_type, allocator_data_compare, allocator_type>;

    allocator_type allocator;
    allocator_data_type key1(1, allocator), key2(2, allocator);
    allocator_data_type data1(1, allocator), data2(2, allocator);
    map_type map1(allocator), map2(allocator);

    typename map_type::value_type v1(key1, data1), v2(key2, data2);

    auto init_list = { v1, v2 };

    allocator_data_type::assert_on_constructions = true;
    map1.emplace(key1, data1);
    map2.emplace(key2, std::move(data2));

    map1.clear();
    map2.clear();

    map1.insert(v1);
    map2.insert(std::move(v2));

    map1.clear();
    map2.clear();

    map1.insert(init_list);

    map1.clear();
    map2.clear();

    map1 = map2;
    map2 = std::move(map1);

    map1.swap(map2);

    allocator_data_type::assert_on_constructions = false;
}
#endif // !__TBB_SCOPED_ALLOCATOR_BROKEN

int TestMain() {
    test_machine();

    test_basic<MyMap>( "concurrent Map" );
    test_basic<MyGreaterMap>( "concurrent greater Map" );
    test_concurrent<MyMap>( "concurrent Map" );
    test_concurrent<MyGreaterMap>( "concurrent greater Map" );
    test_basic<MyMultiMap>( "concurrent MultiMap" );
    test_basic<MyGreaterMultiMap>( "concurrent greater MultiMap" );
    test_concurrent<MyMultiMap>( "concurrent MultiMap" );
    test_concurrent<MyGreaterMultiMap>( "concurrent greater MultiMap" );

    { Check<MyCheckedMap::value_type> checkit; test_basic<MyCheckedMap>( "concurrent map (checked)" ); }
    { Check<MyCheckedMap::value_type> checkit; test_concurrent<MyCheckedMap>( "concurrent map (checked)" ); }
    test_basic<MyCheckedStateMap>("concurrent map (checked state of elements)", tbb::internal::true_type());
    test_concurrent<MyCheckedStateMap>("concurrent map (checked state of elements)");

    { Check<MyCheckedMultiMap::value_type> checkit; test_basic<MyCheckedMultiMap>( "concurrent MultiMap (checked)" ); }
    { Check<MyCheckedMultiMap::value_type> checkit; test_concurrent<MyCheckedMultiMap>( "concurrent MultiMap (checked)" ); }

    multicontainer_specific_test();

    TestInitList< tbb::concurrent_map<int, int>,
                  tbb::concurrent_multimap<int, int> >( {{1,1},{2,2},{3,3},{4,4},{5,5}} );

#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor<MyMap>();
    TestRangeBasedFor<MyMultiMap>();
#endif

    test_rvalue_ref_support<co_map_type>( "concurrent map" );
    test_rvalue_ref_support<co_multimap_type>( "concurrent multimap" );

    TestTypes();

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    TestDeductionGuides<tbb::concurrent_map>();
    TestDeductionGuides<tbb::concurrent_multimap>();
#endif /*__TBB_CPP17_DEDUCTION_GUIDES_PRESENT*/

    node_handling::TestNodeHandling<MyMap>();
    node_handling::TestNodeHandling<MyMultiMap>();
    node_handling::TestMerge<MyMap, MyMultiMap>(1000);

    test_heterogeneous_functions();

    test_allocator_traits<tbb::concurrent_map, int, int, std::less<int>>();
    test_allocator_traits<tbb::concurrent_multimap, int, int, std::less<int>>();

#if !__TBB_SCOPED_ALLOCATOR_BROKEN
    test_scoped_allocator<tbb::concurrent_map>();
    test_scoped_allocator<tbb::concurrent_multimap>();
#endif

    return Harness::Done;
}
#else
int TestMain() {
    return Harness::Skipped;
}
#endif
