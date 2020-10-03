/*
    Copyright (c) 2005-2020 Intel Corporation

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

#include "tbb/concurrent_unordered_map.h"
#if __TBB_INITIALIZER_LISTS_PRESENT
// These operator== are used implicitly in  test_initializer_list.h.
// For some unknown reason clang is not able to find the if they a declared after the
// inclusion of test_initializer_list.h.
template<typename container_type>
bool equal_containers( container_type const& lhs, container_type const& rhs );
template<typename Key, typename Value>
bool operator==( tbb::concurrent_unordered_map<Key, Value> const& lhs, tbb::concurrent_unordered_map<Key, Value> const& rhs ) {
    return equal_containers( lhs, rhs );
}
template<typename Key, typename Value>
bool operator==( tbb::concurrent_unordered_multimap<Key, Value> const& lhs, tbb::concurrent_unordered_multimap<Key, Value> const& rhs ) {
    return equal_containers( lhs, rhs );
}
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */
#include "test_concurrent_unordered_common.h"

typedef tbb::concurrent_unordered_map<int, int, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MyMap;
typedef tbb::concurrent_unordered_map<int, int, degenerate_hash<int>, std::equal_to<int>, MyAllocator> MyDegenerateMap;
typedef tbb::concurrent_unordered_map<int, check_type<int>, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MyCheckedMap;
typedef tbb::concurrent_unordered_map<intptr_t, FooWithAssign, tbb::tbb_hash<intptr_t>, std::equal_to<intptr_t>, MyAllocator> MyCheckedStateMap;
typedef tbb::concurrent_unordered_multimap<int, int, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MyMultiMap;
typedef tbb::concurrent_unordered_multimap<int, int, degenerate_hash<int>, std::equal_to<int>, MyAllocator> MyDegenerateMultiMap;
typedef tbb::concurrent_unordered_multimap<int, check_type<int>, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MyCheckedMultiMap;

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

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct cu_map_type : unordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_unordered_map<element_type, element_type, tbb::tbb_hash<element_type>, std::equal_to<element_type>, allocator_type > type;
    };

    typedef FooPairIterator init_iterator_type;
};

struct cu_multimap_type : unordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_unordered_multimap<element_type, element_type, tbb::tbb_hash<element_type>, std::equal_to<element_type>, allocator_type > type;
    };

    typedef FooPairIterator init_iterator_type;
};
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

template <bool defCtorPresent, typename Key, typename Element, typename Hasher, typename Equality, typename Allocator>
void TestMapSpecificMethods( tbb::concurrent_unordered_map<Key, Element, Hasher, Equality, Allocator> &c,
    const typename tbb::concurrent_unordered_map<Key, Element, Hasher, Equality, Allocator>::value_type &value ) {
        TestMapSpecificMethodsImpl<defCtorPresent>(c, value);
    }

struct UnorderedMapTypesTester{
    template <bool defCtorPresent, typename ValueType>
    void check( const std::list<ValueType> &lst ) {
        typedef typename ValueType::first_type KeyType;
        typedef typename ValueType::second_type ElemType;
        TypeTester< defCtorPresent, tbb::concurrent_unordered_map< KeyType, ElemType, tbb::tbb_hash<KeyType>, Harness::IsEqual>,
                                    tbb::concurrent_unordered_map< KeyType, ElemType, tbb::tbb_hash<KeyType>, Harness::IsEqual, debug_allocator<ValueType> > >( lst );
        TypeTester< defCtorPresent, tbb::concurrent_unordered_multimap< KeyType, ElemType, tbb::tbb_hash<KeyType>, Harness::IsEqual>,
                                    tbb::concurrent_unordered_multimap< KeyType, ElemType, tbb::tbb_hash<KeyType>, Harness::IsEqual, debug_allocator<ValueType> > >( lst );
    }
};

void TestTypes() {
    TestMapCommonTypes<UnorderedMapTypesTester>();

    #if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT
    // Regression test for a problem with excessive requirements of emplace()
    test_emplace_insert<tbb::concurrent_unordered_map< int*, test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    test_emplace_insert<tbb::concurrent_unordered_multimap< int*, test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    #endif /*__TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT*/
}

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <template <typename...> typename TMap>
void TestDeductionGuides() {
    using ComplexType = std::pair<int, std::string>;
    using ComplexTypeConst = std::pair<const int, std::string>;
    std::vector<ComplexType> v;
    auto l = { ComplexTypeConst(1, "one"), ComplexTypeConst(2, "two")};

    // check TMap(InputIterator, InputIterator)
    TMap m0(v.begin(), v.end());
    static_assert(std::is_same<decltype(m0), TMap<int, std::string>>::value);

    // check TMap(InputIterator, InputIterator, size_t)
    TMap m1(v.begin(), v.end(), 1);
    static_assert(std::is_same<decltype(m1), TMap<int, std::string>>::value);


    // check TMap(InputIterator, InputIterator, size_t, Hasher)
    TMap m2(v.begin(), v.end(), 4, std::hash<int>());
    static_assert(std::is_same<decltype(m2), TMap<int, std::string, std::hash<int>>>::value);

    // check TMap(InputIterator, InputIterator, size_t, Hasher, Equality)
    TMap m3(v.begin(), v.end(), 4, std::hash<int>(), std::less<int>());
    static_assert(std::is_same<decltype(m3), TMap<int, std::string, std::hash<int>, std::less<int>>>::value);

    // check TMap(InputIterator, InputIterator, size_t, Hasher, Equality, Allocator)
    TMap m4(v.begin(), v.end(), 4, std::hash<int>(), std::less<int>(), std::allocator<int>());
    static_assert(std::is_same<decltype(m4), TMap<int, std::string, std::hash<int>,
        std::less<int>, std::allocator<int>>>::value);

    // check TMap(InputIterator, InputIterator, size_t, Allocator)
    TMap m5(v.begin(), v.end(), 5, std::allocator<int>());
    static_assert(std::is_same<decltype(m5), TMap<int, std::string, tbb::tbb_hash<int>,
        std::equal_to<int>, std::allocator<int>>>::value);

    // check TMap(InputIterator, InputIterator, size_t, Hasher, Allocator)
    TMap m6(v.begin(), v.end(), 4, std::hash<int>(), std::allocator<int>());
    static_assert(std::is_same<decltype(m6), TMap<int, std::string, std::hash<int>,
        std::equal_to<int>, std::allocator<int>>>::value);

    // check TMap(std::initializer_list)
    TMap m7(l);
    static_assert(std::is_same<decltype(m7), TMap<int, std::string>>::value);

    // check TMap(std::initializer_list, size_t)
    TMap m8(l, 1);
    static_assert(std::is_same<decltype(m8), TMap<int, std::string>>::value);

    // check TMap(std::initializer_list, size_t, Hasher)
    TMap m9(l, 4, std::hash<int>());
    static_assert(std::is_same<decltype(m9), TMap<int, std::string, std::hash<int>>>::value);

    // check TMap(std::initializer_list, size_t, Hasher, Equality)
    TMap m10(l, 4, std::hash<int>(), std::less<int>());
    static_assert(std::is_same<decltype(m10), TMap<int, std::string, std::hash<int>, std::less<int>>>::value);

    // check TMap(std::initializer_list, size_t, Hasher, Equality, Allocator)
    TMap m11(l, 4, std::hash<int>(), std::less<int>(), std::allocator<int>());
    static_assert(std::is_same<decltype(m11), TMap<int, std::string, std::hash<int>,
        std::less<int>, std::allocator<int>>>::value);

    // check TMap(std::initializer_list, size_t, Allocator)
    TMap m12(l, 4, std::allocator<int>());
    static_assert(std::is_same<decltype(m12), TMap<int, std::string, tbb::tbb_hash<int>,
        std::equal_to<int>, std::allocator<int>>>::value);

    // check TMap(std::initializer_list, size_t, Hasher, Allocator)
    TMap m13(l, 4, std::hash<int>(), std::allocator<int>());
    static_assert(std::is_same<decltype(m13), TMap<int, std::string, std::hash<int>,
        std::equal_to<int>, std::allocator<int>>>::value);

    // check TMap(TMap &)
    TMap m14(m1);
    static_assert(std::is_same<decltype(m14), decltype(m1)>::value);

    // check TMap(TMap &, Allocator)
    TMap m15(m5, std::allocator<int>());
    static_assert(std::is_same<decltype(m15), decltype(m5)>::value);

    // check TMap(TMap &&)
    TMap m16(std::move(m1));
    static_assert(std::is_same<decltype(m16), decltype(m1)>::value);

    // check TMap(TMap &&, Allocator)
    TMap m17(std::move(m5), std::allocator<int>());
    static_assert(std::is_same<decltype(m17), decltype(m5)>::value);
}
#endif

int TestMain() {
    test_machine();

    test_basic<MyMap>( "concurrent unordered Map" );
    test_basic<MyDegenerateMap>( "concurrent unordered degenerate Map" );
    test_concurrent<MyMap>( "concurrent unordered Map" );
    test_concurrent<MyDegenerateMap>( "concurrent unordered degenerate Map" );
    test_basic<MyMultiMap>( "concurrent unordered MultiMap" );
    test_basic<MyDegenerateMultiMap>( "concurrent unordered degenerate MultiMap" );
    test_concurrent<MyMultiMap>( "concurrent unordered MultiMap" );
    test_concurrent<MyDegenerateMultiMap>( "concurrent unordered degenerate MultiMap" );
    test_concurrent<MyMultiMap>( "concurrent unordered MultiMap asymptotic", true );

    { Check<MyCheckedMap::value_type> checkit; test_basic<MyCheckedMap>( "concurrent unordered map (checked)" ); }
    { Check<MyCheckedMap::value_type> checkit; test_concurrent<MyCheckedMap>( "concurrent unordered map (checked)" ); }
    test_basic<MyCheckedStateMap>("concurrent unordered map (checked state of elements)", tbb::internal::true_type());
    test_concurrent<MyCheckedStateMap>("concurrent unordered map (checked state of elements)");

    { Check<MyCheckedMultiMap::value_type> checkit; test_basic<MyCheckedMultiMap>( "concurrent unordered MultiMap (checked)" ); }
    { Check<MyCheckedMultiMap::value_type> checkit; test_concurrent<MyCheckedMultiMap>( "concurrent unordered MultiMap (checked)" ); }

#if __TBB_INITIALIZER_LISTS_PRESENT
    TestInitList< tbb::concurrent_unordered_map<int, int>,
                  tbb::concurrent_unordered_multimap<int, int> >( {{1,1},{2,2},{3,3},{4,4},{5,5}} );
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */

#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor<MyMap>();
    TestRangeBasedFor<MyMultiMap>();
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT
    test_rvalue_ref_support<cu_map_type>( "concurrent unordered map" );
    test_rvalue_ref_support<cu_multimap_type>( "concurrent unordered multimap" );
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    TestDeductionGuides<tbb::concurrent_unordered_map>();
    TestDeductionGuides<tbb::concurrent_unordered_multimap>();
#endif

    TestTypes();

#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    node_handling::TestNodeHandling<MyMap>();
    node_handling::TestNodeHandling<MyMultiMap>();
    node_handling::TestMerge<MyMap, MyMultiMap>(10000);
    node_handling::TestMerge<MyMap, MyDegenerateMap>(10000);
#endif /*__TBB_UNORDERED_NODE_HANDLE_PRESENT*/

    return Harness::Done;
}
