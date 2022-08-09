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
#include "tbb/concurrent_set.h"
#if __TBB_INITIALIZER_LISTS_PRESENT
// These operator== are used implicitly in  test_initializer_list.h.
// For some unknown reason clang is not able to find the if they a declared after the
// inclusion of test_initializer_list.h.
template<typename container_type>
bool equal_containers( container_type const& lhs, container_type const& rhs );
template<typename T>
bool operator==(tbb::concurrent_set<T> const& lhs, tbb::concurrent_set<T> const& rhs) {
    return equal_containers( lhs, rhs );
}

template<typename T>
bool operator==(tbb::concurrent_multiset<T> const& lhs, tbb::concurrent_multiset<T> const& rhs) {
    return equal_containers( lhs, rhs );
}
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */
#include "test_concurrent_ordered_common.h"

typedef tbb::concurrent_set<int, std::less<int>, MyAllocator> MySet;
typedef tbb::concurrent_set<int, std::greater<int>, MyAllocator> MyGreaterSet;
typedef tbb::concurrent_set<check_type<int>, std::less<int>, MyAllocator> MyCheckedSet;
typedef tbb::concurrent_set<FooWithAssign, std::less<Foo>, MyAllocator> MyCheckedStateSet;
typedef tbb::concurrent_multiset<int, std::less<int>, MyAllocator> MyMultiSet;
typedef tbb::concurrent_multiset<int, std::greater<int>, MyAllocator> MyGreaterMultiSet;
typedef tbb::concurrent_multiset<check_type<int>, std::less<int>, MyAllocator> MyCheckedMultiSet;

struct co_set_type : ordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_set<element_type, std::less<element_type>, allocator_type > type;
    };

    typedef FooIterator init_iterator_type;
};

struct co_multiset_type : ordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_multiset<element_type, std::less<element_type>, allocator_type > type;
    };

    typedef FooIterator init_iterator_type;
};

struct OrderedSetTypesTester{
    template <bool defCtorPresent, typename ValueType>
    void check( const std::list<ValueType> &lst ) {
        TypeTester< defCtorPresent, tbb::concurrent_set< ValueType >,
                                    tbb::concurrent_set< ValueType , std::less<ValueType>, debug_allocator<ValueType> > >( lst );
        TypeTester< defCtorPresent, tbb::concurrent_multiset< ValueType >,
                                    tbb::concurrent_multiset< ValueType , std::less<ValueType>, debug_allocator<ValueType> > >( lst );
    }
};

void TestTypes() {
    TestSetCommonTypes<OrderedSetTypesTester>();

    #if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT
    // Regression test for a problem with excessive requirements of emplace()
    test_emplace_insert<tbb::concurrent_set< test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    test_emplace_insert<tbb::concurrent_multiset< test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    #endif /*__TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT*/
}

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <template <typename ...> typename TSet>
void TestDeductionGuides() {
    std::vector<int> vc({1, 2, 3});
    TSet set(vc.begin(), vc.end());
    static_assert(std::is_same_v<decltype(set), TSet<int>>, "Wrong");

    std::greater<int> compare;
    std::allocator<int> allocator;

    TSet set2(vc.begin(), vc.end(), compare);
    static_assert(std::is_same_v<decltype(set2), TSet<int, decltype(compare)>>, "Wrong");

    TSet set3(vc.begin(), vc.end(), allocator);
    static_assert(std::is_same_v<decltype(set3), TSet<int, std::less<int>, decltype(allocator)>>, "Wrong");

    TSet set4(vc.begin(), vc.end(), compare, allocator);
    static_assert(std::is_same_v<decltype(set4), TSet<int, decltype(compare), decltype(allocator)>>, "Wrong");

    auto init_list = { int(1), int(2), int(3) };
    TSet set5(init_list);
    static_assert(std::is_same_v<decltype(set5), TSet<int>>, "Wrong");

    TSet set6(init_list, compare);
    static_assert(std::is_same_v<decltype(set6), TSet<int, decltype(compare)>>, "Wrong");

    TSet set7(init_list, allocator);
    static_assert(std::is_same_v<decltype(set7), TSet<int, std::less<int>, decltype(allocator)>>, "Wrong");

    TSet set8(init_list, compare, allocator);
    static_assert(std::is_same_v<decltype(set8), TSet<int, decltype(compare), decltype(allocator)>>, "Wrong");
}
#endif /*__TBB_CPP17_DEDUCTION_GUIDES_PRESENT*/

void test_heterogeneous_functions() {
    check_heterogeneous_functions<tbb::concurrent_set<int, transparent_less> >();
    check_heterogeneous_functions<tbb::concurrent_multiset<int, transparent_less> >();
}

struct compare_keys_less {
    bool operator() (const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
      return std::less<int>()(lhs.first, rhs.first);
    }
};

struct compare_keys_greater {
    bool operator() (const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
      return std::greater<int>()(lhs.first, rhs.first);
    }
};

void multicontainer_specific_test() {
    check_multicontainer_internal_order<tbb::concurrent_multiset<std::pair<int, int>, compare_keys_less > >();
    check_multicontainer_internal_order<tbb::concurrent_multiset<std::pair<int, int>, compare_keys_greater > >();
}

#if !__TBB_SCOPED_ALLOCATOR_BROKEN
#include <scoped_allocator>

template <template<typename...> class Set>
void test_scoped_allocator() {
    using allocator_data_type = allocator_aware_data<std::scoped_allocator_adaptor<tbb::tbb_allocator<int>>>;
    using allocator_type = std::scoped_allocator_adaptor<tbb::tbb_allocator<allocator_data_type>>;
    using set_type = Set<allocator_data_type, allocator_data_compare, allocator_type>;

    allocator_type allocator;
    allocator_data_type v1(1, allocator), v2(2, allocator);
    set_type set1(allocator), set2(allocator);

    auto init_list = { v1, v2 };

    allocator_data_type::assert_on_constructions = true;
    set1.emplace(v1);
    set2.emplace(std::move(v1));

    set1.clear();
    set2.clear();

    set1.insert(v1);
    set2.insert(std::move(v1));

    set1.clear();
    set2.clear();

    set1.insert(init_list);

    set1.clear();
    set2.clear();

    set1 = set2;
    set2 = std::move(set1);

    set1.swap(set2);

    allocator_data_type::assert_on_constructions = false;
}

#endif // !__TBB_SCOPED_ALLOCATOR_BROKEN

int TestMain() {
    test_machine();

    test_basic<MySet>( "concurrent Set" );
    test_basic<MyGreaterSet>( "concurrent greater Set" );
    test_concurrent<MySet>( "concurrent Set" );
    test_concurrent<MyGreaterSet>( "concurrent greater Set" );
    test_basic<MyMultiSet>( "concurrent MultiSet" );
    test_basic<MyGreaterMultiSet>( "concurrent greater MultiSet" );
    test_concurrent<MyMultiSet>( "concurrent MultiSet" );
    test_concurrent<MyGreaterMultiSet>( "concurrent greater MultiSet" );

    { Check<MyCheckedSet::value_type> checkit; test_basic<MyCheckedSet>( "concurrent set (checked)" ); }
    { Check<MyCheckedSet::value_type> checkit; test_concurrent<MyCheckedSet>( "concurrent set (checked)" ); }
    test_basic<MyCheckedStateSet>("concurrent set (checked state of elements)", tbb::internal::true_type());
    test_concurrent<MyCheckedStateSet>("concurrent set (checked state of elements)");

    { Check<MyCheckedMultiSet::value_type> checkit; test_basic<MyCheckedMultiSet>( "concurrent MultiSet (checked)" ); }
    { Check<MyCheckedMultiSet::value_type> checkit; test_concurrent<MyCheckedMultiSet>( "concurrent MultiSet (checked)" ); }

    multicontainer_specific_test();

    TestInitList< tbb::concurrent_set<int>,
                  tbb::concurrent_multiset<int> >( {1,2,3,4,5} );

#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor<MySet>();
    TestRangeBasedFor<MyMultiSet>();
#endif

    test_rvalue_ref_support<co_set_type>( "concurrent map" );
    test_rvalue_ref_support<co_multiset_type>( "concurrent multimap" );

    TestTypes();

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    TestDeductionGuides<tbb::concurrent_set>();
    TestDeductionGuides<tbb::concurrent_multiset>();
#endif

    node_handling::TestNodeHandling<MySet>();
    node_handling::TestNodeHandling<MyMultiSet>();
    node_handling::TestMerge<MySet, MyMultiSet>(1000);

    test_heterogeneous_functions();

    test_allocator_traits<tbb::concurrent_set, int, std::less<int>>();
    test_allocator_traits<tbb::concurrent_multiset, int, std::less<int>>();

#if !__TBB_SCOPED_ALLOCATOR_BROKEN
    test_scoped_allocator<tbb::concurrent_set>();
    test_scoped_allocator<tbb::concurrent_multiset>();
#endif

    return Harness::Done;
}
#else
int TestMain() {
    return Harness::Skipped;
}
#endif
