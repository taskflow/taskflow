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

#if _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "harness_defs.h"
#if !(__TBB_TEST_SECONDARY && __TBB_CPP11_STD_PLACEHOLDERS_LINKAGE_BROKEN)

#define __TBB_EXTRA_DEBUG 1
#include "tbb/concurrent_unordered_set.h"
#include "harness_assert.h"

#if __TBB_TEST_SECONDARY

#include "harness_runtime_loader.h"

#else // __TBB_TEST_SECONDARY
#if __TBB_INITIALIZER_LISTS_PRESENT
// These operator== are used implicitly in  test_initializer_list.h.
// For some unknown reason clang is not able to find the if they a declared after the
// inclusion of test_initializer_list.h.
template<typename container_type>
bool equal_containers( container_type const& lhs, container_type const& rhs );
template<typename T>
bool operator==(tbb::concurrent_unordered_set<T> const& lhs, tbb::concurrent_unordered_set<T> const& rhs) {
    return equal_containers( lhs, rhs );
}

template<typename T>
bool operator==(tbb::concurrent_unordered_multiset<T> const& lhs, tbb::concurrent_unordered_multiset<T> const& rhs) {
    return equal_containers( lhs, rhs );
}
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */
#include "test_concurrent_unordered_common.h"

typedef tbb::concurrent_unordered_set<int, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MySet;
typedef tbb::concurrent_unordered_set<int, degenerate_hash<int>, std::equal_to<int>, MyAllocator> MyDegenerateSet;
typedef tbb::concurrent_unordered_set<check_type<int>, tbb::tbb_hash<check_type<int> >, std::equal_to<check_type<int> >, MyAllocator> MyCheckedSet;
typedef tbb::concurrent_unordered_set<FooWithAssign, tbb::tbb_hash<Foo>, std::equal_to<FooWithAssign>, MyAllocator> MyCheckedStateSet;
typedef tbb::concurrent_unordered_multiset<int, tbb::tbb_hash<int>, std::equal_to<int>, MyAllocator> MyMultiSet;
typedef tbb::concurrent_unordered_multiset<int, degenerate_hash<int>, std::equal_to<int>, MyAllocator> MyDegenerateMultiSet;
typedef tbb::concurrent_unordered_multiset<check_type<int>, tbb::tbb_hash<check_type<int> >, std::equal_to<check_type<int> >, MyAllocator> MyCheckedMultiSet;

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct cu_set_type : unordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_unordered_set<element_type, tbb::tbb_hash<element_type>, std::equal_to<element_type>, allocator_type > type;
    };

    typedef FooIterator init_iterator_type;
};

struct cu_multiset_type : unordered_move_traits_base {
    template<typename element_type, typename allocator_type>
    struct apply {
        typedef tbb::concurrent_unordered_multiset<element_type, tbb::tbb_hash<element_type>, std::equal_to<element_type>, allocator_type > type;
    };

    typedef FooIterator init_iterator_type;
};
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
struct UnorderedSetTypesTester {
    template <bool defCtorPresent, typename value_type>
    void check( const std::list<value_type> &lst ) {
        TypeTester< defCtorPresent, tbb::concurrent_unordered_set<value_type, tbb::tbb_hash<value_type>, Harness::IsEqual>,
            tbb::concurrent_unordered_set< value_type, tbb::tbb_hash<value_type>, Harness::IsEqual, debug_allocator<value_type> > >( lst );
        TypeTester< defCtorPresent, tbb::concurrent_unordered_multiset<value_type, tbb::tbb_hash<value_type>, Harness::IsEqual>,
            tbb::concurrent_unordered_multiset< value_type, tbb::tbb_hash<value_type>, Harness::IsEqual, debug_allocator<value_type> > >( lst );
    }
};

void TestTypes( ) {
    TestSetCommonTypes<UnorderedSetTypesTester>();

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT
    // Regression test for a problem with excessive requirements of emplace()
    test_emplace_insert<tbb::concurrent_unordered_set< test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
    test_emplace_insert<tbb::concurrent_unordered_multiset< test::unique_ptr<int> >,
                        tbb::internal::false_type>( new int, new int );
#endif /*__TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_SMART_POINTERS_PRESENT*/
}

#endif // __TBB_TEST_SECONDARY

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <template <typename ...> typename TSet>
void TestDeductionGuides() {
    using ComplexType = const std::string *;
    std::vector<ComplexType> v;
    std::string s = "s";
    auto l = { ComplexType(&s), ComplexType(&s)};

    // check TSet(InputIterator,InputIterator)
    TSet s1(v.begin(), v.end());
    static_assert(std::is_same<decltype(s1), TSet<ComplexType>>::value);

    // check TSet(InputIterator,InputIterator, size_t, Hasher)
    TSet s2(v.begin(), v.end(), 5, std::hash<ComplexType>());
    static_assert(std::is_same<decltype(s2), TSet<ComplexType, std::hash<ComplexType>>>::value);

    // check TSet(InputIterator,InputIterator, size_t, Hasher, Equality)
    TSet s3(v.begin(), v.end(), 5, std::hash<ComplexType>(), std::less<ComplexType>());
    static_assert(std::is_same<decltype(s3), TSet<ComplexType, std::hash<ComplexType>,
        std::less<ComplexType>>>::value);

    // check TSet(InputIterator,InputIterator, size_t, Hasher, Equality, Allocator)
    TSet s4(v.begin(), v.end(), 5, std::hash<ComplexType>(), std::less<ComplexType>(),
        std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s4), TSet<ComplexType, std::hash<ComplexType>,
        std::less<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(InputIterator,InputIterator, size_t, Allocator)
    TSet s5(v.begin(), v.end(), 5, std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s5), TSet<ComplexType, tbb::tbb_hash<ComplexType>,
        std::equal_to<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(InputIterator,InputIterator, size_t, Hasher, Allocator)
    TSet s6(v.begin(), v.end(), 5, std::hash<ComplexType>(), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s6), TSet<ComplexType, std::hash<ComplexType>,
        std::equal_to<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(std::initializer_list)
    TSet s7(l);
    static_assert(std::is_same<decltype(s7), TSet<ComplexType>>::value);

    // check TSet(std::initializer_list, size_t, Hasher)
    TSet s8(l, 5, std::hash<ComplexType>());
    static_assert(std::is_same<decltype(s8), TSet<ComplexType, std::hash<ComplexType>>>::value);

    // check TSet(std::initializer_list, size_t, Hasher, Equality)
    TSet s9(l, 5, std::hash<ComplexType>(), std::less<ComplexType>());
    static_assert(std::is_same<decltype(s9), TSet<ComplexType, std::hash<ComplexType>,
        std::less<ComplexType>>>::value);

    // check TSet(std::initializer_list, size_t, Hasher, Equality, Allocator)
    TSet s10(l, 5, std::hash<ComplexType>(), std::less<ComplexType>(), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s10), TSet<ComplexType, std::hash<ComplexType>,
        std::less<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(std::initializer_list, size_t, Allocator)
    TSet s11(l, 5, std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s11), TSet<ComplexType, tbb::tbb_hash<ComplexType>,
        std::equal_to<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(std::initializer_list, size_t, Hasher, Allocator)
    TSet s12(l, 5, std::hash<ComplexType>(), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s12), TSet<ComplexType, std::hash<ComplexType>,
        std::equal_to<ComplexType>, std::allocator<ComplexType>>>::value);

    // check TSet(TSet &)
    TSet s13(s1);
    static_assert(std::is_same<decltype(s13), decltype(s1)>::value);

    // check TSet(TSet &, Allocator)
    TSet s14(s5, std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s14), decltype(s5)>::value);

    // check TSet(TSet &&)
    TSet s15(std::move(s1));
    static_assert(std::is_same<decltype(s15), decltype(s1)>::value);

    // check TSet(TSet &&, Allocator)
    TSet s16(std::move(s5), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(s16), decltype(s5)>::value);
}
#endif

#if !__TBB_TEST_SECONDARY
#define INITIALIZATION_TIME_TEST_NAMESPACE            initialization_time_test
#define TEST_INITIALIZATION_TIME_OPERATIONS_NAME      test_initialization_time_operations
void test_initialization_time_operations_external();
#else
#define INITIALIZATION_TIME_TEST_NAMESPACE            initialization_time_test_external
#define TEST_INITIALIZATION_TIME_OPERATIONS_NAME      test_initialization_time_operations_external
#endif

namespace INITIALIZATION_TIME_TEST_NAMESPACE {
    tbb::concurrent_unordered_set<int> static_init_time_set;
    int any_non_zero_value = 89432;
    bool static_init_time_inserted = (static_init_time_set.insert( any_non_zero_value )).second;
    bool static_init_time_found = ((static_init_time_set.find( any_non_zero_value )) != static_init_time_set.end( ));
}
void TEST_INITIALIZATION_TIME_OPERATIONS_NAME( ) {
    using namespace INITIALIZATION_TIME_TEST_NAMESPACE;
#define LOCATION ",in function: " __TBB_STRING(TEST_INITIALIZATION_TIME_OPERATIONS_NAME)
    ASSERT( static_init_time_inserted, "failed to insert an item during initialization of global objects" LOCATION );
    ASSERT( static_init_time_found, "failed to find an item during initialization of global objects" LOCATION );

    bool static_init_time_found_in_main = ((static_init_time_set.find( any_non_zero_value )) != static_init_time_set.end( ));
    ASSERT( static_init_time_found_in_main, "failed to find during main() an item inserted during initialization of global objects" LOCATION );
#undef LOCATION
}

#if !__TBB_TEST_SECONDARY
int TestMain() {
    test_machine( );

    test_basic<MySet>( "concurrent unordered Set" );
    test_basic<MyDegenerateSet>( "concurrent unordered degenerate Set" );
    test_concurrent<MySet>("concurrent unordered Set");
    test_concurrent<MyDegenerateSet>( "concurrent unordered degenerate Set" );
    test_basic<MyMultiSet>("concurrent unordered MultiSet");
    test_basic<MyDegenerateMultiSet>("concurrent unordered degenerate MultiSet");
    test_concurrent<MyMultiSet>( "concurrent unordered MultiSet" );
    test_concurrent<MyDegenerateMultiSet>("concurrent unordered degenerate MultiSet");
    test_concurrent<MyMultiSet>( "concurrent unordered MultiSet asymptotic", true );

    { Check<MyCheckedSet::value_type> checkit; test_basic<MyCheckedSet>( "concurrent_unordered_set (checked)" ); }
    { Check<MyCheckedSet::value_type> checkit; test_concurrent<MyCheckedSet>( "concurrent unordered set (checked)" ); }
    test_basic<MyCheckedStateSet>("concurrent unordered set (checked element state)", tbb::internal::true_type());
    test_concurrent<MyCheckedStateSet>("concurrent unordered set (checked element state)");

    { Check<MyCheckedMultiSet::value_type> checkit; test_basic<MyCheckedMultiSet>("concurrent_unordered_multiset (checked)"); }
    { Check<MyCheckedMultiSet::value_type> checkit; test_concurrent<MyCheckedMultiSet>( "concurrent unordered multiset (checked)" ); }

    test_initialization_time_operations( );
#if !__TBB_CPP11_STD_PLACEHOLDERS_LINKAGE_BROKEN
    test_initialization_time_operations_external( );
#else
    REPORT( "Known issue: global objects initialization time tests skipped.\n" );
#endif //!__TBB_CPP11_STD_PLACEHOLDERS_LINKING_BROKEN

#if __TBB_INITIALIZER_LISTS_PRESENT
    TestInitList< tbb::concurrent_unordered_set<int>,
                  tbb::concurrent_unordered_multiset<int> >( {1,2,3,4,5} );
#endif

#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor<MySet>();
    TestRangeBasedFor<MyMultiSet>();
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT
    test_rvalue_ref_support<cu_set_type>( "concurrent unordered set" );
    test_rvalue_ref_support<cu_multiset_type>( "concurrent unordered multiset" );
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    TestTypes();

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    TestDeductionGuides<tbb::concurrent_unordered_set>();
    TestDeductionGuides<tbb::concurrent_unordered_multiset>();
#endif

#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    node_handling::TestNodeHandling<MySet>();
    node_handling::TestNodeHandling<MyMultiSet>();
    node_handling::TestMerge<MySet, MyMultiSet>(10000);
    node_handling::TestMerge<MySet, MyDegenerateSet>(10000);
#endif /*__TBB_UNORDERED_NODE_HANDLE_PRESENT*/

    return Harness::Done;
}
#endif //#if !__TBB_TEST_SECONDARY
#endif //!(__TBB_TEST_SECONDARY && __TBB_CPP11_STD_PLACEHOLDERS_LINKAGE_BROKEN)
