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

/* Some tests in this source file are based on PPL tests provided by Microsoft. */
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "harness.h"
#include "test_container_move_support.h"
// Test that unordered containers do not require keys have default constructors.
#define __HARNESS_CHECKTYPE_DEFAULT_CTOR 0
#include "harness_checktype.h"
#undef  __HARNESS_CHECKTYPE_DEFAULT_CTOR
#include "harness_allocator.h"

#if _MSC_VER
#pragma warning(disable: 4189) // warning 4189 -- local variable is initialized but not referenced
#pragma warning(disable: 4127) // warning 4127 -- while (true) has a constant expression in it
#endif

// TestInitListSupportWithoutAssign with an empty initializer list causes internal error in Intel Compiler.
#define __TBB_ICC_EMPTY_INIT_LIST_TESTS_BROKEN (__INTEL_COMPILER && __INTEL_COMPILER <= 1500)

typedef local_counting_allocator<debug_allocator<std::pair<const int,int>,std::allocator> > MyAllocator;

template<typename Table>
inline void CheckAllocator(typename Table::allocator_type& a, size_t expected_allocs, size_t expected_frees,
                           bool exact = true) {
    if(exact) {
        ASSERT( a.allocations == expected_allocs, NULL); ASSERT( a.frees == expected_frees, NULL);
    } else {
        ASSERT( a.allocations >= expected_allocs, NULL); ASSERT( a.frees >= expected_frees, NULL);
        ASSERT( a.allocations - a.frees == expected_allocs - expected_frees, NULL );
    }
}

// Check that only dummy node allocated if table is empty
// Specialize this function for custom container, if it node allocation size > 1
#define CheckEmptyContainerAllocatorE(t,a,f) CheckEmptyContainerAllocator(t,a,f,true,__LINE__)
#define CheckEmptyContainerAllocatorA(t,a,f) CheckEmptyContainerAllocator(t,a,f,false,__LINE__)
template<typename MyTable>
inline void CheckEmptyContainerAllocator(MyTable &table, size_t expected_allocs, size_t expected_frees, bool exact = true, int line = 0);

template<typename T>
struct strip_const { typedef T type; };

template<typename T>
struct strip_const<const T> { typedef T type; };

// value generator for map
template <typename K, typename V = std::pair<const K, K> >
struct ValueFactory {
    typedef typename strip_const<K>::type Kstrip;
    static V make(const K &value) { return V(value, value); }
    static Kstrip key(const V &value) { return value.first; }
    static Kstrip get(const V &value) { return (Kstrip)value.second; }
    template< typename U >
    static U convert(const V &value) { return U(value.second); }
};

// generator for set
template <typename T>
struct ValueFactory<T, T> {
    static T make(const T &value) { return value; }
    static T key(const T &value) { return value; }
    static T get(const T &value) { return value; }
    template< typename U >
    static U convert(const T &value) { return U(value); }
};

template <typename T>
struct Value : ValueFactory<typename T::key_type, typename T::value_type> {
    template<typename U>
    static bool compare( const typename T::iterator& it, U val ) {
        return (Value::template convert<U>(*it) == val);
    }
};

template<Harness::StateTrackableBase::StateValue desired_state, typename T>
void check_value_state(/* typename do_check_element_state =*/ tbb::internal::true_type, T const& t, const char* filename, int line )
{
    ASSERT_CUSTOM(is_state_f<desired_state>()(t), "", filename, line);
}

template<Harness::StateTrackableBase::StateValue desired_state, typename T>
void check_value_state(/* typename do_check_element_state =*/ tbb::internal::false_type, T const&, const char* , int ) {/*do nothing*/}

#define ASSERT_VALUE_STATE(do_check_element_state,state,value) check_value_state<state>(do_check_element_state,value,__FILE__,__LINE__)

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename T, typename do_check_element_state, typename V>
void test_rvalue_insert(V v1, V v2)
{
    typedef T container_t;

    container_t cont;

    std::pair<typename container_t::iterator, bool> ins = cont.insert(Value<container_t>::make(v1));
    ASSERT(ins.second == true && Value<container_t>::get(*(ins.first)) == v1, "Element 1 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::MoveInitialized,*ins.first);

    typename container_t::iterator it2 = cont.insert(ins.first, Value<container_t>::make(v2));
    ASSERT(Value<container_t>::get(*(it2)) == v2, "Element 2 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::MoveInitialized,*it2);

}
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
// The test does not use variadic templates, but emplace() does.

namespace emplace_helpers {
template<typename container_t, typename arg_t, typename value_t>
std::pair<typename container_t::iterator, bool> call_emplace_impl(container_t& c, arg_t&& k, value_t *){
    // this is a set
    return c.emplace(std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t, typename first_t, typename second_t>
std::pair<typename container_t::iterator, bool> call_emplace_impl(container_t& c, arg_t&& k, std::pair<first_t, second_t> *){
    // this is a map
    return c.emplace(k, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t>
std::pair<typename container_t::iterator, bool> call_emplace(container_t& c, arg_t&& k){
    typename container_t::value_type * selector = NULL;
    return call_emplace_impl(c, std::forward<arg_t>(k), selector);
}

template<typename container_t, typename arg_t, typename value_t>
typename container_t::iterator call_emplace_hint_impl(container_t& c, typename container_t::const_iterator hint, arg_t&& k, value_t *){
    // this is a set
    return c.emplace_hint(hint, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t, typename first_t, typename second_t>
typename container_t::iterator call_emplace_hint_impl(container_t& c, typename container_t::const_iterator hint, arg_t&& k, std::pair<first_t, second_t> *){
    // this is a map
    return c.emplace_hint(hint, k, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t>
typename container_t::iterator call_emplace_hint(container_t& c, typename container_t::const_iterator hint, arg_t&& k){
    typename container_t::value_type * selector = NULL;
    return call_emplace_hint_impl(c, hint, std::forward<arg_t>(k), selector);
}
}
template<typename T, typename do_check_element_state, typename V>
void test_emplace_insert(V v1, V v2){
    typedef T container_t;
    container_t cont;

    std::pair<typename container_t::iterator, bool> ins = emplace_helpers::call_emplace(cont, v1);
    ASSERT(ins.second == true && Value<container_t>::compare(ins.first, v1), "Element 1 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::DirectInitialized,*ins.first);

    typename container_t::iterator it2 = emplace_helpers::call_emplace_hint(cont, ins.first, v2);
    ASSERT(Value<container_t>::compare(it2, v2), "Element 2 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::DirectInitialized,*it2);
}
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

template<typename ContainerType, typename Iterator, typename RangeType>
std::pair<intptr_t,intptr_t> CheckRecursiveRange(RangeType range) {
    std::pair<intptr_t,intptr_t> sum(0, 0); // count, sum
    for( Iterator i = range.begin(), e = range.end(); i != e; ++i ) {
        ++sum.first; sum.second += Value<ContainerType>::get(*i);
    }
    if( range.is_divisible() ) {
        RangeType range2( range, tbb::split() );
        std::pair<intptr_t,intptr_t> sum1 = CheckRecursiveRange<ContainerType,Iterator, RangeType>( range );
        std::pair<intptr_t,intptr_t> sum2 = CheckRecursiveRange<ContainerType,Iterator, RangeType>( range2 );
        sum1.first += sum2.first; sum1.second += sum2.second;
        ASSERT( sum == sum1, "Mismatched ranges after division");
    }
    return sum;
}

template <typename Map>
void SpecialMapTests( const char *str ){
    Map cont;
    const Map &ccont( cont );

    // mapped_type& operator[](const key_type& k);
    cont[1] = 2;

    // bool empty() const;
    ASSERT( !ccont.empty( ), "Concurrent container empty after adding an element" );

    // size_type size() const;
    ASSERT( ccont.size( ) == 1, "Concurrent container size incorrect" );
    ASSERT( cont[1] == 2, "Concurrent container value incorrect" );

    // mapped_type& at( const key_type& k );
    // const mapped_type& at(const key_type& k) const;
    ASSERT( cont.at( 1 ) == 2, "Concurrent container value incorrect" );
    ASSERT( ccont.at( 1 ) == 2, "Concurrent container value incorrect" );

    // iterator find(const key_type& k);
    typename Map::iterator it = cont.find( 1 );
    ASSERT( it != cont.end( ) && Value<Map>::get( *(it) ) == 2, "Element with key 1 not properly found" );
    cont.unsafe_erase( it );

    it = cont.find( 1 );
    ASSERT( it == cont.end( ), "Element with key 1 not properly erased" );
    REMARK( "passed -- specialized %s tests\n", str );
}

template <typename MultiMap>
void CheckMultiMap(MultiMap &m, int *targets, int tcount, int key) {
    std::vector<bool> vfound(tcount,false);
    std::pair<typename MultiMap::iterator, typename MultiMap::iterator> range = m.equal_range( key );
    for(typename MultiMap::iterator it = range.first; it != range.second; ++it) {
        bool found = false;
        for( int i = 0; i < tcount; ++i) {
            if((*it).second == targets[i]) {
                if(!vfound[i])  { // we can insert duplicate values
                    vfound[i] = found = true;
                    break;
                }
            }
        }
        // just in case an extra value in equal_range...
        ASSERT(found, "extra value from equal range");
    }
    for(int i = 0; i < tcount; ++i) ASSERT(vfound[i], "missing value");
}

template <typename MultiMap>
void MultiMapEraseTests(){
    MultiMap cont1, cont2;

    typename MultiMap::iterator erased_it;
    for (int i = 0; i < 10; ++i) {
        if ( i != 1 ) {
            cont1.insert(std::make_pair(1, i));
            cont2.insert(std::make_pair(1, i));
        } else {
            erased_it = cont1.insert(std::make_pair(1, i)).first;
        }
    }

    cont1.unsafe_erase(erased_it);

    ASSERT(cont1.size() == cont2.size(), "Incorrect count of elements was erased");
    typename MultiMap::iterator it1 = cont1.begin();
    typename MultiMap::iterator it2 = cont2.begin();

    for (typename MultiMap::size_type i = 0; i < cont2.size(); ++i) {
        ASSERT(*(it1++) == *(it2++), "Multimap repetitive key was not erased properly");
    }
}

template <typename MultiMap>
void SpecialMultiMapTests( const char *str ){
    int one_values[] = { 7, 2, 13, 23, 13 };
    int zero_values[] = { 4, 9, 13, 29, 42, 111};
    int n_zero_values = sizeof(zero_values) / sizeof(int);
    int n_one_values = sizeof(one_values) / sizeof(int);
    MultiMap cont;
    const MultiMap &ccont( cont );
    // mapped_type& operator[](const key_type& k);
    cont.insert( std::make_pair( 1, one_values[0] ) );

    // bool empty() const;
    ASSERT( !ccont.empty( ), "Concurrent container empty after adding an element" );

    // size_type size() const;
    ASSERT( ccont.size( ) == 1, "Concurrent container size incorrect" );
    ASSERT( (*(cont.begin( ))).second == one_values[0], "Concurrent container value incorrect" );
    ASSERT( (*(cont.equal_range( 1 )).first).second == one_values[0], "Improper value from equal_range" );
    ASSERT( (cont.equal_range( 1 )).second == cont.end( ), "Improper iterator from equal_range" );

    cont.insert( std::make_pair( 1, one_values[1] ) );

    // bool empty() const;
    ASSERT( !ccont.empty( ), "Concurrent container empty after adding an element" );

    // size_type size() const;
    ASSERT( ccont.size( ) == 2, "Concurrent container size incorrect" );
    CheckMultiMap(cont, one_values, 2, 1);

    // insert the other {1,x} values
    for( int i = 2; i < n_one_values; ++i ) {
        cont.insert( std::make_pair( 1, one_values[i] ) );
    }

    CheckMultiMap(cont, one_values, n_one_values, 1);
    ASSERT( (cont.equal_range( 1 )).second == cont.end( ), "Improper iterator from equal_range" );

    cont.insert( std::make_pair( 0, zero_values[0] ) );

    // bool empty() const;
    ASSERT( !ccont.empty( ), "Concurrent container empty after adding an element" );

    // size_type size() const;
    ASSERT( ccont.size( ) == (size_t)(n_one_values+1), "Concurrent container size incorrect" );
    CheckMultiMap(cont, one_values, n_one_values, 1);
    CheckMultiMap(cont, zero_values, 1, 0);
    ASSERT( (*(cont.begin( ))).second == zero_values[0], "Concurrent container value incorrect" );
    // insert the rest of the zero values
    for( int i = 1; i < n_zero_values; ++i) {
        cont.insert( std::make_pair( 0, zero_values[i] ) );
    }
    CheckMultiMap(cont, one_values, n_one_values, 1);
    CheckMultiMap(cont, zero_values, n_zero_values, 0);

    // clear, reinsert interleaved
    cont.clear();
    int bigger_num = ( n_one_values > n_zero_values ) ? n_one_values : n_zero_values;
    for( int i = 0; i < bigger_num; ++i ) {
        if(i < n_one_values) cont.insert( std::make_pair( 1, one_values[i] ) );
        if(i < n_zero_values) cont.insert( std::make_pair( 0, zero_values[i] ) );
    }
    CheckMultiMap(cont, one_values, n_one_values, 1);
    CheckMultiMap(cont, zero_values, n_zero_values, 0);

    MultiMapEraseTests<MultiMap>();

    REMARK( "passed -- specialized %s tests\n", str );
}

template <typename T>
struct SpecialTests {
    static void Test(const char *str) {REMARK("skipped -- specialized %s tests\n", str);}
};



#if __TBB_RANGE_BASED_FOR_PRESENT
#include "test_range_based_for.h"

template <typename Container>
void TestRangeBasedFor() {
    using namespace range_based_for_support_tests;

    REMARK( "testing range based for loop compatibility \n" );
    Container cont;
    const int sequence_length = 100;
    for ( int i = 1; i <= sequence_length; ++i ) {
        cont.insert( Value<Container>::make(i) );
    }

    ASSERT( range_based_for_accumulate( cont, unified_summer(), 0 ) ==
            gauss_summ_of_int_sequence( sequence_length ),
            "incorrect accumulated value generated via range based for ?" );
}
#endif /* __TBB_RANGE_BASED_FOR_PRESENT */

#if __TBB_INITIALIZER_LISTS_PRESENT
// Required by test_initializer_list.h
template<typename container_type>
bool equal_containers(container_type const& lhs, container_type const& rhs) {
        if ( lhs.size() != rhs.size() ) {
            return false;
        }
        return std::equal( lhs.begin(), lhs.end(), rhs.begin(), Harness::IsEqual() );
}

#include "test_initializer_list.h"

template <typename Table, typename MultiTable>
void TestInitList( std::initializer_list<typename Table::value_type> il ) {
    using namespace initializer_list_support_tests;
    REMARK("testing initializer_list methods \n");

    TestInitListSupportWithoutAssign<Table,test_special_insert>(il);
    TestInitListSupportWithoutAssign<MultiTable, test_special_insert>( il );

#if __TBB_ICC_EMPTY_INIT_LIST_TESTS_BROKEN
    REPORT( "Known issue: TestInitListSupportWithoutAssign with an empty initializer list is skipped.\n");
#else
    TestInitListSupportWithoutAssign<Table, test_special_insert>( {} );
    TestInitListSupportWithoutAssign<MultiTable, test_special_insert>( {} );
#endif
}
#endif //if __TBB_INITIALIZER_LISTS_PRESENT

template<typename T, typename do_check_element_state>
void test_basic_common(const char * str, do_check_element_state)
{
    T cont;
    const T &ccont(cont);
    CheckEmptyContainerAllocatorE(cont, 1, 0); // one dummy is always allocated
    // bool empty() const;
    ASSERT(ccont.empty(), "Concurrent container is not empty after construction");

    // size_type size() const;
    ASSERT(ccont.size() == 0, "Concurrent container is not empty after construction");

    // size_type max_size() const;
    ASSERT(ccont.max_size() > 0, "Concurrent container max size is invalid");

    //iterator begin();
    //iterator end();
    ASSERT(cont.begin() == cont.end(), "Concurrent container iterators are invalid after construction");
    ASSERT(ccont.begin() == ccont.end(), "Concurrent container iterators are invalid after construction");
    ASSERT(cont.cbegin() == cont.cend(), "Concurrent container iterators are invalid after construction");

    //std::pair<iterator, bool> insert(const value_type& obj);
    std::pair<typename T::iterator, bool> ins = cont.insert(Value<T>::make(1));
    ASSERT(ins.second == true && Value<T>::get(*(ins.first)) == 1, "Element 1 has not been inserted properly");

#if __TBB_CPP11_RVALUE_REF_PRESENT
    test_rvalue_insert<T,do_check_element_state>(1,2);
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    test_emplace_insert<T,do_check_element_state>(1,2);
#endif // __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

    // bool empty() const;
    ASSERT(!ccont.empty(), "Concurrent container is empty after adding an element");

    // size_type size() const;
    ASSERT(ccont.size() == 1, "Concurrent container size is incorrect");

    std::pair<typename T::iterator, bool> ins2 = cont.insert(Value<T>::make(1));

    if (T::allow_multimapping)
    {
        // std::pair<iterator, bool> insert(const value_type& obj);
        ASSERT(ins2.second == true && Value<T>::get(*(ins2.first)) == 1, "Element 1 has not been inserted properly");

        // size_type size() const;
        ASSERT(ccont.size() == 2, "Concurrent container size is incorrect");

        // size_type count(const key_type& k) const;
        ASSERT(ccont.count(1) == 2, "Concurrent container count(1) is incorrect");
        // std::pair<iterator, iterator> equal_range(const key_type& k);
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        typename T::iterator it = range.first;
        ASSERT(it != cont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        unsigned int count = 0;
        for (; it != range.second; it++)
        {
            count++;
            ASSERT(Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        }

        ASSERT(count == 2, "Range doesn't have the right number of elements");
    }
    else
    {
        // std::pair<iterator, bool> insert(const value_type& obj);
        ASSERT(ins2.second == false && ins2.first == ins.first, "Element 1 should not be re-inserted");

        // size_type size() const;
        ASSERT(ccont.size() == 1, "Concurrent container size is incorrect");

        // size_type count(const key_type& k) const;
        ASSERT(ccont.count(1) == 1, "Concurrent container count(1) is incorrect");

        // std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const;
        // std::pair<iterator, iterator> equal_range(const key_type& k);
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        typename T::iterator it = range.first;
        ASSERT(it != cont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
    }

    // const_iterator find(const key_type& k) const;
    // iterator find(const key_type& k);
    typename T::iterator it = cont.find(1);
    ASSERT(it != cont.end() && Value<T>::get(*(it)) == 1, "Element 1 has not been found properly");
    ASSERT(ccont.find(1) == it, "Element 1 has not been found properly");

    // Will be implemented in unordered containers later
#if !__TBB_UNORDERED_TEST
    //bool contains(const key_type&k) const
    ASSERT(cont.contains(1), "contains() cannot detect existing element");
    ASSERT(!cont.contains(0), "contains() detect not existing element");
#endif /*__TBB_UNORDERED_TEST*/

    // iterator insert(const_iterator hint, const value_type& obj);
    typename T::iterator it2 = cont.insert(ins.first, Value<T>::make(2));
    ASSERT(Value<T>::get(*it2) == 2, "Element 2 has not been inserted properly");

    // T(const T& _Umap)
    T newcont = ccont;
    ASSERT(T::allow_multimapping ? (newcont.size() == 3) : (newcont.size() == 2), "Copy construction has not copied the elements properly");

    // this functionality not implemented yet
    // size_type unsafe_erase(const key_type& k);
    typename T::size_type size = cont.unsafe_erase(1);
    ASSERT(T::allow_multimapping ? (size == 2) : (size == 1), "Erase has not removed the right number of elements");

    // iterator unsafe_erase(iterator position);
    typename T::iterator it4 = cont.unsafe_erase(cont.find(2));
    ASSERT(it4 == cont.end() && cont.size() == 0, "Erase has not removed the last element properly");

    // iterator unsafe_erase(const_iterator position);
    cont.insert(Value<T>::make(3));
    typename T::iterator it5 = cont.unsafe_erase(cont.cbegin());
    ASSERT(it5 == cont.end() && cont.size() == 0, "Erase has not removed the last element properly");

    // template<class InputIterator> void insert(InputIterator first, InputIterator last);
    cont.insert(newcont.begin(), newcont.end());
    ASSERT(T::allow_multimapping ? (cont.size() == 3) : (cont.size() == 2), "Range insert has not copied the elements properly");

    // this functionality not implemented yet
    // iterator unsafe_erase(const_iterator first, const_iterator last);
    std::pair<typename T::iterator, typename T::iterator> range2 = newcont.equal_range(1);
    newcont.unsafe_erase(range2.first, range2.second);
    ASSERT(newcont.size() == 1, "Range erase has not erased the elements properly");

    // void clear();
    newcont.clear();
    ASSERT(newcont.begin() == newcont.end() && newcont.size() == 0, "Clear has not cleared the container");

#if __TBB_INITIALIZER_LISTS_PRESENT
#if __TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    REPORT("Known issue: the test for insert with initializer_list is skipped.\n");
#else
    // void insert(const std::initializer_list<value_type> &il);
    newcont.insert( { Value<T>::make( 1 ), Value<T>::make( 2 ), Value<T>::make( 1 ) } );
    if (T::allow_multimapping) {
        ASSERT(newcont.size() == 3, "Concurrent container size is incorrect");
        ASSERT(newcont.count(1) == 2, "Concurrent container count(1) is incorrect");
        ASSERT(newcont.count(2) == 1, "Concurrent container count(2) is incorrect");
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        unsigned int count = 0;
        for (; it != range.second; it++) {
            count++;
            ASSERT(Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        }
        ASSERT(count == 2, "Range doesn't have the right number of elements");
        range = newcont.equal_range(2); it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        count = 0;
        for (; it != range.second; it++) {
            count++;
            ASSERT(Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        }
        ASSERT(count == 1, "Range doesn't have the right number of elements");
    } else {
        ASSERT(newcont.size() == 2, "Concurrent container size is incorrect");
        ASSERT(newcont.count(1) == 1, "Concurrent container count(1) is incorrect");
        ASSERT(newcont.count(2) == 1, "Concurrent container count(2) is incorrect");
        std::pair<typename T::iterator, typename T::iterator> range = newcont.equal_range(1);
        it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
        range = newcont.equal_range(2); it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
    }
#endif /* __TBB_CPP11_INIT_LIST_TEMP_OBJS_COMPILATION_BROKEN */
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */

    // T& operator=(const T& _Umap)
    newcont = ccont;
    ASSERT(T::allow_multimapping ? (newcont.size() == 3) : (newcont.size() == 2), "Assignment operator has not copied the elements properly");

    REMARK("passed -- basic %s tests\n", str);

#if defined (VERBOSE)
    REMARK("container dump debug:\n");
    cont._Dump();
    REMARK("container dump release:\n");
    cont.dump();
    REMARK("\n");
#endif

    cont.clear();
    CheckEmptyContainerAllocatorA(cont, 1, 0); // one dummy is always allocated
    for (int i = 0; i < 256; i++)
    {
        std::pair<typename T::iterator, bool> ins3 = cont.insert(Value<T>::make(i));
        ASSERT(ins3.second == true && Value<T>::get(*(ins3.first)) == i, "Element 1 has not been inserted properly");
    }
    ASSERT(cont.size() == 256, "Wrong number of elements have been inserted");
    ASSERT((256 == CheckRecursiveRange<T,typename T::iterator>(cont.range()).first), NULL);
    ASSERT((256 == CheckRecursiveRange<T,typename T::const_iterator>(ccont.range()).first), NULL);

    // void swap(T&);
    cont.swap(newcont);
    ASSERT(newcont.size() == 256, "Wrong number of elements after swap");
    ASSERT(newcont.count(200) == 1, "Element with key 200 is not present after swap");
    ASSERT(newcont.count(16) == 1, "Element with key 16 is not present after swap");
    ASSERT(newcont.count(99) == 1, "Element with key 99 is not present after swap");
    ASSERT(T::allow_multimapping ? (cont.size() == 3) : (cont.size() == 2), "Assignment operator has not copied the elements properly");

    // Need to be enabled
    SpecialTests<T>::Test(str);
}

template<typename T>
void test_basic_common(const char * str){
    test_basic_common<T>(str, tbb::internal::false_type());
}

void test_machine() {
    ASSERT(__TBB_ReverseByte(0)==0, NULL );
    ASSERT(__TBB_ReverseByte(1)==0x80, NULL );
    ASSERT(__TBB_ReverseByte(0xFE)==0x7F, NULL );
    ASSERT(__TBB_ReverseByte(0xFF)==0xFF, NULL );
}

template<typename T>
class FillTable: NoAssign {
    T &table;
    const int items;
    bool my_asymptotic;
    typedef std::pair<typename T::iterator, bool> pairIB;
public:
    FillTable(T &t, int i, bool asymptotic) : table(t), items(i), my_asymptotic(asymptotic) {
        ASSERT( !(items&1) && items > 100, NULL);
    }
    void operator()(int threadn) const {
        if( threadn == 0 ) { // Fill even keys forward (single thread)
            bool last_inserted = true;
            for( int i = 0; i < items; i+=2 ) {
                pairIB pib = table.insert(Value<T>::make(my_asymptotic?1:i));
                ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic?1:i), "Element not properly inserted");
                ASSERT( last_inserted || !pib.second, "Previous key was not inserted but this is inserted" );
                last_inserted = pib.second;
            }
        } else if( threadn == 1 ) { // Fill even keys backward (single thread)
            bool last_inserted = true;
            for( int i = items-2; i >= 0; i-=2 ) {
                pairIB pib = table.insert(Value<T>::make(my_asymptotic?1:i));
                ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic?1:i), "Element not properly inserted");
                ASSERT( last_inserted || !pib.second, "Previous key was not inserted but this is inserted" );
                last_inserted = pib.second;
            }
        } else if( !(threadn&1) ) { // Fill odd keys forward (multiple threads)
            for( int i = 1; i < items; i+=2 )
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
                if ( i % 32 == 1 && i + 6 < items ) {
                    if (my_asymptotic) {
                        table.insert({ Value<T>::make(1), Value<T>::make(1), Value<T>::make(1) });
                        ASSERT(Value<T>::get(*table.find(1)) == 1, "Element not properly inserted");
                    }
                    else {
                        table.insert({ Value<T>::make(i), Value<T>::make(i + 2), Value<T>::make(i + 4) });
                        ASSERT(Value<T>::get(*table.find(i)) == i, "Element not properly inserted");
                        ASSERT(Value<T>::get(*table.find(i + 2)) == i + 2, "Element not properly inserted");
                        ASSERT(Value<T>::get(*table.find(i + 4)) == i + 4, "Element not properly inserted");
                    }
                    i += 4;
                } else
#endif
                {
                    pairIB pib = table.insert(Value<T>::make(my_asymptotic ? 1 : i));
                    ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic ? 1 : i), "Element not properly inserted");
                }
        } else { // Check odd keys backward (multiple threads)
            if (!my_asymptotic) {
                bool last_found = false;
                for( int i = items-1; i >= 0; i-=2 ) {
                    typename T::iterator it = table.find(i);
                    if( it != table.end() ) { // found
                        ASSERT(Value<T>::get(*it) == i, "Element not properly inserted");
                        last_found = true;
                    } else {
                        ASSERT( !last_found, "Previous key was found but this is not" );
                    }
                }
            }
        }
    }
};

typedef tbb::atomic<unsigned char> AtomicByte;

template<typename ContainerType, typename RangeType>
struct ParallelTraverseBody: NoAssign {
    const int n;
    AtomicByte* const array;
    ParallelTraverseBody( AtomicByte an_array[], int a_n ) :
        n(a_n), array(an_array)
    {}
    void operator()( const RangeType& range ) const {
        for( typename RangeType::iterator i = range.begin(); i!=range.end(); ++i ) {
            int k = static_cast<int>(Value<ContainerType>::key(*i));
            ASSERT( k == Value<ContainerType>::get(*i), NULL );
            ASSERT( 0<=k && k<n, NULL );
            array[k]++;
        }
    }
};

// if multimapping, oddCount is the value that each odd-indexed array element should have.
// not meaningful for non-multimapped case.
void CheckRange( AtomicByte array[], int n, bool allowMultiMapping, int oddCount ) {
    if(allowMultiMapping) {
        for( int k = 0; k<n; ++k) {
            if(k%2) {
                if( array[k] != oddCount ) {
                    REPORT("array[%d]=%d (should be %d)\n", k, int(array[k]), oddCount);
                    ASSERT(false,NULL);
                }
            }
            else {
                if(array[k] != 2) {
                    REPORT("array[%d]=%d\n", k, int(array[k]));
                    ASSERT(false,NULL);
                }
            }
        }
    }
    else {
        for( int k=0; k<n; ++k ) {
            if( array[k] != 1 ) {
                REPORT("array[%d]=%d\n", k, int(array[k]));
                ASSERT(false,NULL);
            }
        }
    }
}

template<typename T>
class CheckTable: NoAssign {
    T &table;
public:
    CheckTable(T &t) : NoAssign(), table(t) {}
    void operator()(int i) const {
        int c = (int)table.count( i );
        ASSERT( c, "must exist" );
    }
};

template<typename T>
void test_concurrent_common(const char *tablename, bool asymptotic = false) {
#if TBB_USE_ASSERT
    int items = 2000;
#else
    int items = 20000;
#endif
    int nItemsInserted = 0;
    int nThreads = 0;
#if __TBB_UNORDERED_TEST
    T table(items/1000);
#else
    T table;
#endif
    #if __bgp__
    nThreads = 6;
    #else
    nThreads = 16;
    #endif
    if(T::allow_multimapping) {
        // even passes (threads 0 & 1) put N/2 items each
        // odd passes  (threads > 1)   put N/2 if thread is odd, else checks if even.
        items = 4*items / (nThreads + 2);  // approximately same number of items inserted.
        nItemsInserted = items + (nThreads-2) * items / 4;
    }
    else {
        nItemsInserted = items;
    }
    REMARK("%s items == %d\n", tablename, items);
    tbb::tick_count t0 = tbb::tick_count::now();
    NativeParallelFor( nThreads, FillTable<T>(table, items, asymptotic) );
    tbb::tick_count t1 = tbb::tick_count::now();
    REMARK( "time for filling '%s' by %d items = %g\n", tablename, table.size(), (t1-t0).seconds() );
    ASSERT( int(table.size()) == nItemsInserted, NULL);

    if(!asymptotic) {
        AtomicByte* array = new AtomicByte[items];
        memset( static_cast<void*>(array), 0, items*sizeof(AtomicByte) );

        typename T::range_type r = table.range();
        std::pair<intptr_t,intptr_t> p = CheckRecursiveRange<T,typename T::iterator>(r);
        ASSERT((nItemsInserted == p.first), NULL);
        tbb::parallel_for( r, ParallelTraverseBody<T, typename T::const_range_type>( array, items ));
        CheckRange( array, items, T::allow_multimapping, (nThreads - 1)/2 );

        const T &const_table = table;
        memset( static_cast<void*>(array), 0, items*sizeof(AtomicByte) );
        typename T::const_range_type cr = const_table.range();
        ASSERT((nItemsInserted == CheckRecursiveRange<T,typename T::const_iterator>(cr).first), NULL);
        tbb::parallel_for( cr, ParallelTraverseBody<T, typename T::const_range_type>( array, items ));
        CheckRange( array, items, T::allow_multimapping, (nThreads - 1) / 2 );
        delete[] array;

        tbb::parallel_for( 0, items, CheckTable<T>( table ) );
    }

    table.clear();
    CheckEmptyContainerAllocatorA(table, items+1, items); // one dummy is always allocated

}

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include "test_container_move_support.h"

template<typename container_traits>
void test_rvalue_ref_support(const char* container_name){
    TestMoveConstructor<container_traits>();
    TestMoveAssignOperator<container_traits>();
#if TBB_USE_EXCEPTIONS
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure<container_traits>();
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor<container_traits>();
#endif //TBB_USE_EXCEPTIONS
    REMARK("passed -- %s move support tests\n", container_name);
}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

namespace test_select_size_t_constant{
    __TBB_STATIC_ASSERT((tbb::internal::select_size_t_constant<1234,1234>::value == 1234),"select_size_t_constant::value is not compile time constant");
//    There will be two constant used in the test 32 bit and 64 bit one.
//    The 64 bit constant should chosen so that it 32 bit halves adds up to the 32 bit one ( first constant used in the test).
//    % ~0U is used to sum up 32bit halves of the 64 constant.  ("% ~0U" essentially adds the 32-bit "digits", like "%9" adds
//    the digits (modulo 9) of a number in base 10).
//    So iff select_size_t_constant is correct result of the calculation below will be same on both 32bit and 64bit platforms.
    __TBB_STATIC_ASSERT((tbb::internal::select_size_t_constant<0x12345678U,0x091A2B3C091A2B3CULL>::value % ~0U == 0x12345678U),
            "select_size_t_constant have chosen the wrong constant");
}

#if __TBB_CPP11_SMART_POINTERS_PRESENT
// For the sake of simplified testing, make unique_ptr implicitly convertible to/from the pointer
namespace test {
    template<typename T>
    class unique_ptr : public std::unique_ptr<T> {
    public:
        typedef typename std::unique_ptr<T>::pointer pointer;
        unique_ptr( pointer p ) : std::unique_ptr<T>(p) {}
        operator pointer() const { return this->get(); }
    };
}
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

#include <vector>
#include <list>
#include <algorithm>

template <typename ValueType>
class TestRange : NoAssign {
    const std::list<ValueType> &my_lst;
    std::vector< tbb::atomic<bool> > &my_marks;
public:
    TestRange( const std::list<ValueType> &lst, std::vector< tbb::atomic<bool> > &marks ) : my_lst( lst ), my_marks( marks ) {
        std::fill( my_marks.begin(), my_marks.end(), false );
    }
    template <typename Range>
    void operator()( const Range &r ) const { doTestRange( r.begin(), r.end() ); }
    template<typename Iterator>
    void doTestRange( Iterator i, Iterator j ) const {
        for ( Iterator it = i; it != j; ) {
            Iterator prev_it = it++;
            typename std::list<ValueType>::const_iterator it2 = std::search( my_lst.begin(), my_lst.end(), prev_it, it, Harness::IsEqual() );
            ASSERT( it2 != my_lst.end(), NULL );
            typename std::list<ValueType>::difference_type dist = std::distance( my_lst.begin( ), it2 );
            ASSERT( !my_marks[dist], NULL );
            my_marks[dist] = true;
        }
    }
};

// The helper to call a function only when a doCall == true.
template <bool doCall> struct CallIf {
    template<typename FuncType> void operator() ( FuncType func ) const { func(); }
};
template <> struct CallIf<false> {
    template<typename FuncType> void operator()( FuncType ) const {}
};

template <typename Table>
class TestOperatorSquareBrackets : NoAssign {
    typedef typename Table::value_type ValueType;
    Table &my_c;
    const ValueType &my_value;
public:
    TestOperatorSquareBrackets( Table &c, const ValueType &value ) : my_c( c ), my_value( value ) {}
    void operator()() const {
        ASSERT( Harness::IsEqual()(my_c[my_value.first], my_value.second), NULL );
    }
};

template <bool defCtorPresent, typename Table, typename Value>
void TestMapSpecificMethodsImpl(Table &c, const Value &value){
        CallIf<defCtorPresent>()(TestOperatorSquareBrackets<Table>( c, value ));
        ASSERT( Harness::IsEqual()(c.at( value.first ), value.second), NULL );
        const Table &constC = c;
        ASSERT( Harness::IsEqual()(constC.at( value.first ), value.second), NULL );
}

// do nothing for common case
template <bool defCtorPresent, typename Table, typename Value>
void TestMapSpecificMethods( Table&, const Value& ) {}

template <bool defCtorPresent, typename Table>
class CheckValue : NoAssign {
    Table &my_c;
public:
    CheckValue( Table &c ) : my_c( c ) {}
    void operator()( const typename Table::value_type &value ) {
        typedef typename Table::iterator Iterator;
        typedef typename Table::const_iterator ConstIterator;
        const Table &constC = my_c;
        ASSERT( my_c.count( Value<Table>::key( value ) ) == 1, NULL );
        // find
        ASSERT( Harness::IsEqual()(*my_c.find( Value<Table>::key( value ) ), value), NULL );
        ASSERT( Harness::IsEqual()(*constC.find( Value<Table>::key( value ) ), value), NULL );
        // erase
        ASSERT( my_c.unsafe_erase( Value<Table>::key( value ) ), NULL );
        ASSERT( my_c.count( Value<Table>::key( value ) ) == 0, NULL );
        // insert
        std::pair<Iterator, bool> res = my_c.insert( value );
        ASSERT( Harness::IsEqual()(*res.first, value), NULL );
        ASSERT( res.second, NULL);
        // erase
        Iterator it = res.first;
        it++;
        ASSERT( my_c.unsafe_erase( res.first ) == it, NULL );
        // insert
        ASSERT( Harness::IsEqual()(*my_c.insert( my_c.begin(), value ), value), NULL );
        // equal_range
        std::pair<Iterator, Iterator> r1 = my_c.equal_range( Value<Table>::key( value ) );
        ASSERT( Harness::IsEqual()(*r1.first, value) && ++r1.first == r1.second, NULL );
        std::pair<ConstIterator, ConstIterator> r2 = constC.equal_range( Value<Table>::key( value ) );
        ASSERT( Harness::IsEqual()(*r2.first, value) && ++r2.first == r2.second, NULL );

        TestMapSpecificMethods<defCtorPresent>( my_c, value );
    }
};

#include "tbb/task_scheduler_init.h"

template <bool defCtorPresent, typename Table>
void CommonExamine( Table c, const std::list<typename Table::value_type> lst) {
    typedef typename Table::value_type ValueType;

    ASSERT( !c.empty() && c.size() == lst.size() && c.max_size() >= c.size(), NULL );

    std::for_each( lst.begin(), lst.end(), CheckValue<defCtorPresent, Table>( c ) );

    std::vector< tbb::atomic<bool> > marks( lst.size() );

    TestRange<ValueType>( lst, marks ).doTestRange( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    TestRange<ValueType>( lst, marks ).doTestRange( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    const Table constC = c;
    ASSERT( c.size() == constC.size(), NULL );

    TestRange<ValueType>( lst, marks ).doTestRange( constC.cbegin(), constC.cend() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    tbb::task_scheduler_init init;

    tbb::parallel_for( c.range(), TestRange<ValueType>( lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    tbb::parallel_for( constC.range( ), TestRange<ValueType>( lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    Table c2;
    typename std::list<ValueType>::const_iterator begin5 = lst.begin();
    std::advance( begin5, 5 );
    c2.insert( lst.begin(), begin5 );
    std::for_each( lst.begin(), begin5, CheckValue<defCtorPresent, Table>( c2 ) );

    c2.swap( c );
    ASSERT( c2.size() == lst.size(), NULL );
    ASSERT( c.size() == 5, NULL );
    std::for_each( lst.begin(), lst.end(), CheckValue<defCtorPresent, Table>( c2 ) );

    c2.clear();
    ASSERT( c2.size() == 0, NULL );

    typename Table::allocator_type a = c.get_allocator();
    ValueType *ptr = a.allocate( 1 );
    ASSERT( ptr, NULL );
    a.deallocate( ptr, 1 );
}

// overload for set and multiset
// second argument is needed just for right deduction
template <typename Checker>
void TestSetCommonTypes() {
    Checker CheckTypes;
    const int NUMBER = 10;

    std::list<int> arrInt;
    for ( int i = 0; i<NUMBER; ++i ) arrInt.push_back( i );
    CheckTypes.template check</*defCtorPresent = */true>( arrInt );

    std::list< tbb::atomic<int> > arrTbb(NUMBER);
    int seq = 0;
    for ( std::list< tbb::atomic<int> >::iterator it = arrTbb.begin(); it != arrTbb.end(); ++it, ++seq ) *it = seq;
    CheckTypes.template check</*defCtorPresent = */true>( arrTbb );

#if __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN
    std::list< std::reference_wrapper<int> > arrRef;
    for ( std::list<int>::iterator it = arrInt.begin( ); it != arrInt.end( ); ++it )
        arrRef.push_back( std::reference_wrapper<int>(*it) );
    CheckTypes.template check</*defCtorPresent = */false>( arrRef );
#endif /* __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN */

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    std::list< std::shared_ptr<int> > arrShr;
    for ( int i = 0; i<NUMBER; ++i ) arrShr.push_back( std::make_shared<int>( i ) );
    CheckTypes.template check</*defCtorPresent = */true>( arrShr );

    std::list< std::weak_ptr<int> > arrWk;
    std::copy( arrShr.begin( ), arrShr.end( ), std::back_inserter( arrWk ) );
    CheckTypes.template check</*defCtorPresent = */true>( arrWk );
#else
    REPORT( "Known issue: C++11 smart pointer tests are skipped.\n" );
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}

template <typename Checker>
void TestMapCommonTypes() {
    Checker CheckTypes;
    const int NUMBER = 10;

    std::list< std::pair<const int, int> > arrIntInt;
    for ( int i = 0; i < NUMBER; ++i ) arrIntInt.push_back( std::make_pair( i, NUMBER - i ) );
    CheckTypes.template check</*def_ctor_present = */true>( arrIntInt );

    std::list< std::pair< const int, tbb::atomic<int> > > arrIntTbb;
    for ( int i = 0; i < NUMBER; ++i ) {
        tbb::atomic<int> b;
        b = NUMBER - i;
        arrIntTbb.push_back( std::make_pair( i, b ) );
    }
    CheckTypes.template check</*defCtorPresent = */true>( arrIntTbb );

#if __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN
    std::list< std::pair<const std::reference_wrapper<const int>, int> > arrRefInt;
    for ( std::list< std::pair<const int, int> >::iterator it = arrIntInt.begin(); it != arrIntInt.end(); ++it )
        arrRefInt.push_back( std::make_pair( std::reference_wrapper<const int>( it->first ), it->second ) );
    CheckTypes.template check</*defCtorPresent = */true>( arrRefInt );

    std::list< std::pair<const int, std::reference_wrapper<int> > > arrIntRef;
    for ( std::list< std::pair<const int, int> >::iterator it = arrIntInt.begin(); it != arrIntInt.end(); ++it ) {
        // Using std::make_pair below causes compilation issues with early implementations of std::reference_wrapper.
        arrIntRef.push_back( std::pair<const int, std::reference_wrapper<int> >( it->first, std::reference_wrapper<int>( it->second ) ) );
    }
    CheckTypes.template check</*defCtorPresent = */false>( arrIntRef );
#endif /* __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN */

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    std::list< std::pair< const std::shared_ptr<int>, std::shared_ptr<int> > > arrShrShr;
    for ( int i = 0; i < NUMBER; ++i ) {
        const int NUMBER_minus_i = NUMBER - i;
        arrShrShr.push_back( std::make_pair( std::make_shared<int>( i ), std::make_shared<int>( NUMBER_minus_i ) ) );
    }
    CheckTypes.template check</*defCtorPresent = */true>( arrShrShr );

    std::list< std::pair< const std::weak_ptr<int>, std::weak_ptr<int> > > arrWkWk;
    std::copy( arrShrShr.begin(), arrShrShr.end(), std::back_inserter( arrWkWk ) );
    CheckTypes.template check</*defCtorPresent = */true>( arrWkWk );

#else
    REPORT( "Known issue: C++11 smart pointer tests are skipped.\n" );
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}


#if __TBB_UNORDERED_NODE_HANDLE_PRESENT || __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT
namespace node_handling{
    template<typename Handle>
    bool compare_handle_getters(
        const Handle& node, const std::pair<typename Handle::key_type, typename Handle::mapped_type>& expected
    ) {
        return node.key() == expected.first && node.mapped() == expected.second;
    }

    template<typename Handle>
    bool compare_handle_getters( const Handle& node, const typename Handle::value_type& value) {
        return node.value() == value;
    }

    template<typename Handle>
    void set_node_handle_value(
        Handle& node, const std::pair<typename Handle::key_type, typename Handle::mapped_type>& value
    ) {
        node.key() = value.first;
        node.mapped() = value.second;
    }

    template<typename Handle>
    void set_node_handle_value( Handle& node, const typename Handle::value_type& value) {
        node.value() = value;
    }

    template <typename node_type>
    void TestTraits() {
        ASSERT( !std::is_copy_constructible<node_type>::value,
                "Node handle: Handle is copy constructable" );
        ASSERT( !std::is_copy_assignable<node_type>::value,
                "Node handle: Handle is copy assignable" );
        ASSERT( std::is_move_constructible<node_type>::value,
                "Node handle: Handle is not move constructable" );
        ASSERT( std::is_move_assignable<node_type>::value,
                "Node handle: Handle is not move constructable" );
        ASSERT( std::is_default_constructible<node_type>::value,
                "Node handle:  Handle is not default constructable" );
        ASSERT( std::is_destructible<node_type>::value,
                "Node handle: Handle is not destructible" );
    }

    template <typename Table>
    void TestHandle( Table test_table ) {
        ASSERT( test_table.size()>1, "Node handle: Container must contains 2 or more elements" );
        // Initialization
        using node_type = typename Table::node_type;

        TestTraits<node_type>();

        // Default Ctor and empty function
        node_type nh;
        ASSERT( nh.empty(), "Node handle: Node is not empty after initialization" );

        // Move Assign
        // key/mapped/value function
        auto expected_value = *test_table.begin();

        nh = test_table.unsafe_extract(test_table.begin());
        ASSERT( !nh.empty(), "Node handle: Node handle is empty after valid move assigning" );
        ASSERT( compare_handle_getters(nh,expected_value),
                "Node handle: After valid move assigning "
                "node handle does not contains expected value");

        // Move Ctor
        // key/mapped/value function
        node_type nh2(std::move(nh));
        ASSERT( nh.empty(), "Node handle: After valid move construction node handle is empty" );
        ASSERT( !nh2.empty(), "Node handle: After valid move construction "
                              "argument hode handle was not moved" );
        ASSERT( compare_handle_getters(nh2,expected_value),
                "Node handle: After valid move construction "
                "node handle does not contains expected value" );

        // Bool conversion
        ASSERT( nh2, "Node handle: Wrong not handle bool conversion" );

        // Change key/mapped/value of node handle
        auto expected_value2 = *test_table.begin();
        set_node_handle_value(nh2, expected_value2);
        ASSERT( compare_handle_getters(nh2, expected_value2),
                "Node handle: Wrong node handle key/mapped/value changing behavior" );

        // Member/non member swap check
        node_type empty_node;
        // We extract this element for nh2 and nh3 difference
        test_table.unsafe_extract(test_table.begin());
        auto expected_value3 =  *test_table.begin();
        node_type nh3(test_table.unsafe_extract(test_table.begin()));

        // Both of node handles are not empty
        nh3.swap(nh2);
        ASSERT( compare_handle_getters(nh3, expected_value2),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( compare_handle_getters(nh2, expected_value3),
                "Node handle: Wrong node handle swap behavior" );

        std::swap(nh2,nh3);
        ASSERT( compare_handle_getters(nh3, expected_value3),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( compare_handle_getters(nh2, expected_value2),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( !nh2.empty(), "Node handle: Wrong node handle swap behavior" );
        ASSERT( !nh3.empty(), "Node handle: Wrong node handle swap behavior" );

        // One of nodes is empty
        nh3.swap(empty_node);
        ASSERT( compare_handle_getters(std::move(empty_node), expected_value3),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( nh3.empty(), "Node handle: Wrong node handle swap behavior" );

        std::swap(empty_node, nh3);
        ASSERT( compare_handle_getters(std::move(nh3), expected_value3),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( empty_node.empty(), "Node handle: Wrong node handle swap behavior" );

        empty_node.swap(nh3);
        ASSERT( compare_handle_getters(std::move(empty_node), expected_value3),
                "Node handle: Wrong node handle swap behavior" );
        ASSERT( nh3.empty(), "Node handle: Wrong node handle swap behavior" );
    }

    template <typename Table>
    typename Table::node_type GenerateNodeHandle(const typename Table::value_type& value) {
        Table temp_table;
        temp_table.insert(value);
        return temp_table.unsafe_extract(temp_table.cbegin());
    }

    template <typename Table>
    void IteratorAssertion( const Table& table,
                            const typename Table::iterator& result,
                            const typename Table::value_type* node_value = nullptr ) {
        if (node_value==nullptr) {
            ASSERT( result==table.end(), "Insert: Result iterator does not "
                                         "contains end pointer after empty node insertion" );
        } else {
            if (!Table::allow_multimapping) {
                ASSERT( result==table.find(Value<Table>::key( *node_value )) &&
                        result != table.end(),
                        "Insert: After node insertion result iterator"
                        " doesn't contains address to equal element in table" );
            } else {
                ASSERT( *result==*node_value, "Insert: Result iterator contains"
                                              "wrong content after successful insertion" );

                for (auto it = table.begin(); it != table.end(); ++it) {
                    if (it == result) return;
                }
                ASSERT( false, "Insert: After successful insertion result "
                               "iterator contains address that is not in the table" );
            }
        }
    }
    // overload for multitable or insertion with hint iterator
    template <typename Table>
    void InsertAssertion( const Table& table,
                          const typename Table::iterator& result,
                          bool,
                          const typename Table::value_type* node_value = nullptr ) {
        IteratorAssertion(table, result, node_value);
    }

    // Not multitable overload
    template <typename Table>
    void InsertAssertion( const Table& table,
                          const std::pair<typename Table::iterator, bool>& result,
                          bool second_value,
                          const typename Table::value_type* node_value = nullptr ) {
        IteratorAssertion(table, result.first, node_value);

        ASSERT( result.second == second_value || Table::allow_multimapping,
                "Insert: Returned bool wrong value after node insertion" );
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    // Internal func for testing
    // Can't delete ref from "Table" argument because hint must point to element of table
    namespace {
        template <typename Table, typename... Hint>
        void TestInsertOverloads( Table& table_to_insert,
                                  const typename Table::value_type &value, const Hint&... hint ) {
            // Insert empty element
            typename Table::node_type nh;

            auto table_size = table_to_insert.size();
            auto result = table_to_insert.insert(hint..., std::move(nh));
            InsertAssertion(table_to_insert, result, /*second_value*/ false);
            ASSERT( table_to_insert.size() == table_size,
                    "Insert: After empty node insertion table size changed" );

            // Standard insertion
            nh = GenerateNodeHandle<Table>(value);

            result = table_to_insert.insert(hint..., std::move(nh));
            ASSERT( nh.empty(), "Insert: Not empty handle after successful insertion" );
            InsertAssertion(table_to_insert, result, /*second_value*/ true, &value);

            // Insert existing node
            nh = GenerateNodeHandle<Table>(value);

            result = table_to_insert.insert(hint..., std::move(nh));

            InsertAssertion(table_to_insert, result, /*second_value*/ false, &value);

            if (Table::allow_multimapping){
                ASSERT( nh.empty(), "Insert: Failed insertion to multitable" );
            } else {
                ASSERT( !nh.empty() , "Insert: Empty handle after failed insertion" );
                ASSERT( compare_handle_getters( std::move(nh), value ),
                        "Insert: Existing data does not equal to the one being inserted" );
            }
        }
    }

    template <typename Table>
    void TestInsert( Table table, const typename Table::value_type & value) {
        ASSERT( !table.empty(), "Insert: Map should contains 1 or more elements" );
        Table table_backup(table);
        TestInsertOverloads(table, value);
        TestInsertOverloads(table_backup, value, table_backup.begin());
    }
#endif /*__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT*/

    template <typename Table>
    void TestExtract( Table table_for_extract, typename Table::key_type new_key ) {
        ASSERT( table_for_extract.size()>1, "Extract: Container must contains 2 or more element" );
        ASSERT( table_for_extract.find(new_key)==table_for_extract.end(),
                "Extract: Table must not contains new element!");

        // Extract new element
        auto nh = table_for_extract.unsafe_extract(new_key);
        ASSERT( nh.empty(), "Extract: Node handle is not empty after wrong key extraction" );

        // Valid key extraction
        auto expected_value = *table_for_extract.cbegin();
        auto key = Value<Table>::key( expected_value );
        auto count = table_for_extract.count(key);

        nh = table_for_extract.unsafe_extract(key);
        ASSERT( !nh.empty(),
                "Extract: After successful extraction by key node handle is empty" );
        ASSERT( compare_handle_getters(std::move(nh), expected_value),
                "Extract: After successful extraction by key node handle contains wrong value" );
        ASSERT( table_for_extract.count(key) == count - 1,
                "Extract: After successful node extraction by key, table still contains this key" );

        // Valid iterator overload
        auto expected_value2 = *table_for_extract.cbegin();
        auto key2 = Value<Table>::key( expected_value2 );
        auto count2 = table_for_extract.count(key2);

        nh = table_for_extract.unsafe_extract(table_for_extract.cbegin());
        ASSERT( !nh.empty(),
                "Extract: After successful extraction by iterator node handle is empty" );
        ASSERT( compare_handle_getters(std::move(nh), expected_value2),
                "Extract: After successful extraction by iterator node handle contains wrong value" );
        ASSERT( table_for_extract.count(key2) == count2 - 1,
                "Extract: After successful extraction table also contains this element" );
    }

    // All test exclude merge
    template <typename Table>
    void NodeHandlingTests ( const Table& table,
                             const typename Table::value_type& new_value) {
        TestHandle(table);
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
        TestInsert(table, new_value);
#endif /*__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT*/
        TestExtract(table,  Value<Table>::key( new_value ));
    }

    template <typename TableType1, typename TableType2>
    void TestMerge( TableType1 table1, TableType2&& table2 ) {
        using Table2PureType = typename std::decay<TableType2>::type;
        // Initialization
        TableType1 table1_backup = table1;
        // For copying lvalue
        Table2PureType table2_backup = table2;

        table1.merge(std::forward<TableType2>(table2));
        for (auto it: table2) {
            ASSERT( table1.find( Value<Table2PureType>::key( it ) ) != table1.end(),
                    "Merge: Some key(s) was not merged" );
        }

        // After the following step table1 will contains only merged elements from table2
        for (auto it: table1_backup) {
            table1.unsafe_extract(Value<TableType1>::key( it ));
        }
        // After the following step table2_backup will contains only merged elements from table2
         for (auto it: table2) {
            table2_backup.unsafe_extract(Value<Table2PureType>::key( it ));
        }

        ASSERT ( table1.size() == table2_backup.size(), "Merge: Size of tables is not equal" );
        for (auto it: table2_backup) {
            ASSERT( table1.find( Value<Table2PureType>::key( it ) ) != table1.end(),
                    "Merge: Wrong merge behavior" );
        }
    }

    // Testing of rvalue and lvalue overloads
    template <typename TableType1, typename TableType2>
    void TestMergeOverloads( const TableType1& table1, TableType2 table2 ) {
        TableType2 table_backup(table2);
        TestMerge(table1, table2);
        TestMerge(table1, std::move(table_backup));
    }

    template <typename Table, typename MultiTable>
    void TestMergeTransposition( Table table1, Table table2,
                                 MultiTable multitable1, MultiTable multitable2 ) {
        Table empty_map;
        MultiTable empty_multimap;

        // Map transpositions
        node_handling::TestMergeOverloads(table1, table2);
        node_handling::TestMergeOverloads(table1, empty_map);
        node_handling::TestMergeOverloads(empty_map, table2);

        // Multimap transpositions
        node_handling::TestMergeOverloads(multitable1, multitable2);
        node_handling::TestMergeOverloads(multitable1, empty_multimap);
        node_handling::TestMergeOverloads(empty_multimap, multitable2);

        // Map/Multimap transposition
        node_handling::TestMergeOverloads(table1, multitable1);
        node_handling::TestMergeOverloads(multitable2, table2);
    }

    template <typename SrcTableType, typename DestTableType>
    void AssertionConcurrentMerge ( SrcTableType& start_data, DestTableType& dest_table,
                                    std::vector<SrcTableType>& src_tables, std::true_type) {
        ASSERT( dest_table.size() == start_data.size() * src_tables.size(),
                "Merge: Incorrect merge for some elements" );

        for(auto it: start_data) {
            ASSERT( dest_table.count( Value<DestTableType>::key( it ) ) ==
                    start_data.count( Value<SrcTableType>::key( it ) ) * src_tables.size(),
                                      "Merge: Incorrect merge for some element" );
        }

        for (size_t i = 0; i < src_tables.size(); i++) {
            ASSERT( src_tables[i].empty(), "Merge: Some elements were not merged" );
        }
    }

    template <typename SrcTableType, typename DestTableType>
    void AssertionConcurrentMerge ( SrcTableType& start_data, DestTableType& dest_table,
                                    std::vector<SrcTableType>& src_tables, std::false_type) {
        SrcTableType expected_result;
        for (auto table: src_tables)
            for (auto it: start_data) {
                // If we cannot find some element in some table, then it has been moved
                if (table.find( Value<SrcTableType>::key( it ) ) == table.end()){
                    bool result = expected_result.insert( it ).second;
                    ASSERT( result, "Merge: Some element was merged twice or was not "
                                    "returned to his owner after unsuccessful merge");
                }
            }

        ASSERT( expected_result.size() == dest_table.size() && start_data.size() == dest_table.size(),
                "Merge: wrong size of result table");
        for (auto it: expected_result) {
            if ( dest_table.find( Value<SrcTableType>::key( it ) ) != dest_table.end() &&
                 start_data.find( Value<DestTableType>::key( it ) ) != start_data.end() ){
                dest_table.unsafe_extract(Value<SrcTableType>::key( it ));
                start_data.unsafe_extract(Value<DestTableType>::key( it ));
            } else {
                ASSERT( false, "Merge: Incorrect merge for some element" );
            }
        }

        ASSERT( dest_table.empty()&&start_data.empty(), "Merge: Some elements were not merged" );
    }

    template <typename SrcTableType, typename DestTableType>
    void TestConcurrentMerge (SrcTableType table_data) {
        for (auto num_threads = MinThread + 1; num_threads <= MaxThread; num_threads++){
            std::vector<SrcTableType> src_tables;
            DestTableType dest_table;

            for (auto j = 0; j < num_threads; j++){
                src_tables.push_back(table_data);
            }

            NativeParallelFor( num_threads, [&](size_t index){ dest_table.merge(src_tables[index]); } );

            AssertionConcurrentMerge( table_data, dest_table, src_tables,
                                      std::integral_constant<bool, DestTableType::allow_multimapping>{});
        }
    }

    template <typename Table>
    void TestNodeHandling(){
        Table table;

        for (int i = 1; i < 5; i++)
            table.insert(Value<Table>::make(i));

        if (Table::allow_multimapping)
            table.insert(Value<Table>::make(4));

        node_handling::NodeHandlingTests(table, Value<Table>::make(5));
    }

    template <typename TableType1, typename TableType2>
    void TestMerge(int size){
        TableType1 table1_1;
        TableType1 table1_2;
        int i = 1;
        for (; i < 5; ++i) {
            table1_1.insert(Value<TableType1>::make(i));
            table1_2.insert(Value<TableType1>::make(i*i));
        }
        if (TableType1::allow_multimapping) {
            table1_1.insert(Value<TableType1>::make(i));
            table1_2.insert(Value<TableType1>::make(i*i));
        }

        TableType2 table2_1;
        TableType2 table2_2;
        for (i = 3; i < 7; ++i) {
            table1_1.insert(Value<TableType2>::make(i));
            table1_2.insert(Value<TableType2>::make(i*i));
        }
        if (TableType2::allow_multimapping) {
            table2_1.insert(Value<TableType2>::make(i));
            table2_2.insert(Value<TableType2>::make(i*i));
        }

        node_handling::TestMergeTransposition(table1_1, table1_2,
                                              table2_1, table2_2);

        TableType1 table1_3;
        for (i = 0; i<size; ++i){
            table1_3.insert(Value<TableType1>::make(i));
        }
        node_handling::TestConcurrentMerge<TableType1, TableType2>(table1_3);

        TableType2 table2_3;
        for (i = 0; i<size; ++i){
            table2_3.insert(Value<TableType2>::make(i));
        }
        node_handling::TestConcurrentMerge<TableType2, TableType1>(table2_3);
}
}
#endif // __TBB_UNORDERED_NODE_HANDLE_PRESENT || __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT
