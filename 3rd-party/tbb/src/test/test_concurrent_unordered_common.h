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

#define __TBB_UNORDERED_TEST 1

#include "test_concurrent_associative_common.h"

template<typename MyTable>
inline void CheckEmptyContainerAllocator(MyTable &table, size_t expected_allocs, size_t expected_frees, bool exact, int line) {
    typename MyTable::allocator_type a = table.get_allocator();
    REMARK("#%d checking allocators: items %u/%u, allocs %u/%u\n", line,
        unsigned(a.items_allocated), unsigned(a.items_freed), unsigned(a.allocations), unsigned(a.frees) );
    ASSERT( a.items_allocated == a.allocations, NULL); ASSERT( a.items_freed == a.frees, NULL);
    ASSERT( a.items_allocated == a.items_freed + 1, NULL);
    CheckAllocator<MyTable>(a, expected_allocs, expected_frees, exact);
}

template<typename T>
struct degenerate_hash {
    size_t operator()(const T& /*a*/) const {
        return 1;
    }
};

template <typename T>
void test_unordered_methods(){
    T cont;
    cont.insert(Value<T>::make(1));
    cont.insert(Value<T>::make(2));
    // unordered_specific
    // void rehash(size_type n);
    cont.rehash(16);

    // float load_factor() const;
    // float max_load_factor() const;
    ASSERT(cont.load_factor() <= cont.max_load_factor(), "Load factor is invalid");

    // void max_load_factor(float z);
    cont.max_load_factor(16.0f);
    ASSERT(cont.max_load_factor() == 16.0f, "Max load factor has not been changed properly");

    // hasher hash_function() const;
    cont.hash_function();

    // key_equal key_eq() const;
    cont.key_eq();

    cont.clear();
    CheckEmptyContainerAllocatorA(cont, 1, 0); // one dummy is always allocated
    for (int i = 0; i < 256; i++)
    {
        std::pair<typename T::iterator, bool> ins3 = cont.insert(Value<T>::make(i));
        ASSERT(ins3.second == true && Value<T>::get(*(ins3.first)) == i, "Element 1 has not been inserted properly");
    }
    ASSERT(cont.size() == 256, "Wrong number of elements have been inserted");
    // size_type unsafe_bucket_count() const;
    ASSERT(cont.unsafe_bucket_count() == 16, "Wrong number of buckets");

    // size_type unsafe_max_bucket_count() const;
    ASSERT(cont.unsafe_max_bucket_count() > 65536, "Wrong max number of buckets");

    for (unsigned int i = 0; i < 256; i++)
    {
        typename T::size_type buck = cont.unsafe_bucket(i);

        // size_type unsafe_bucket(const key_type& k) const;
        ASSERT(buck < 16, "Wrong bucket mapping");
    }

    typename T::size_type bucketSizeSum = 0;
    typename T::size_type iteratorSizeSum = 0;

    for (unsigned int i = 0; i < 16; i++)
    {
        bucketSizeSum += cont.unsafe_bucket_size(i);
        for (typename T::iterator bit = cont.unsafe_begin(i); bit != cont.unsafe_end(i); bit++) iteratorSizeSum++;
    }
    ASSERT(bucketSizeSum == 256, "sum of bucket counts incorrect");
    ASSERT(iteratorSizeSum == 256, "sum of iterator counts incorrect");
}

template<typename T, typename do_check_element_state>
void test_basic(const char * str, do_check_element_state)
{
    test_basic_common<T>(str, do_check_element_state());
    test_unordered_methods<T>();
}

template<typename T>
void test_basic(const char * str){
    test_basic_common<T>(str, tbb::internal::false_type());
    test_unordered_methods<T>();
}

template<typename T>
void test_concurrent(const char *tablename, bool asymptotic = false) {
    test_concurrent_common<T>(tablename, asymptotic);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct unordered_move_traits_base {
    enum{ expected_number_of_items_to_allocate_for_steal_move = 3 };

    template <typename unordered_type, typename iterator_type>
    static unordered_type& construct_container(tbb::aligned_space<unordered_type> & storage, iterator_type begin, iterator_type end){
        new (storage.begin()) unordered_type(begin, end);
        return * storage.begin();
    }

    template <typename unordered_type, typename iterator_type, typename allocator_type>
    static unordered_type& construct_container(tbb::aligned_space<unordered_type> & storage, iterator_type begin, iterator_type end, allocator_type const& a ){
        size_t deault_n_of_buckets = 8; //can not use concurrent_unordered_base::n_of_buckets as it is inaccessible
        new (storage.begin()) unordered_type(begin, end, deault_n_of_buckets, typename unordered_type::hasher(), typename unordered_type::key_equal(), a);
        return * storage.begin();
    }

    template<typename unordered_type, typename iterator>
    static bool equal(unordered_type const& c, iterator begin, iterator end){
        bool equal_sizes = ( static_cast<size_t>(std::distance(begin, end)) == c.size() );
        if (!equal_sizes)
            return false;

        for (iterator it = begin; it != end; ++it ){
            if (c.find( Value<unordered_type>::key(*it)) == c.end()){
                return false;
            }
        }
        return true;
    }
};
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT*/

#if __TBB_CPP11_SMART_POINTERS_PRESENT
namespace tbb {
    template<> class tbb_hash< std::shared_ptr<int> > {
    public:
        size_t operator()( const std::shared_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< const std::shared_ptr<int> > {
    public:
        size_t operator()( const std::shared_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& key ) const { return tbb_hasher( *key.lock( ) ); }
    };
    template<> class tbb_hash< const std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& key ) const { return tbb_hasher( *key.lock( ) ); }
    };
    template<> class tbb_hash< test::unique_ptr<int> > {
    public:
        size_t operator()( const test::unique_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< const test::unique_ptr<int> > {
    public:
        size_t operator()( const test::unique_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
}
#endif /*__TBB_CPP11_SMART_POINTERS_PRESENT*/

template <bool defCtorPresent, typename Table>
void CustomExamine( Table c, const std::list<typename Table::value_type> lst) {
    typedef typename Table::value_type ValueType;
    typedef typename Table::size_type SizeType;
    const Table constC = c;

    const SizeType bucket_count = c.unsafe_bucket_count();
    ASSERT( c.unsafe_max_bucket_count() >= bucket_count, NULL );
    SizeType counter = SizeType( 0 );
    for ( SizeType i = 0; i < bucket_count; ++i ) {
        const SizeType size = c.unsafe_bucket_size( i );
        typedef typename Table::difference_type diff_type;
        ASSERT( std::distance( c.unsafe_begin( i ), c.unsafe_end( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( c.unsafe_cbegin( i ), c.unsafe_cend( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( constC.unsafe_begin( i ), constC.unsafe_end( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( constC.unsafe_cbegin( i ), constC.unsafe_cend( i ) ) == diff_type( size ), NULL );
        counter += size;
    }
    ASSERT( counter == lst.size(), NULL );

    for ( typename std::list<ValueType>::const_iterator it = lst.begin(); it != lst.end(); ) {
        const SizeType index = c.unsafe_bucket( Value<Table>::key( *it ) );
        typename std::list<ValueType>::const_iterator prev_it = it++;
        ASSERT( std::search( c.unsafe_begin( index ), c.unsafe_end( index ), prev_it, it, Harness::IsEqual() ) != c.unsafe_end( index ), NULL );
    }

    c.rehash( 2 * bucket_count );
    ASSERT( c.unsafe_bucket_count() > bucket_count, NULL );

    ASSERT( c.load_factor() <= c.max_load_factor(), NULL );

    c.max_load_factor( 1.0f );
    c.hash_function();
    c.key_eq();
}

template <bool defCtorPresent, typename Table>
void Examine( Table c, const std::list<typename Table::value_type> &lst) {
    CommonExamine<defCtorPresent>(c, lst);
    CustomExamine<defCtorPresent>(c, lst);
}

template <bool defCtorPresent, typename Table, typename TableDebugAlloc>
void TypeTester( const std::list<typename Table::value_type> &lst ) {
    ASSERT( lst.size() >= 5, "Array should have at least 5 elements" );
    ASSERT( lst.size() <= 100, "The test has O(n^2) complexity so a big number of elements can lead long execution time" );
    // Construct an empty table.
    Table c1;
    c1.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c1, lst );

    typename Table::size_type initial_bucket_number = 8;
    typename Table::allocator_type allocator;
    typename Table::hasher hasher;
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    // Constructor from an initializer_list.
    typename std::list<typename Table::value_type>::const_iterator it = lst.begin();
    Table c2( { *it++, *it++, *it++ } );
    c2.insert( it, lst.end( ) );
    Examine<defCtorPresent>( c2, lst );

    it = lst.begin();
    // Constructor from an initializer_list, default hasher and key equality and non-default allocator
    Table c2_alloc( { *it++, *it++, *it++ }, initial_bucket_number, allocator);
    c2_alloc.insert( it, lst.end() );
    Examine<defCtorPresent>( c2_alloc, lst );

    it = lst.begin();
    // Constructor from an initializer_list, default key equality and non-default hasher and allocator
    Table c2_hash_alloc( { *it++, *it++, *it++ }, initial_bucket_number, hasher, allocator );
    c2_hash_alloc.insert( it, lst.end() );
    Examine<defCtorPresent>( c2_hash_alloc, lst );
#endif
    // Copying constructor.
    Table c3( c1 );
    Examine<defCtorPresent>( c3, lst );
    // Construct with non-default allocator
    TableDebugAlloc c4;
    c4.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c4, lst );
    // Copying constructor for a container with a different allocator type.
    TableDebugAlloc c5( c4 );
    Examine<defCtorPresent>( c5, lst );
    // Construction empty table with n preallocated buckets.
    Table c6( lst.size() );
    c6.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6, lst );

    // Construction empty table with n preallocated buckets, default hasher and key equality and non-default allocator
    Table c6_alloc( lst.size(), allocator );
    c6_alloc.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6_alloc, lst );

    // Construction empty table with n preallocated buckets, default key equality and non-default hasher and allocator
    Table c6_hash_alloc( lst.size(), hasher, allocator );
    c6_hash_alloc.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6_hash_alloc, lst );

    TableDebugAlloc c7( lst.size( ) );
    c7.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c7, lst );
    // Construction with a copying iteration range and a given allocator instance.
    Table c8( c1.begin(), c1.end() );
    Examine<defCtorPresent>( c8, lst );

    // Construction with a copying iteration range, default hasher and key equality and non-default allocator
    Table c8_alloc( c1.begin(), c1.end(), initial_bucket_number, allocator );
    Examine<defCtorPresent>( c8_alloc, lst );

    // Construction with a copying iteration range, default key equality and non-default hasher and allocator
    Table c8_hash_alloc( c1.begin(), c1.end(), initial_bucket_number, hasher, allocator );
    Examine<defCtorPresent>( c8_hash_alloc, lst);

    // Construction with an instance of non-default allocator
    typename TableDebugAlloc::allocator_type a;
    TableDebugAlloc c9( a );
    c9.insert( c7.begin(), c7.end() );
    Examine<defCtorPresent>( c9, lst );
}
