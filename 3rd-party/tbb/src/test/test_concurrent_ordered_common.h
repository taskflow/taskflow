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

#include "test_concurrent_associative_common.h"

// Now empty ordered container allocations count is checked by upper bound (calculated manually)
const size_t dummy_head_max_size = 584;

template<typename MyTable>
inline void CheckEmptyContainerAllocator(MyTable &table, size_t expected_allocs, size_t expected_frees, bool exact, int line) {
    typename MyTable::allocator_type a = table.get_allocator();
    REMARK("#%d checking allocators: items %u/%u, allocs %u/%u\n", line,
        unsigned(a.items_allocated), unsigned(a.items_freed), unsigned(a.allocations), unsigned(a.frees) );
    CheckAllocator<MyTable>(a, expected_allocs, expected_frees, exact);
    ASSERT( a.items_allocated <= a.items_freed + dummy_head_max_size, NULL);
}

template <typename Table>
struct order_checker {
    typename Table::value_compare& val_comp;
    typename Table::key_compare& key_comp;

    order_checker(typename Table::value_compare& _val_c,typename Table::key_compare& _key_c): val_comp(_val_c), key_comp(_key_c){}


    bool operator()(const typename Table::value_type& lhs, const typename Table::value_type& rhs){
        if (Table::allow_multimapping)
            // We need to use not greater comparator for multicontainers
            return !val_comp(rhs, lhs) && !key_comp(Value<Table>::key(rhs), Value<Table>::key(lhs));
        return val_comp(lhs,rhs) && key_comp(Value<Table>::key(lhs),Value<Table>::key(rhs));
    }
};

template< typename Table>
void check_container_order(const Table& cont) {
    if (!cont.empty()){
        typename Table::key_compare key_comp = cont.key_comp();
        typename Table::value_compare value_comp = cont.value_comp();
        order_checker<Table> check_order(value_comp, key_comp);

        for (auto it = cont.begin(); std::next(it)!=cont.end();){
            auto pr_it = it++;
            ASSERT(check_order(*pr_it, *it),"The order of the elements is broken");
        }
    }
}

template <typename T>
void test_ordered_methods() {
    T cont;

    int r, random_threshold = 10, uncontained_key = random_threshold / 2;
    for (int i = 0; i < 100; i++) {
        r = std::rand() % random_threshold;
        if ( r != uncontained_key) {
            cont.insert(Value<T>::make(r));
        }
    }

    check_container_order(cont);

    typename T::value_compare val_comp = cont.value_comp();
    typename T::iterator l_bound_check, u_bound_check;
    for (int key = -1; key < random_threshold + 1; key++) {

        auto eq_range = cont.equal_range(key);
        // Check equal_range() content
        for (auto it = eq_range.first; it != eq_range.second; it++)
            ASSERT(*it == Value<T>::make(key), "equal_range() contain wrong value");

        // Manual search of upper and lower bounds
        l_bound_check = cont.end();
        u_bound_check = cont.end();
        for (auto it = cont.begin() ; it != cont.end(); it++){
            if (!val_comp(*it, Value<T>::make(key)) && l_bound_check == cont.end()){
                l_bound_check = it;
            }
            if (val_comp(Value<T>::make(key),*it) && u_bound_check == cont.end()){
                u_bound_check = it;
                break;
            }
        }

        typename T::iterator l_bound = cont.lower_bound(key);
        typename T::iterator u_bound = cont.upper_bound(key);

        ASSERT(l_bound == l_bound_check, "lower_bound() contains wrong value");
        ASSERT(u_bound == u_bound_check, "upper_bound() contains wrong value");

        ASSERT(l_bound == eq_range.first && u_bound == eq_range.second, NULL);
    }
}

template<typename T, typename do_check_element_state>
void test_basic(const char * str, do_check_element_state)
{
    test_basic_common<T>(str, do_check_element_state());
    test_ordered_methods<T>();
}

template<typename T>
void test_basic(const char * str){
    test_basic_common<T>(str);
    test_ordered_methods<T>();
}

template<typename T>
void test_concurrent_order() {
    for (auto num_threads = MinThread + 1; num_threads <= MaxThread; num_threads++) {
        T cont;
        int items = 1000;
        NativeParallelFor( num_threads, [&](size_t index){
            int step = index % 4 + 1;
            bool reverse = (step % 2 == 0);
            if (reverse) {
                for (int i = 0; i < items; i+=step){
                    cont.insert(Value<T>::make(i));
                }
            } else {
                for (int i = items; i > 0; i-=step){
                    cont.insert(Value<T>::make(i));
                }
            }
        } );

        check_container_order(cont);
    }
}

template<typename T>
void test_concurrent(const char *tablename, bool asymptotic = false) {
    test_concurrent_common<T>(tablename, asymptotic);
    test_concurrent_order<T>();
}

// If the inserted elements look the same for the comparator,
// they must be inserted in order from the first inserted to the last.
template<typename T>
void check_multicontainer_internal_order(){
    T cont;
    for (int counter = 0; counter < 10; counter++){
        cont.emplace(1, counter);
    }

    for ( auto it = cont.begin(); std::next(it) != cont.end();){
        auto it_pr = it++;
        ASSERT(it_pr->second < it->second, "Internal multicontainers order is broken");
    }
}

struct ordered_move_traits_base {
    enum{ expected_number_of_items_to_allocate_for_steal_move = dummy_head_max_size };

    template <typename ordered_type, typename iterator_type>
    static ordered_type& construct_container(tbb::aligned_space<ordered_type> & storage, iterator_type begin, iterator_type end){
        new (storage.begin()) ordered_type(begin, end);
        return * storage.begin();
    }

    template <typename ordered_type, typename iterator_type, typename allocator_type>
    static ordered_type& construct_container(tbb::aligned_space<ordered_type> & storage, iterator_type begin, iterator_type end, allocator_type const& a ){
        new (storage.begin()) ordered_type(begin, end, typename ordered_type::key_compare(), a);
        return * storage.begin();
    }

    template<typename ordered_type, typename iterator>
    static bool equal(ordered_type const& c, iterator begin, iterator end){
        bool equal_sizes = ( static_cast<size_t>(std::distance(begin, end)) == c.size() );
        if (!equal_sizes)
            return false;
        for (iterator it = begin; it != end; ++it ){
            if (c.find( Value<ordered_type>::key(*it)) == c.end()){
                return false;
            }
        }
        return true;
    }
};

namespace std {
    template<> struct less< std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& lhs, const std::weak_ptr<int>& rhs ) const { return *lhs.lock() < * rhs.lock(); }
    };
    template<> struct less< const std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& lhs, const std::weak_ptr<int>& rhs ) const { return *lhs.lock() < * rhs.lock(); }
    };
}

template <bool defCtorPresent, typename Table>
void CustomExamine( Table, const std::list<typename Table::value_type>) {
    /*order check - see unordered example*/
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

    typename Table::key_compare compare;

    typename Table::allocator_type allocator;
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    // Constructor from an initializer_list.
    typename std::list<typename Table::value_type>::const_iterator it = lst.begin();
    Table c2( { *it++, *it++, *it++ } );
    c2.insert( it, lst.end( ) );
    Examine<defCtorPresent>( c2, lst );

    it = lst.begin();
    // Constructor from an initializer_list, default comparator and non-default allocator
    Table c2_alloc( { *it++, *it++, *it++ }, allocator);
    c2_alloc.insert( it, lst.end() );
    Examine<defCtorPresent>( c2_alloc, lst );

    it = lst.begin();
    // Constructor from an initializer_list, non-default comparator and allocator
    Table c2_comp_alloc( { *it++, *it++, *it++ }, compare, allocator );
    c2_comp_alloc.insert( it, lst.end() );
    Examine<defCtorPresent>( c2_comp_alloc, lst );
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

    // Construction empty table with non-default comparator
    Table c6( compare );
    c6.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6, lst );

    // Construction empty table with non-default allocator
    Table c6_alloc( allocator );
    c6_alloc.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6_alloc, lst );

    // Construction empty table with a non-default comparator and allocator
    Table c6_comp_alloc( compare, allocator );
    c6_comp_alloc.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6_alloc, lst );

    // Construction empty table with a  non-default comparator and allocator
    TableDebugAlloc c7( compare );
    c7.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c7, lst );

    // Construction with a copying iteration range and a given allocator instance.
    Table c8( c1.begin(), c1.end() );
    Examine<defCtorPresent>( c8, lst );

    // Construction with a copying iteration range, default compare and non-default allocator
    Table c8_alloc( c1.begin(), c1.end(), allocator );
    Examine<defCtorPresent>( c8_alloc, lst );

    // Construction with a copying iteration range, non-default compare and allocator
    Table c8_comp_alloc( c1.begin(), c1.end(), compare, allocator );
    Examine<defCtorPresent>( c8_comp_alloc, lst);

    // Construction with an instance of non-default allocator
    typename TableDebugAlloc::allocator_type a;
    TableDebugAlloc c9( a );
    c9.insert( c7.begin(), c7.end() );
    Examine<defCtorPresent>( c9, lst );
}

struct int_key {
        int_key(int i) : my_item(i) {}
        int my_item;
};

bool operator<(const int_key& ik, int i) { return ik.my_item < i; }
bool operator<(int i, const int_key& ik) { return i < ik.my_item; }
bool operator<(const int_key& ik1, const int_key& ik2) { return ik1.my_item < ik2.my_item; }

struct transparent_less {
    template <typename T, typename U>
    auto operator()( T&& lhs, U&& rhs ) const
    -> decltype(std::forward<T>(lhs) < std::forward<U>(rhs)){
        return lhs < rhs;
    }

    using is_transparent = void;
};

template <typename Container>
void check_heterogeneous_functions() {
    static_assert(std::is_same<typename Container::key_type, int>::value,
                  "incorrect key_type for heterogeneous lookup test");
    // Initialization
    Container c;
    int size = 10;
    for (int i = 0; i < size; i++){
        c.insert(Value<Container>::make(i));
    }
    // Insert first duplicated element for multicontainers
    if (Container::allow_multimapping){
        c.insert(Value<Container>::make(0));
    }

    // Look up testing
    for (int i = 0; i < size; i++) {
        int_key k(i);
        int key = i;
        ASSERT(c.find(k) == c.find(key), "Incorrect heterogeneous find return value");
        ASSERT(c.lower_bound(k) == c.lower_bound(key), "Incorrect heterogeneous lower_bound return value");
        ASSERT(c.upper_bound(k) == c.upper_bound(key), "Incorrect heterogeneous upper_bound return value");
        ASSERT(c.equal_range(k) == c.equal_range(key), "Incorrect heterogeneous equal_range return value");
        ASSERT(c.count(k) == c.count(key), "Incorrect heterogeneous count return value");
        ASSERT(c.contains(k) == c.contains(key), "Incorrect heterogeneous contains return value");
    }

    // Erase testing
    for (int i = 0; i < size; i++){
        auto count_before_erase = c.count(i);
        auto result = c.unsafe_erase(int_key(i));
        ASSERT(count_before_erase==result,"Incorrent erased elements count");
        ASSERT(c.count(i)==0, "Some elements was not erased");
    }
}

template <template<typename...> class ContainerType, typename... ContainerArgs>
void test_allocator_traits() {
    using namespace propagating_allocators;
    using always_propagating_container = ContainerType<ContainerArgs..., always_propagating_allocator>;
    using never_propagating_container = ContainerType<ContainerArgs..., never_propagating_allocator>;
    using pocma_container = ContainerType<ContainerArgs..., pocma_allocator>;
    using pocca_container = ContainerType<ContainerArgs..., pocca_allocator>;
    using pocs_container = ContainerType<ContainerArgs..., pocs_allocator>;

    test_allocator_traits_support<always_propagating_container>();
    test_allocator_traits_support<never_propagating_container>();
    test_allocator_traits_support<pocma_container>();
    test_allocator_traits_support<pocca_container>();
    test_allocator_traits_support<pocs_container>();

    test_allocator_traits_with_non_movable_value_type<pocma_container>();
}

// Comparator for scoped_allocator tests
struct allocator_data_compare {
    template <typename A>
    bool operator()(const allocator_aware_data<A>& d1, const allocator_aware_data<A>& d2) const {
        return d1.value() < d2.value();
    }
};
