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

// have to expose the reset_node method to be able to reset a function_body

#define TBB_USE_SOURCE_NODE_AS_ALIAS __TBB_CPF_BUILD
#define TBB_DEPRECATED_INPUT_NODE_BODY !__TBB_CPF_BUILD

#include "harness.h"

#if __TBB_CPF_BUILD
#define TBB_DEPRECATED_FLOW_NODE_EXTRACTION 1
#endif



#include "harness_graph.h"
#include "tbb/flow_graph.h"
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"

const int N = 1000;

template< typename T >
class test_push_receiver : public tbb::flow::receiver<T>, NoAssign {

    tbb::atomic<int> my_counters[N];
    tbb::flow::graph& my_graph;

public:

    test_push_receiver(tbb::flow::graph& g) : my_graph(g) {
        for (int i = 0; i < N; ++i )
            my_counters[i] = 0;
    }

    int get_count( int i ) {
       int v = my_counters[i];
       return v;
    }

    typedef typename tbb::flow::receiver<T>::predecessor_type predecessor_type;

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
    typedef typename tbb::flow::receiver<T>::built_predecessors_type built_predecessors_type;
    typedef typename tbb::flow::receiver<T>::predecessor_list_type predecessor_list_type;
    built_predecessors_type bpt;
    built_predecessors_type &built_predecessors() __TBB_override { return bpt; }
    void internal_add_built_predecessor( predecessor_type & ) __TBB_override { }
    void internal_delete_built_predecessor( predecessor_type & ) __TBB_override { }
    void copy_predecessors( predecessor_list_type & ) __TBB_override { }
    size_t predecessor_count() __TBB_override { return 0; }
#endif

    tbb::task *try_put_task( const T &v ) __TBB_override {
       int i = (int)v;
       ++my_counters[i];
       return const_cast<tbb::task *>(SUCCESSFULLY_ENQUEUED);
    }

    tbb::flow::graph& graph_reference() const __TBB_override {
        return my_graph;
    }

    void reset_receiver(tbb::flow::reset_flags /*f*/) __TBB_override {}
};

template< typename T >
class source_body {

   unsigned my_count;
   int *ninvocations;

public:

   source_body() : ninvocations(NULL) { my_count = 0; }
   source_body(int &_inv) : ninvocations(&_inv)  { my_count = 0; }

#if TBB_DEPRECATED_INPUT_NODE_BODY
   bool operator()( T &v ) {
      v = (T)my_count++;
      if(ninvocations) ++(*ninvocations);
      if ( (int)v < N )
         return true;
      else
         return false;
   }
#else
    T operator()( tbb::flow_control& fc ) {
        T v = (T)my_count++;
        if(ninvocations) ++(*ninvocations);
        if ( (int)v < N ){
            return v;
        }else{
            fc.stop();
            return T();
        }
    }
#endif

};

template< typename T >
class function_body {

    tbb::atomic<int> *my_counters;

public:

    function_body( tbb::atomic<int> *counters ) : my_counters(counters) {
        for (int i = 0; i < N; ++i )
            my_counters[i] = 0;
    }

    bool operator()( T v ) {
        ++my_counters[(int)v];
        return true;
    }

};

template< typename T >
void test_single_dest() {

   // push only
   tbb::flow::graph g;
   tbb::flow::source_node<T> src(g, source_body<T>() );
   test_push_receiver<T> dest(g);
   tbb::flow::make_edge( src, dest );
#if TBB_USE_SOURCE_NODE_AS_ALIAS
   src.activate();
#endif
   g.wait_for_all();
   for (int i = 0; i < N; ++i ) {
       ASSERT( dest.get_count(i) == 1, NULL );
   }

   // push only
   tbb::atomic<int> counters3[N];
   tbb::flow::source_node<T> src3(g, source_body<T>() );
   function_body<T> b3( counters3 );
   tbb::flow::function_node<T,bool> dest3(g, tbb::flow::unlimited, b3 );
   tbb::flow::make_edge( src3, dest3 );
#if TBB_USE_SOURCE_NODE_AS_ALIAS
   src3.activate();
#endif
   g.wait_for_all();
   for (int i = 0; i < N; ++i ) {
       int v = counters3[i];
       ASSERT( v == 1, NULL );
   }

   // push & pull
   tbb::flow::source_node<T> src2(g, source_body<T>() );
   tbb::atomic<int> counters2[N];
   function_body<T> b2( counters2 );
   tbb::flow::function_node<T,bool,tbb::flow::rejecting> dest2(g, tbb::flow::serial, b2 );
   tbb::flow::make_edge( src2, dest2 );

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
   ASSERT(src2.successor_count() == 1, NULL);
   typename tbb::flow::source_node<T>::successor_list_type my_succs;
   src2.copy_successors(my_succs);
   ASSERT(my_succs.size() == 1, NULL);
#endif

#if TBB_USE_SOURCE_NODE_AS_ALIAS
   src2.activate();
#endif
   g.wait_for_all();
   for (int i = 0; i < N; ++i ) {
       int v = counters2[i];
       ASSERT( v == 1, NULL );
   }

   // test copy constructor
   tbb::flow::source_node<T> src_copy(src);
   test_push_receiver<T> dest_c(g);
   ASSERT( src_copy.register_successor(dest_c), NULL );
#if TBB_USE_SOURCE_NODE_AS_ALIAS
   src_copy.activate();
#endif
   g.wait_for_all();
   for (int i = 0; i < N; ++i ) {
       ASSERT( dest_c.get_count(i) == 1, NULL );
   }
}

void test_reset() {
    //    source_node -> function_node
    tbb::flow::graph g;
    tbb::atomic<int> counters3[N];
    tbb::flow::source_node<int> src3(g, source_body<int>() );
#if TBB_USE_SOURCE_NODE_AS_ALIAS
    tbb::flow::source_node<int> src_inactive(g, source_body<int>() );
#else
    tbb::flow::source_node<int> src_inactive(g, source_body<int>(), /*is_active=*/ false );
#endif
    function_body<int> b3( counters3 );
    tbb::flow::function_node<int,bool> dest3(g, tbb::flow::unlimited, b3 );
    tbb::flow::make_edge( src3, dest3 );

 #if TBB_USE_SOURCE_NODE_AS_ALIAS
    src3.activate();
 #endif
    //    source_node is now in active state.  Let the graph run,
    g.wait_for_all();
    //    check the array for each value.
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 1, NULL );
        counters3[i] = 0;
    }
    g.reset(tbb::flow::rf_reset_bodies);  // <-- re-initializes the counts.
 #if TBB_USE_SOURCE_NODE_AS_ALIAS
    src3.activate();
 #endif
    // and spawns task to run source
    g.wait_for_all();
    //    check output queue again.  Should be the same contents.
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 1, NULL );
        counters3[i] = 0;
    }
    g.reset();  // doesn't reset the source_node_body to initial state, but does spawn a task
                // to run the source_node.

    g.wait_for_all();
    // array should be all zero
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 0, NULL );
    }

    remove_edge(src3, dest3);
    make_edge(src_inactive, dest3);

    // src_inactive doesn't run
    g.wait_for_all();
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 0, NULL );
    }

    // run graph
    src_inactive.activate();
    g.wait_for_all();
    // check output
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 1, NULL );
        counters3[i] = 0;
    }
    g.reset(tbb::flow::rf_reset_bodies);  // <-- reinitializes the counts
    // src_inactive doesn't run
    g.wait_for_all();
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 0, NULL );
    }

    // start it up
    src_inactive.activate();
    g.wait_for_all();
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 1, NULL );
        counters3[i] = 0;
    }
    g.reset();  // doesn't reset the source_node_body to initial state, and doesn't
                // spawn a task to run the source_node.

    g.wait_for_all();
    // array should be all zero
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 0, NULL );
    }
    src_inactive.activate();
    // source_node_body is already in final state, so source_node will not forward a message.
    g.wait_for_all();
    for (int i = 0; i < N; ++i ) {
        int v = counters3[i];
        ASSERT( v == 0, NULL );
    }
}

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
void test_extract() {
    int counts = 0;
    tbb::flow::tuple<int,int> dont_care;
    tbb::flow::graph g;
    typedef tbb::flow::source_node<int> snode_type;
    typedef snode_type::successor_list_type successor_list_type;
#if TBB_USE_SOURCE_NODE_AS_ALIAS
    snode_type s0(g, source_body<int>(counts));
#else
    snode_type s0(g, source_body<int>(counts), false);
#endif
    tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::reserving > j0(g);
    tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::reserving > j1(g);
    tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::reserving > j2(g);
    tbb::flow::queue_node<int> q0(g);
    tbb::flow::queue_node<tbb::flow::tuple<int,int> > q1(g);
    tbb::flow::make_edge(s0, tbb::flow::get<0>(j0.input_ports()));
    /*  s0 ----+    */
    /*         | j0 */
    /*         +    */
    ASSERT(!counts, "source_node activated too soon");
    s0.activate();
    g.wait_for_all();  // should produce one value, buffer it.
    ASSERT(counts == 1, "source_node did not react to activation");

    g.reset(tbb::flow::rf_reset_bodies);
    counts = 0;
    s0.extract();
    /*  s0     +    */
    /*         | j0 */
    /*         +    */
    s0.activate();
    g.wait_for_all();  // no successors, so the body will not execute
    ASSERT(counts == 0, "source_node shouldn't forward (no successors)");
    g.reset(tbb::flow::rf_reset_bodies);

    tbb::flow::make_edge(s0, tbb::flow::get<0>(j0.input_ports()));
    tbb::flow::make_edge(s0, tbb::flow::get<0>(j1.input_ports()));
    tbb::flow::make_edge(s0, tbb::flow::get<0>(j2.input_ports()));

    /*        /+    */
    /*       / | j0 */
    /*      /  +    */
    /*     /        */
    /*    / /--+    */
    /*  s0-/   | j1 */
    /*    \    +    */
    /*     \        */
    /*      \--+    */
    /*         | j2 */
    /*         +    */

    // do all joins appear in successor list?
    successor_list_type jv1;
    jv1.push_back(&(tbb::flow::get<0>(j0.input_ports())));
    jv1.push_back(&(tbb::flow::get<0>(j1.input_ports())));
    jv1.push_back(&(tbb::flow::get<0>(j2.input_ports())));
    snode_type::successor_list_type sv;
    s0.copy_successors(sv);
    ASSERT(lists_match(sv, jv1), "mismatch in successor list");

    tbb::flow::make_edge(q0, tbb::flow::get<1>(j2.input_ports()));
    tbb::flow::make_edge(j2, q1);
    s0.activate();

    /*        /+           */
    /*       / | j0        */
    /*      /  +           */
    /*     /               */
    /*    / /--+           */
    /*  s0-/   | j1        */
    /*    \    +           */
    /*     \               */
    /*      \--+           */
    /*         | j2----q1  */
    /*  q0-----+           */

    q0.try_put(1);
    g.wait_for_all();
    ASSERT(q1.try_get(dont_care), "join did not emit result");
    j2.extract();
    tbb::flow::make_edge(q0, tbb::flow::get<1>(j2.input_ports()));
    tbb::flow::make_edge(j2, q1);

    /*        /+           */
    /*       / | j0        */
    /*      /  +           */
    /*     /               */
    /*    / /--+           */
    /*  s0-/   | j1        */
    /*         +           */
    /*                     */
    /*         +           */
    /*         | j2----q1  */
    /*  q0-----+           */

    jv1.clear();
    jv1.push_back(&(tbb::flow::get<0>(j0.input_ports())));
    jv1.push_back(&(tbb::flow::get<0>(j1.input_ports())));
    s0.copy_successors(sv);
    ASSERT(lists_match(sv, jv1), "mismatch in successor list");

    q0.try_put(1);
    g.wait_for_all();
    ASSERT(!q1.try_get(dont_care), "extract of successor did not remove pred link");

    s0.extract();

    /*         +           */
    /*         | j0        */
    /*         +           */
    /*                     */
    /*         +           */
    /*  s0     | j1        */
    /*         +           */
    /*                     */
    /*         +           */
    /*         | j2----q1  */
    /*  q0-----+           */

    ASSERT(s0.successor_count() == 0, "successor list not cleared");
    s0.copy_successors(sv);
    ASSERT(sv.size() == 0, "non-empty successor list");

    tbb::flow::make_edge(s0, tbb::flow::get<0>(j2.input_ports()));

    /*         +           */
    /*         | j0        */
    /*         +           */
    /*                     */
    /*         +           */
    /*  s0     | j1        */
    /*    \    +           */
    /*     \               */
    /*      \--+           */
    /*         | j2----q1  */
    /*  q0-----+           */

    jv1.clear();
    jv1.push_back(&(tbb::flow::get<0>(j2.input_ports())));
    s0.copy_successors(sv);
    ASSERT(lists_match(sv, jv1), "mismatch in successor list");

    q0.try_put(1);
    g.wait_for_all();
    ASSERT(!q1.try_get(dont_care), "extract of successor did not remove pred link");
}
#endif  /* TBB_DEPRECATED_FLOW_NODE_EXTRACTION */

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
#if TBB_DEPRECATED_INPUT_NODE_BODY
    bool source_body_f(int& i) { return i > 5; }
#else
    int source_body_f(tbb::flow_control&) { return 42; }
#endif

void test_deduction_guides() {
    using namespace tbb::flow;
    graph g;

#if TBB_DEPRECATED_INPUT_NODE_BODY
    auto lambda = [](int& i) { return i > 5; };
    auto non_const_lambda = [](int& i) mutable { return i > 5; };
#else
    auto lambda = [](tbb::flow_control&) { return 42; };
    auto non_const_lambda = [](tbb::flow_control&) mutable { return 42; };
#endif

    // Tests for source_node(graph&, Body)
    source_node s1(g, lambda);
    static_assert(std::is_same_v<decltype(s1), source_node<int>>);

    source_node s2(g, non_const_lambda);
    static_assert(std::is_same_v<decltype(s2), source_node<int>>);

    source_node s3(g, source_body_f);
    static_assert(std::is_same_v<decltype(s3), source_node<int>>);

    source_node s4(s3);
    static_assert(std::is_same_v<decltype(s4), source_node<int>>);

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    broadcast_node<int> bc(g);

    // Tests for source_node(const node_set<Args...>&, Body)
    source_node s5(precedes(bc), lambda);
    static_assert(std::is_same_v<decltype(s5), source_node<int>>);

    source_node s6(precedes(bc), non_const_lambda);
    static_assert(std::is_same_v<decltype(s6), source_node<int>>);

    source_node s7(precedes(bc), source_body_f);
    static_assert(std::is_same_v<decltype(s7), source_node<int>>);
#endif
    g.wait_for_all();
}

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#include <array>
void test_follows_and_precedes_api() {
    using namespace tbb::flow;

    graph g;

    std::array<buffer_node<bool>, 3> successors {{
        buffer_node<bool>(g),
        buffer_node<bool>(g),
        buffer_node<bool>(g)
    }};

    bool do_try_put = true;

    source_node<bool> src(precedes(successors[0], successors[1], successors[2]),
    #if TBB_DEPRECATED_INPUT_NODE_BODY
    [&](bool& v) -> bool {
        if(do_try_put) {
            v = do_try_put;
            do_try_put = false;
            return true;
        }
        else {
            return false;
        }
    }
    #else
    [&](tbb::flow_control& fc) -> bool {
        if(!do_try_put)
            fc.stop();
        do_try_put = !do_try_put;
        return true;
    }
    #endif
    );

    src.activate();
    g.wait_for_all();

    bool storage;
    for(auto& successor: successors) {
        ASSERT((successor.try_get(storage) && !successor.try_get(storage)),
            "Not exact edge quantity was made");
    }
}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

int TestMain() {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for ( int p = MinThread; p < MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        test_single_dest<int>();
        test_single_dest<float>();
    }
    test_reset();
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
    test_extract();
#endif
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    test_follows_and_precedes_api();
#endif
#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    test_deduction_guides();
#endif
    return Harness::Done;
}

