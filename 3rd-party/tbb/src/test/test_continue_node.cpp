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

#if __TBB_CPF_BUILD
#define TBB_DEPRECATED_FLOW_NODE_EXTRACTION 1
#endif

#include "harness.h"
#include "harness_graph.h"

#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"
#include "test_follows_and_precedes_api.h"

#define N 1000
#define MAX_NODES 4
#define C 8

struct empty_no_assign : private NoAssign {
   empty_no_assign() {}
   empty_no_assign( int ) {}
   operator int() { return 0; }
};

// A class to use as a fake predecessor of continue_node
struct fake_continue_sender : public tbb::flow::sender<tbb::flow::continue_msg>
{
    typedef tbb::flow::sender<tbb::flow::continue_msg>::successor_type successor_type;
    // Define implementations of virtual methods that are abstract in the base class
    bool register_successor( successor_type& ) __TBB_override { return false; }
    bool remove_successor( successor_type& )   __TBB_override { return false; }
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
    typedef tbb::flow::sender<tbb::flow::continue_msg>::built_successors_type built_successors_type;
    built_successors_type bst;
    built_successors_type &built_successors() __TBB_override { return bst; }
    void internal_add_built_successor( successor_type &) __TBB_override { }
    void internal_delete_built_successor( successor_type &) __TBB_override { }
    void copy_successors(successor_list_type &) __TBB_override {}
    size_t successor_count() __TBB_override {return 0;}
#endif
};

template< typename InputType >
struct parallel_puts : private NoAssign {

    tbb::flow::receiver< InputType > * const my_exe_node;

    parallel_puts( tbb::flow::receiver< InputType > &exe_node ) : my_exe_node(&exe_node) {}

    void operator()( int ) const  {
        for ( int i = 0; i < N; ++i ) {
            // the nodes will accept all puts
            ASSERT( my_exe_node->try_put( InputType() ) == true, NULL );
        }
    }

};

template< typename OutputType >
void run_continue_nodes( int p, tbb::flow::graph& g, tbb::flow::continue_node< OutputType >& n ) {
    fake_continue_sender fake_sender;
    for (size_t i = 0; i < N; ++i) {
        n.register_predecessor( fake_sender );
    }

    for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
        std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));
        harness_graph_executor<tbb::flow::continue_msg, OutputType>::execute_count = 0;

        for (size_t r = 0; r < num_receivers; ++r ) {
            tbb::flow::make_edge( n, receivers[r] );
        }
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        ASSERT(n.successor_count() == (size_t)num_receivers, NULL);
        ASSERT(n.predecessor_count() == 0, NULL);
        typename tbb::flow::continue_node<OutputType>::successor_list_type my_succs;
        typedef typename tbb::flow::continue_node<OutputType>::successor_list_type::iterator sv_iter_type;
        n.copy_successors(my_succs);
        ASSERT(my_succs.size() == num_receivers, NULL);
#endif

        NativeParallelFor( p, parallel_puts<tbb::flow::continue_msg>(n) );
        g.wait_for_all();

        // 2) the nodes will receive puts from multiple predecessors simultaneously,
        size_t ec = harness_graph_executor<tbb::flow::continue_msg, OutputType>::execute_count;
        ASSERT( (int)ec == p, NULL );
        for (size_t r = 0; r < num_receivers; ++r ) {
            size_t c = receivers[r].my_count;
            // 3) the nodes will send to multiple successors.
            ASSERT( (int)c == p, NULL );
        }

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
        for(sv_iter_type si=my_succs.begin(); si != my_succs.end(); ++si) {
            tbb::flow::remove_edge( n, **si );
        }
#else
        for (size_t r = 0; r < num_receivers; ++r ) {
            tbb::flow::remove_edge( n, receivers[r] );
        }
#endif
    }
}

template< typename OutputType, typename Body >
void continue_nodes( Body body ) {
    for (int p = 1; p < 2*MaxThread; ++p) {
        tbb::flow::graph g;
        tbb::flow::continue_node< OutputType > exe_node( g, body );
        run_continue_nodes( p, g, exe_node);
        exe_node.try_put(tbb::flow::continue_msg());
        tbb::flow::continue_node< OutputType > exe_node_copy( exe_node );
        run_continue_nodes( p, g, exe_node_copy);
    }
}

const size_t Offset = 123;
tbb::atomic<size_t> global_execute_count;

template< typename OutputType >
struct inc_functor {

    tbb::atomic<size_t> local_execute_count;
    inc_functor( ) { local_execute_count = 0; }
    inc_functor( const inc_functor &f ) { local_execute_count = f.local_execute_count; }
    void operator=(const inc_functor &f) { local_execute_count = f.local_execute_count; }

    OutputType operator()( tbb::flow::continue_msg ) {
       ++global_execute_count;
       ++local_execute_count;
       return OutputType();
    }

};

template< typename OutputType >
void continue_nodes_with_copy( ) {

    for (int p = 1; p < 2*MaxThread; ++p) {
        tbb::flow::graph g;
        inc_functor<OutputType> cf;
        cf.local_execute_count = Offset;
        global_execute_count = Offset;

        tbb::flow::continue_node< OutputType > exe_node( g, cf );
        fake_continue_sender fake_sender;
        for (size_t i = 0; i < N; ++i) {
           exe_node.register_predecessor( fake_sender );
        }

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
            std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));

            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::make_edge( exe_node, receivers[r] );
            }

            NativeParallelFor( p, parallel_puts<tbb::flow::continue_msg>(exe_node) );
            g.wait_for_all();

            // 2) the nodes will receive puts from multiple predecessors simultaneously,
            for (size_t r = 0; r < num_receivers; ++r ) {
                size_t c = receivers[r].my_count;
                // 3) the nodes will send to multiple successors.
                ASSERT( (int)c == p, NULL );
            }
            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::remove_edge( exe_node, receivers[r] );
            }
        }

        // validate that the local body matches the global execute_count and both are correct
        inc_functor<OutputType> body_copy = tbb::flow::copy_body< inc_functor<OutputType> >( exe_node );
        const size_t expected_count = p*MAX_NODES + Offset;
        size_t global_count = global_execute_count;
        size_t inc_count = body_copy.local_execute_count;
        ASSERT( global_count == expected_count && global_count == inc_count, NULL );
        g.reset(tbb::flow::rf_reset_bodies);
        body_copy = tbb::flow::copy_body< inc_functor<OutputType> >( exe_node );
        inc_count = body_copy.local_execute_count;
        ASSERT( Offset == inc_count, "reset(rf_reset_bodies) did not reset functor" );

    }
}

template< typename OutputType >
void run_continue_nodes() {
    harness_graph_executor< tbb::flow::continue_msg, OutputType>::max_executors = 0;
    #if __TBB_CPP11_LAMBDAS_PRESENT
    continue_nodes<OutputType>( []( tbb::flow::continue_msg i ) -> OutputType { return harness_graph_executor<tbb::flow::continue_msg, OutputType>::func(i); } );
    #endif
    continue_nodes<OutputType>( &harness_graph_executor<tbb::flow::continue_msg, OutputType>::func );
    continue_nodes<OutputType>( typename harness_graph_executor<tbb::flow::continue_msg, OutputType>::functor() );
    continue_nodes_with_copy<OutputType>();
}

//! Tests limited concurrency cases for nodes that accept data messages
void test_concurrency(int num_threads) {
    tbb::task_scheduler_init init(num_threads);
    run_continue_nodes<tbb::flow::continue_msg>();
    run_continue_nodes<int>();
    run_continue_nodes<empty_no_assign>();
}
/*
 * Connection of two graphs is not currently supported, but works to some limited extent.
 * This test is included to check for backward compatibility. It checks that a continue_node
 * with predecessors in two different graphs receives the required
 * number of continue messages before it executes.
 */
using namespace tbb::flow;

struct add_to_counter {
    int* counter;
    add_to_counter(int& var):counter(&var){}
    void operator()(continue_msg){*counter+=1;}
};

void test_two_graphs(){
    int count=0;

    //graph g with broadcast_node and continue_node
    graph g;
    broadcast_node<continue_msg> start_g(g);
    continue_node<continue_msg> first_g(g, add_to_counter(count));

    //graph h with broadcast_node
    graph h;
    broadcast_node<continue_msg> start_h(h);

    //making two edges to first_g from the two graphs
    make_edge(start_g,first_g);
    make_edge(start_h, first_g);

    //two try_puts from the two graphs
    start_g.try_put(continue_msg());
    start_h.try_put(continue_msg());
    g.wait_for_all();
    ASSERT(count==1, "Not all continue messages received");

    //two try_puts from the graph that doesn't contain the node
    count=0;
    start_h.try_put(continue_msg());
    start_h.try_put(continue_msg());
    g.wait_for_all();
    ASSERT(count==1, "Not all continue messages received -1");

    //only one try_put
    count=0;
    start_g.try_put(continue_msg());
    g.wait_for_all();
    ASSERT(count==0, "Node executed without waiting for all predecessors");
}

#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
void test_extract() {
    int my_count = 0;
    tbb::flow::continue_msg cm;
    tbb::flow::graph g;
    tbb::flow::broadcast_node<tbb::flow::continue_msg> b0(g);
    tbb::flow::broadcast_node<tbb::flow::continue_msg> b1(g);
    tbb::flow::continue_node<tbb::flow::continue_msg>  c0(g, add_to_counter(my_count));
    tbb::flow::queue_node<tbb::flow::continue_msg> q0(g);

    tbb::flow::make_edge(b0, c0);
    tbb::flow::make_edge(b1, c0);
    tbb::flow::make_edge(c0, q0);
    for( int i = 0; i < 2; ++i ) {
        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(c0.predecessor_count() == 2 && c0.successor_count() == 1, "c0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");

        /* b0         */
        /*   \        */
        /*    c0 - q0 */
        /*   /        */
        /* b1         */

        b0.try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        ASSERT(my_count == 0, "continue_node fired too soon");
        b1.try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        ASSERT(my_count == 1, "continue_node didn't fire");
        ASSERT(q0.try_get(cm), "continue_node didn't forward");

        b0.extract();

        /* b0         */
        /*            */
        /*    c0 - q0 */
        /*   /        */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(c0.predecessor_count() == 1 && c0.successor_count() == 1, "c0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(tbb::flow::continue_msg());
        b0.try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        ASSERT(my_count == 1, "b0 messages being forwarded to continue_node even though it is disconnected");
        b1.try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        ASSERT(my_count == 2, "continue_node didn't fire though it has only one predecessor");
        ASSERT(q0.try_get(cm), "continue_node didn't forward second time");

        c0.extract();

        /* b0         */
        /*            */
        /*    c0   q0 */
        /*            */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(c0.predecessor_count() == 0 && c0.successor_count() == 0, "c0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(tbb::flow::continue_msg());
        b0.try_put(tbb::flow::continue_msg());
        b1.try_put(tbb::flow::continue_msg());
        b1.try_put(tbb::flow::continue_msg());
        g.wait_for_all();
        ASSERT(my_count == 2, "continue didn't fire though it has only one predecessor");
        ASSERT(!q0.try_get(cm), "continue_node forwarded though it shouldn't");
        make_edge(b0, c0);

        /* b0         */
        /*   \        */
        /*    c0   q0 */
        /*            */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(c0.predecessor_count() == 1 && c0.successor_count() == 0, "c0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");

        b0.try_put(tbb::flow::continue_msg());
        g.wait_for_all();

        ASSERT(my_count == 3, "continue didn't fire though it has only one predecessor");
        ASSERT(!q0.try_get(cm), "continue_node forwarded though it shouldn't");

        tbb::flow::make_edge(b1, c0);
        tbb::flow::make_edge(c0, q0);
        my_count = 0;
    }
}
#endif

struct lightweight_policy_body : NoAssign {
    const tbb::tbb_thread::id my_thread_id;
    tbb::atomic<size_t> my_count;

    lightweight_policy_body() : my_thread_id(tbb::this_tbb_thread::get_id()) {
        my_count = 0;
    }
    void operator()(tbb::flow::continue_msg) {
        ++my_count;
        tbb::tbb_thread::id body_thread_id = tbb::this_tbb_thread::get_id();
        ASSERT(body_thread_id == my_thread_id, "Body executed as not lightweight");
    }
};

void test_lightweight_policy() {
    tbb::flow::graph g;
    tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight> node1(g, lightweight_policy_body());
    tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight> node2(g, lightweight_policy_body());

    tbb::flow::make_edge(node1, node2);
    const size_t n = 10;
    for(size_t i = 0; i < n; ++i) {
        node1.try_put(tbb::flow::continue_msg());
    }
    g.wait_for_all();

    lightweight_policy_body body1 = tbb::flow::copy_body<lightweight_policy_body>(node1);
    lightweight_policy_body body2 = tbb::flow::copy_body<lightweight_policy_body>(node2);
    ASSERT(body1.my_count == n, "Body of the first node needs to be executed N times");
    ASSERT(body2.my_count == n, "Body of the second node needs to be executed N times");
}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#include <array>
#include <vector>
void test_follows_and_precedes_api() {
    using msg_t = tbb::flow::continue_msg;

    std::array<msg_t, 3> messages_for_follows = { { msg_t(), msg_t(), msg_t() } };
    std::vector<msg_t> messages_for_precedes  = { msg_t() };

    auto pass_through = [](const msg_t& msg) { return msg; };

    follows_and_precedes_testing::test_follows
        <msg_t, tbb::flow::continue_node<msg_t>>
        (messages_for_follows, pass_through);

    follows_and_precedes_testing::test_precedes
        <msg_t, tbb::flow::continue_node<msg_t>>
        (messages_for_precedes, pass_through);
}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename ExpectedType, typename Body>
void test_deduction_guides_common(Body body) {
    using namespace tbb::flow;
    graph g;

    continue_node c1(g, body);
    static_assert(std::is_same_v<decltype(c1), continue_node<ExpectedType>>);

    continue_node c2(g, body, lightweight());
    static_assert(std::is_same_v<decltype(c2), continue_node<ExpectedType, lightweight>>);

    continue_node c3(g, 5, body);
    static_assert(std::is_same_v<decltype(c3), continue_node<ExpectedType>>);

    continue_node c4(g, 5, body, lightweight());
    static_assert(std::is_same_v<decltype(c4), continue_node<ExpectedType, lightweight>>);

#if __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
    continue_node c5(g, body, node_priority_t(5));
    static_assert(std::is_same_v<decltype(c5), continue_node<ExpectedType>>);

    continue_node c6(g, body, lightweight(), node_priority_t(5));
    static_assert(std::is_same_v<decltype(c6), continue_node<ExpectedType, lightweight>>);

    continue_node c7(g, 5, body, node_priority_t(5));
    static_assert(std::is_same_v<decltype(c7), continue_node<ExpectedType>>);

    continue_node c8(g, 5, body, lightweight(), node_priority_t(5));
    static_assert(std::is_same_v<decltype(c8), continue_node<ExpectedType, lightweight>>);
#endif

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    broadcast_node<continue_msg> b(g);

    continue_node c9(follows(b), body);
    static_assert(std::is_same_v<decltype(c9), continue_node<ExpectedType>>);

    continue_node c10(follows(b), body, lightweight());
    static_assert(std::is_same_v<decltype(c10), continue_node<ExpectedType, lightweight>>);

    continue_node c11(follows(b), 5, body);
    static_assert(std::is_same_v<decltype(c11), continue_node<ExpectedType>>);

    continue_node c12(follows(b), 5, body, lightweight());
    static_assert(std::is_same_v<decltype(c12), continue_node<ExpectedType, lightweight>>);

#if __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
    continue_node c13(follows(b), body, node_priority_t(5));
    static_assert(std::is_same_v<decltype(c13), continue_node<ExpectedType>>);

    continue_node c14(follows(b), body, lightweight(), node_priority_t(5));
    static_assert(std::is_same_v<decltype(c14), continue_node<ExpectedType, lightweight>>);

    continue_node c15(follows(b), 5, body, node_priority_t(5));
    static_assert(std::is_same_v<decltype(c15), continue_node<ExpectedType>>);

    continue_node c16(follows(b), 5, body, lightweight(), node_priority_t(5));
    static_assert(std::is_same_v<decltype(c16), continue_node<ExpectedType, lightweight>>);
#endif
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

    continue_node c17(c1);
    static_assert(std::is_same_v<decltype(c17), continue_node<ExpectedType>>);
}

int continue_body_f(const tbb::flow::continue_msg&) { return 1; }
void continue_void_body_f(const tbb::flow::continue_msg&) {}

void test_deduction_guides() {
    using tbb::flow::continue_msg;
    test_deduction_guides_common<int>([](const continue_msg&)->int { return 1; } );
    test_deduction_guides_common<continue_msg>([](const continue_msg&) {});

    test_deduction_guides_common<int>([](const continue_msg&) mutable ->int { return 1; });
    test_deduction_guides_common<continue_msg>([](const continue_msg&) mutable {});

    test_deduction_guides_common<int>(continue_body_f);
    test_deduction_guides_common<continue_msg>(continue_void_body_f);
}

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

// TODO: use pass_through from test_function_node instead
template<typename T>
struct passing_body {
    T operator()(const T& val) {
        return val;
    }
};

/*
    The test covers the case when a node with non-default mutex type is a predecessor for continue_node,
    because there used to be a bug when make_edge(node, continue_node)
    did not update continue_node's predecesosor threshold
    since the specialization of node's successor_cache for a continue_node was not chosen.
*/
void test_successor_cache_specialization() {
    using namespace tbb::flow;

    graph g;

    broadcast_node<continue_msg> node_with_default_mutex_type(g);
    buffer_node<continue_msg> node_with_non_default_mutex_type(g);

    continue_node<continue_msg> node(g, passing_body<continue_msg>());

    make_edge(node_with_default_mutex_type, node);
    make_edge(node_with_non_default_mutex_type, node);

    buffer_node<continue_msg> buf(g);

    make_edge(node, buf);

    node_with_default_mutex_type.try_put(continue_msg());
    node_with_non_default_mutex_type.try_put(continue_msg());

    g.wait_for_all();

    continue_msg storage;
    ASSERT((buf.try_get(storage) && !buf.try_get(storage)),
            "Wrong number of messages is passed via continue_node");
}

int TestMain() {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
       test_concurrency(p);
   }
   test_two_graphs();
#if __TBB_PREVIEW_LIGHTWEIGHT_POLICY
   test_lightweight_policy();
#endif
#if TBB_DEPRECATED_FLOW_NODE_EXTRACTION
   test_extract();
#endif
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    test_follows_and_precedes_api();
#endif
#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    test_deduction_guides();
#endif
   test_successor_cache_specialization();
   return Harness::Done;
}
