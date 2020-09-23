/*
    Copyright (c) 2018-2020 Intel Corporation

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

#include "harness_defs.h"

#if __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES
#define TBB_DEPRECATED_INPUT_NODE_BODY __TBB_CPF_BUILD

#include "harness_graph.h"
#include "harness_barrier.h"

#include "tbb/flow_graph.h"
#include "tbb/tbb_thread.h"
#include "tbb/parallel_for.h"
#include "tbb/concurrent_queue.h"

#include <vector>
#include <cstdlib>

using namespace tbb::flow;

tbb::atomic<unsigned> g_task_num;

void spin_for( double delta ) {
    tbb::tick_count start = tbb::tick_count::now();
    while( (tbb::tick_count::now() - start).seconds() < delta ) ;
}

namespace PriorityNodesTakePrecedence {

struct TaskInfo {
    TaskInfo() : my_priority(-1), my_task_index(-1) {}
    TaskInfo( int priority, int task_index )
        : my_priority(priority), my_task_index(task_index) {}
    int my_priority;
    int my_task_index;
};
std::vector<TaskInfo> g_task_info;
tbb::atomic<bool> g_work_submitted;

const unsigned node_num = 100;
const unsigned start_index = node_num / 3;
const unsigned end_index = node_num * 2 / 3;
tbb::atomic<unsigned> g_priority_task_index;

void body_func( int priority ) {
    while( !g_work_submitted ) __TBB_Yield();
    int current_task_index = g_task_num++;
    if( priority )
        g_task_info[g_priority_task_index++] = TaskInfo( priority, current_task_index );
}

struct FunctionBody {
    FunctionBody( int priority ) : my_priority( priority ) {}
    int operator()( int msg ) const {
        body_func( my_priority );
        return msg;
    }
private:
    int my_priority;
};

typedef multifunction_node< int,tuple<int> > multi_node;

struct MultifunctionBody {
    MultifunctionBody( int priority ) : my_priority( priority ) {}
    void operator()( int msg, multi_node::output_ports_type& op ) const {
        body_func( my_priority );
        get<0>(op).try_put( msg );
    }
private:
    int my_priority;
};

template<typename NodeType, typename BodyType>
NodeType* node_creator( graph& g, unsigned index ) {
    if( start_index <= index && index < end_index )
        return new NodeType( g, unlimited, BodyType(index), node_priority_t(index) );
    else
        return new NodeType( g, unlimited, BodyType(0) );
}

struct passthru_body {
    continue_msg operator()( int ) const {
        return continue_msg();
    }
};

template<typename NodeType> sender<int>& get_sender( NodeType& node ) { return node; }
template<> sender<int>& get_sender<multi_node>( multi_node& node ) { return output_port<0>(node); }

template<typename NodeType, typename NodeTypeCreator, typename NodePortRetriever>
void test_node( NodeTypeCreator node_creator_func, NodePortRetriever get_sender ) {
    graph g;
    broadcast_node<int> bn(g);
    function_node<int> tn(g, unlimited, passthru_body());
    // Using pointers to nodes to avoid errors on compilers, which try to generate assignment
    // operator for the nodes
    std::vector<NodeType*> nodes;
    for( unsigned i = 0; i < node_num; ++i ) {
        nodes.push_back( node_creator_func(g, i) );
        make_edge( bn, *nodes.back() );
        make_edge( get_sender(*nodes.back()), tn );
    }

    const size_t repeats = 50;
    const size_t priority_nodes_num = end_index - start_index;
    size_t internal_order_failures = 0;
    size_t global_order_failures = 0;
    for( size_t repeat = 0; repeat < repeats; ++repeat ) {
        g_work_submitted = false;
        g_task_num = g_priority_task_index = 0;
        g_task_info.clear(); g_task_info.resize( priority_nodes_num );

        bn.try_put( 0 );
        // Setting of the flag is based on the knowledge that the calling thread broadcasts the message
        // to successor nodes, that is spawns tasks. Thus, this makes this test to be a whitebox test to
        // some extent.
        g_work_submitted = true;

        g.wait_for_all();

        ASSERT( g_priority_task_index == g_task_info.size(), "Incorrect number of tasks with priority" );
        bool found_max = false;
        bool found_min = false;
        for( unsigned i = 0; i < g_priority_task_index/2; ++i ) {
            if( g_task_info[i].my_priority == int(end_index-1) )
                found_max = true;
            if( g_task_info[g_priority_task_index-1-i].my_priority == int(start_index) )
                found_min = true;
        }
        if( !found_min || !found_max )
            ++internal_order_failures;
        for( unsigned i = 0; i < g_priority_task_index; ++i ) {
            // This check might fail because priorities do not guarantee ordering, i.e. assumption
            // that all priority nodes should increment the task counter before any subsequent
            // no-priority node is not correct. In the worst case, a thread that took a priority
            // node might be preempted and become the last to increment the counter. That's why the
            // test passing is based on statistics, which could be affected by machine overload
            // unfortunately.
            // TODO: make the test deterministic.
            if( g_task_info[i].my_task_index > int(priority_nodes_num) + MaxThread )
                ++global_order_failures;
        }
    }
    float failure_ratio = float(internal_order_failures) / float(repeats);
    ASSERT(
        failure_ratio <= 0.3f,
        "Nodes with priorities executed in wrong order among each other too frequently."
    );
    failure_ratio = float(global_order_failures) / float(repeats*priority_nodes_num);
    ASSERT(
        failure_ratio <= 0.1f,
        "Nodes with priorities executed in wrong order too frequently over non-prioritized nodes."
    );
    for( size_t i = 0; i < nodes.size(); ++i )
        delete nodes[i];
}

void test( int num_threads ) {
    REMARK( "Testing execution of nodes with priority takes precedence (num_threads=%d) - ", num_threads );
    tbb::task_scheduler_init init(num_threads);
    test_node< function_node<int,int> >( &node_creator<function_node<int,int>, FunctionBody>,
                                         &get_sender< function_node<int,int> > );
    test_node<multi_node>( &node_creator<multi_node, MultifunctionBody>, &get_sender< multi_node > );
    REMARK( "done\n" );
}
} /* namespace PriorityNodesTakePrecedence */

namespace ThreadsEagerReaction {

using Harness::SpinBarrier;

enum task_type_t { no_task, regular_task, async_task };

struct profile_t {
    task_type_t task_type;
    unsigned global_task_id;
    double elapsed;
};

std::vector<unsigned> g_async_task_ids;

typedef unsigned data_type;
typedef async_node<data_type, data_type> async_node_type;
typedef multifunction_node<
    data_type, tuple<data_type, data_type> > decider_node_type;
struct AsyncActivity {
    typedef async_node_type::gateway_type gateway_type;

    struct work_type { data_type input; gateway_type* gateway; };
    bool done;
    tbb::concurrent_queue<work_type> my_queue;
    tbb::tbb_thread my_service_thread;

    struct ServiceThreadFunc {
        SpinBarrier& my_barrier;
        ServiceThreadFunc(SpinBarrier& barrier) : my_barrier(barrier) {}
        void operator()(AsyncActivity* activity) {
            while (!activity->done) {
                work_type work;
                while (activity->my_queue.try_pop(work)) {
                    g_async_task_ids.push_back( ++g_task_num );
                    work.gateway->try_put(work.input);
                    work.gateway->release_wait();
                    my_barrier.wait();
                }
            }
        }
    };
    void stop_and_wait() { done = true; my_service_thread.join(); }

    void submit(data_type input, gateway_type* gateway) {
        work_type work = { input, gateway };
        gateway->reserve_wait();
        my_queue.push(work);
    }
    AsyncActivity(SpinBarrier& barrier)
        : done(false), my_service_thread(ServiceThreadFunc(barrier), this) {}
};

struct StartBody {
    bool has_run;
#if TBB_DEPRECATED_INPUT_NODE_BODY
    bool operator()(data_type& input) {
        if (has_run) return false;
        else {
            input = 1;
            has_run = true;
            return true;
        }
    }
#else
    data_type operator()(tbb::flow_control& fc) {
        if (has_run){
            fc.stop();
            return data_type();
        }
        has_run = true;
        return 1;
    }
#endif
    StartBody() : has_run(false) {}
};

struct ParallelForBody {
    SpinBarrier& my_barrier;
    const data_type& my_input;
    ParallelForBody(SpinBarrier& barrier, const data_type& input)
        : my_barrier(barrier), my_input(input) {}
    void operator()(const data_type&) const {
        my_barrier.wait();
        ++g_task_num;
    }
};

struct CpuWorkBody {
    SpinBarrier& my_barrier;
    const int my_tasks_count;
    data_type operator()(const data_type& input) {
        tbb::parallel_for(0, my_tasks_count, ParallelForBody(my_barrier, input), tbb::simple_partitioner());
        return input;
    }
    CpuWorkBody(SpinBarrier& barrier, int tasks_count)
        : my_barrier(barrier), my_tasks_count(tasks_count) {}
};

struct DeciderBody {
    const data_type& my_limit;
    DeciderBody( const data_type& limit ) : my_limit( limit ) {}
    void operator()(data_type input, decider_node_type::output_ports_type& ports) {
        if (input < my_limit)
            get<0>(ports).try_put(input + 1);
    }
};

struct AsyncSubmissionBody {
    AsyncActivity* my_activity;
    void operator()(data_type input, async_node_type::gateway_type& gateway) {
        my_activity->submit(input, &gateway);
    }
    AsyncSubmissionBody(AsyncActivity* activity) : my_activity(activity) {}
};

void test( int num_threads ) {
    REMARK( "Testing threads react eagerly on asynchronous tasks (num_threads=%d) - ", num_threads );
    if( num_threads == tbb::task_scheduler_init::default_num_threads() ) {
        // one thread is required for asynchronous compute resource
        REMARK("skipping test since it is designed to work on less number of threads than "
               "hardware concurrency allows\n");
        return;
    }
    const unsigned cpu_threads = unsigned(num_threads);
    const unsigned cpu_tasks_per_thread = 4;
    const unsigned nested_cpu_tasks = cpu_tasks_per_thread * cpu_threads;
    const unsigned async_subgraph_reruns = 8;
    const unsigned cpu_subgraph_reruns = 2;

    SpinBarrier barrier(cpu_threads + /*async thread=*/1);
    g_task_num = 0;
    g_async_task_ids.clear();
    g_async_task_ids.reserve(async_subgraph_reruns);

    tbb::task_scheduler_init init(cpu_threads);
    AsyncActivity activity(barrier);
    graph g;

    input_node<data_type> starter_node(g, StartBody());
    function_node<data_type, data_type> cpu_work_node(
        g, unlimited, CpuWorkBody(barrier, nested_cpu_tasks));
    decider_node_type cpu_restarter_node(g, unlimited, DeciderBody(cpu_subgraph_reruns));
    async_node_type async_node(g, unlimited, AsyncSubmissionBody(&activity));
    decider_node_type async_restarter_node(
        g, unlimited, DeciderBody(async_subgraph_reruns), node_priority_t(1)
    );

    make_edge(starter_node, cpu_work_node);
    make_edge(cpu_work_node, cpu_restarter_node);
    make_edge(output_port<0>(cpu_restarter_node), cpu_work_node);

    make_edge(starter_node, async_node);
    make_edge(async_node, async_restarter_node);
    make_edge(output_port<0>(async_restarter_node), async_node);

    starter_node.activate();
    g.wait_for_all();
    activity.stop_and_wait();

    const size_t async_task_num = size_t(async_subgraph_reruns);
    ASSERT( g_async_task_ids.size() == async_task_num, "Incorrect number of async tasks." );
    unsigned max_span = unsigned(2 * cpu_threads + 1);
    for( size_t idx = 1; idx < async_task_num; ++idx ) {
        ASSERT( g_async_task_ids[idx] - g_async_task_ids[idx-1] <= max_span,
                "Async tasks were not able to interfere with CPU tasks." );
    }
    REMARK("done\n");
}
} /* ThreadsEagerReaction */

namespace LimitingExecutionToPriorityTask {

enum work_type_t { NONPRIORITIZED_WORK, PRIORITIZED_WORK };

struct execution_tracker_t {
    execution_tracker_t() { reset(); }
    void reset() {
        prioritized_work_submitter = tbb::tbb_thread::id();
        prioritized_work_started = false;
        prioritized_work_finished = false;
        prioritized_work_interrupted = false;
    }
    tbb::tbb_thread::id prioritized_work_submitter;
    bool prioritized_work_started;
    bool prioritized_work_finished;
    bool prioritized_work_interrupted;
} exec_tracker;

template<work_type_t work_type>
void do_node_work( int work_size );

template<work_type_t>
void do_nested_work( const tbb::tbb_thread::id& tid, const tbb::blocked_range<int>& subrange );

template<work_type_t work_type>
struct CommonBody {
    CommonBody() : my_body_size( 0 ) { }
    CommonBody( int body_size ) : my_body_size( body_size ) { }
    continue_msg operator()( const continue_msg& msg ) const {
        do_node_work<work_type>(my_body_size);
        return msg;
    }
    void operator()( const tbb::blocked_range<int>& subrange ) const {
        do_nested_work<work_type>( /*tid=*/tbb::this_tbb_thread::get_id(), subrange );
    }
    int my_body_size;
};

template<work_type_t work_type>
void do_node_work(int work_size) {
    tbb::parallel_for( tbb::blocked_range<int>(0, work_size), CommonBody<work_type>(),
                       tbb::simple_partitioner() );
}

template<work_type_t>
void do_nested_work( const tbb::tbb_thread::id& tid, const tbb::blocked_range<int>& /*subrange*/ ) {
    // This is non-prioritized work...
    if( exec_tracker.prioritized_work_submitter != tid )
        return;
    // ...being executed by the thread that initially started prioritized one...
    ASSERT( exec_tracker.prioritized_work_started,
            "Prioritized work should have been started by that time." );
    // ...prioritized work has been started already...
    if( exec_tracker.prioritized_work_finished )
        return;
    // ...but has not been finished yet
    exec_tracker.prioritized_work_interrupted = true;
}

struct IsolationFunctor {
    int work_size;
    IsolationFunctor(int ws) : work_size(ws) {}
    void operator()() const {
        tbb::parallel_for( tbb::blocked_range<int>(0, work_size), CommonBody<PRIORITIZED_WORK>(),
                           tbb::simple_partitioner() );
    }
};

template<>
void do_node_work<PRIORITIZED_WORK>(int work_size) {
    exec_tracker.prioritized_work_submitter = tbb::this_tbb_thread::get_id();
    exec_tracker.prioritized_work_started = true;
    tbb::this_task_arena::isolate( IsolationFunctor(work_size) );
    exec_tracker.prioritized_work_finished = true;
}

template<>
void do_nested_work<PRIORITIZED_WORK>( const tbb::tbb_thread::id& tid,
                                       const tbb::blocked_range<int>& /*subrange*/ ) {
    if( exec_tracker.prioritized_work_submitter == tid ) {
        ASSERT( !exec_tracker.prioritized_work_interrupted,
                "Thread was not fully devoted to processing of prioritized task." );
    } else {
        // prolong processing of prioritized work so that the thread that started
        // prioritized work has higher probability to help with non-prioritized one.
        spin_for(0.1);
    }
}

// Using pointers to nodes to avoid errors on compilers, which try to generate assignment operator
// for the nodes
typedef std::vector< continue_node<continue_msg>* > nodes_container_t;

void create_nodes( nodes_container_t& nodes, graph& g, int num, int body_size ) {
    for( int i = 0; i < num; ++i )
        nodes.push_back(
            new continue_node<continue_msg>( g, CommonBody<NONPRIORITIZED_WORK>( body_size ) )
        );
}

void test( int num_threads ) {
    REMARK( "Testing limit execution to priority tasks (num_threads=%d) - ", num_threads );

    tbb::task_scheduler_init init( num_threads );

    const int nodes_num = 100;
    const int priority_node_position_part = 10;
    const int pivot = nodes_num / priority_node_position_part;
    const int nodes_in_lane = 3 * num_threads;
    const int small_problem_size = 100;
    const int large_problem_size = 1000;

    graph g;
    nodes_container_t nodes;
    create_nodes( nodes, g, pivot, large_problem_size );
    nodes.push_back(
        new continue_node<continue_msg>(
            g, CommonBody<PRIORITIZED_WORK>(small_problem_size), node_priority_t(1)
        )
    );
    create_nodes( nodes, g, nodes_num - pivot - 1, large_problem_size );

    broadcast_node<continue_msg> bn(g);
    for( int i = 0; i < nodes_num; ++i )
        if( i % nodes_in_lane == 0 )
            make_edge( bn, *nodes[i] );
        else
            make_edge( *nodes[i-1], *nodes[i] );
    exec_tracker.reset();
    bn.try_put( continue_msg() );
    g.wait_for_all();

    for( size_t i = 0; i < nodes.size(); ++i )
        delete nodes[i];
    REMARK( "done\n" );
}

} /* namespace LimitingExecutionToPriorityTask */

#include "tbb/task_arena.h"
namespace NestedCase {

using tbb::task_arena;

struct ResetGraphFunctor {
    graph& my_graph;
    ResetGraphFunctor(graph& g) : my_graph(g) {}
    // copy constructor to please some old compilers
    ResetGraphFunctor(const ResetGraphFunctor& rgf) : my_graph(rgf.my_graph) {}
    void operator()() const { my_graph.reset(); }
};

struct InnerBody {
    continue_msg operator()( const continue_msg& ) const {
        return continue_msg();
    }
};

struct OuterBody {
    int my_max_threads;
    task_arena& my_inner_arena;
    OuterBody( int max_threads, task_arena& inner_arena )
        : my_max_threads(max_threads), my_inner_arena(inner_arena) {}
    // copy constructor to please some old compilers
    OuterBody( const OuterBody& rhs )
        : my_max_threads(rhs.my_max_threads), my_inner_arena(rhs.my_inner_arena) {}
    int operator()( const int& ) {
        graph inner_graph;
        continue_node<continue_msg> start_node(inner_graph, InnerBody());
        continue_node<continue_msg> mid_node1(inner_graph, InnerBody(), node_priority_t(5));
        continue_node<continue_msg> mid_node2(inner_graph, InnerBody());
        continue_node<continue_msg> end_node(inner_graph, InnerBody(), node_priority_t(15));
        make_edge( start_node, mid_node1 );
        make_edge( mid_node1, end_node );
        make_edge( start_node, mid_node2 );
        make_edge( mid_node2, end_node );
        my_inner_arena.execute( ResetGraphFunctor(inner_graph) );
        start_node.try_put( continue_msg() );
        inner_graph.wait_for_all();
        return 13;
    }
};

void execute_outer_graph( bool same_arena, task_arena& inner_arena, int max_threads,
                          graph& outer_graph, function_node<int,int>& start_node ) {
    if( same_arena ) {
        start_node.try_put( 42 );
        outer_graph.wait_for_all();
        return;
    }
    for( int num_threads = 1; num_threads <= max_threads; ++num_threads ) {
        inner_arena.initialize( num_threads );
        start_node.try_put( 42 );
        outer_graph.wait_for_all();
        inner_arena.terminate();
    }
}

void test_in_arena( int max_threads, task_arena& outer_arena, task_arena& inner_arena ) {
    graph outer_graph;
    const unsigned num_outer_nodes = 10;
    const size_t concurrency = unlimited;
    std::vector< function_node<int,int>* > outer_nodes;
    for( unsigned node_index = 0; node_index < num_outer_nodes; ++node_index ) {
        internal::node_priority_t priority = internal::no_priority;
        if( node_index == num_outer_nodes / 2 )
            priority = 10;

        outer_nodes.push_back(
            new function_node<int,int>(
                outer_graph, concurrency, OuterBody(max_threads, inner_arena), priority
            )
        );
    }

    for( unsigned node_index1 = 0; node_index1 < num_outer_nodes; ++node_index1 )
        for( unsigned node_index2 = node_index1+1; node_index2 < num_outer_nodes; ++node_index2 )
            make_edge( *outer_nodes[node_index1], *outer_nodes[node_index2] );

    bool same_arena = &outer_arena == &inner_arena;
    for( int num_threads = 1; num_threads <= max_threads; ++num_threads ) {
        REMARK( "Testing nested nodes with specified priority in %s arenas, num_threads=%d) - ",
                same_arena? "same" : "different", num_threads );
        outer_arena.initialize( num_threads );
        outer_arena.execute( ResetGraphFunctor(outer_graph) );
        execute_outer_graph( same_arena, inner_arena, max_threads, outer_graph, *outer_nodes[0] );
        outer_arena.terminate();
        REMARK( "done\n" );
    }

    for( size_t i = 0; i < outer_nodes.size(); ++i )
        delete outer_nodes[i];
}

void test( int max_threads ) {
    tbb::task_scheduler_init init( max_threads );
    task_arena outer_arena; task_arena inner_arena;
    test_in_arena( max_threads, outer_arena, outer_arena );
    test_in_arena( max_threads, outer_arena, inner_arena );
}
}

int TestMain() {
    if( MinThread < 1 ) {
        REPORT( "Number of threads must be positive\n" );
        return Harness::Skipped;
    }
    for( int p = MinThread; p <= MaxThread; ++p ) {
        PriorityNodesTakePrecedence::test( p );
        ThreadsEagerReaction::test( p );
        LimitingExecutionToPriorityTask::test( p );
    }
    NestedCase::test( MaxThread );
    return Harness::Done;
}
#else /* __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES */
#define HARNESS_SKIP_TEST 1
#include "harness.h"
#endif /* __TBB_PREVIEW_FLOW_GRAPH_PRIORITIES */
