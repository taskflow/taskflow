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

#define TBB_DEPRECATED_FLOW_NODE_ALLOCATOR __TBB_CPF_BUILD

#include "harness.h"
#include "harness_graph.h"
#include "harness_barrier.h"
#include "tbb/concurrent_queue.h"
#include "tbb/flow_graph.h"
#include "tbb/task.h"
#include "tbb/tbb_thread.h"
#include "tbb/mutex.h"
#include "tbb/compat/condition_variable"

#include <string>

class minimal_type {
    template<typename T>
    friend struct place_wrapper;

    int value;

public:
    minimal_type() : value(-1) {}
    minimal_type(int v) : value(v) {}
    minimal_type(const minimal_type &m) : value(m.value) { }
    minimal_type &operator=(const minimal_type &m) { value = m.value; return *this; }
};

template <typename T>
struct place_wrapper {
    typedef T wrapped_type;
    T value;
    tbb::tbb_thread::id thread_id;
    tbb::task* task_ptr;

    place_wrapper( ) : value(0) {
        thread_id = tbb::this_tbb_thread::get_id();
        task_ptr = &tbb::task::self();
    }
    place_wrapper( int v ) : value(v) {
        thread_id = tbb::this_tbb_thread::get_id();
        task_ptr = &tbb::task::self();
    }

    place_wrapper( const place_wrapper<int> &v ) : value(v.value), thread_id(v.thread_id), task_ptr(v.task_ptr) { }

    place_wrapper( const place_wrapper<minimal_type> &v ) : value(v.value), thread_id(v.thread_id), task_ptr(v.task_ptr) { }

    place_wrapper<minimal_type>& operator=(const place_wrapper<minimal_type> &v) {
        if( this != &v ) {
            value = v.value;
            thread_id = v.thread_id;
            task_ptr = v.task_ptr;
        }
        return *this;
    }
};

template<typename T1, typename T2>
struct wrapper_helper {
    static void check(const T1 &, const T2 &) { }

    static void copy_value(const T1 &in, T2 &out) {
        out = in;
    }
};

template<typename T1, typename T2>
struct wrapper_helper< place_wrapper<T1>, place_wrapper<T2> > {
    static void check(const place_wrapper<T1> &a, const place_wrapper<T2> &b) {
       REMARK("a.task_ptr == %p != b.task_ptr == %p\n", a.task_ptr, b.task_ptr);
       ASSERT( (a.thread_id != b.thread_id), "same thread used to execute adjacent nodes");
       ASSERT( (a.task_ptr != b.task_ptr), "same task used to execute adjacent nodes");
       return;
    }
    static void copy_value(const place_wrapper<T1> &in, place_wrapper<T2> &out) {
        out.value = in.value;
    }
};

const int NUMBER_OF_MSGS = 10;
const int UNKNOWN_NUMBER_OF_ITEMS = -1;
tbb::atomic<int> async_body_exec_count;
tbb::atomic<int> async_activity_processed_msg_count;
tbb::atomic<int> end_body_exec_count;

// queueing required in test_reset for testing of cancellation
typedef tbb::flow::async_node< int, int, tbb::flow::queueing > counting_async_node_type;
typedef counting_async_node_type::gateway_type counting_gateway_type;

struct counting_async_body {
    tbb::atomic<int> my_async_body_exec_count;

    counting_async_body() {
        my_async_body_exec_count = 0;
    }

    void operator()( const int &input, counting_gateway_type& gateway) {
        REMARK( "Body execution with input == %d\n", input);
        ++my_async_body_exec_count;
        ++async_body_exec_count;
        if ( input == -1 ) {
            bool result = tbb::task::self().group()->cancel_group_execution();
            REMARK( "Canceling graph execution\n" );
            ASSERT( result == true, "attempted to cancel graph twice" );
            Harness::Sleep(50);
        }
        gateway.try_put(input);
    }
};

void test_reset() {
    const int N = NUMBER_OF_MSGS;
    async_body_exec_count = 0;

    tbb::flow::graph g;
    counting_async_node_type a(g, tbb::flow::serial, counting_async_body() );

    const int R = 3;
    std::vector< harness_counting_receiver<int> > r(R, harness_counting_receiver<int>(g));

    for (int i = 0; i < R; ++i) {
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
        tbb::flow::make_edge(a, r[i]);
#else
        tbb::flow::make_edge( tbb::flow::output_port<0>(a), r[i] );
#endif
    }

    REMARK( "One body execution\n" );
    a.try_put(-1);
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    // should be canceled with only 1 item reaching the async_body and the counting receivers
    // and N items left in the node's queue
    ASSERT( g.is_cancelled() == true, "task group not canceled" );

    counting_async_body b1 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == int(b1.my_async_body_exec_count), "body and global body counts are different" );
    ASSERT( int(async_body_exec_count) == 1, "global body execution count not 1"  );
    for (int i = 0; i < R; ++i) {
        ASSERT( int(r[i].my_count) == 1, "counting receiver count not 1" );
    }

    // should clear the async_node queue, but retain its local count at 1 and keep all edges
    g.reset(tbb::flow::rf_reset_protocol);

    REMARK( "N body executions\n" );
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    ASSERT( g.is_cancelled() == false, "task group not canceled" );

    // a total of N+1 items should have passed through the node body
    // the local body count should also be N+1
    // and the counting receivers should all have a count of N+1
    counting_async_body b2 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == int(b2.my_async_body_exec_count), "local and global body execution counts are different" );
    REMARK( "async_body_exec_count==%d\n", int(async_body_exec_count) );
    ASSERT( int(async_body_exec_count) == N+1, "globcal body execution count not N+1"  );
    for (int i = 0; i < R; ++i) {
        ASSERT( int(r[i].my_count) == N+1, "counting receiver has not received N+1 items" );
    }

    REMARK( "N body executions with new bodies\n" );
    // should clear the async_node queue and reset its local count to 0, but keep all edges
    g.reset(tbb::flow::rf_reset_bodies);
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    ASSERT( g.is_cancelled() == false, "task group not canceled" );

    // a total of 2N+1 items should have passed through the node body
    // the local body count should be N
    // and the counting receivers should all have a count of 2N+1
    counting_async_body b3 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == 2*N+1, "global body execution count not 2N+1"  );
    ASSERT( int(b3.my_async_body_exec_count) == N, "local body execution count not N"  );
    for (int i = 0; i < R; ++i) {
        ASSERT( int(r[i].my_count) == 2*N+1, "counting receiver has not received 2N+1 items" );
    }

    // should clear the async_node queue and keep its local count at N and remove all edges
    REMARK( "N body executions with no edges\n" );
    g.reset(tbb::flow::rf_clear_edges);
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    ASSERT( g.is_cancelled() == false, "task group not canceled" );

    // a total of 3N+1 items should have passed through the node body
    // the local body count should now be 2*N
    // and the counting receivers should remain at a count of 2N+1
    counting_async_body b4 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == 3*N+1, "global body execution count not 3N+1"  );
    ASSERT( int(b4.my_async_body_exec_count) == 2*N, "local body execution count not 2N"  );
    for (int i = 0; i < R; ++i) {
        ASSERT( int(r[i].my_count) == 2*N+1, "counting receiver has not received 2N+1 items" );
    }

    // put back 1 edge to receiver 0
    REMARK( "N body executions with 1 edge\n" );
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
    tbb::flow::make_edge(a, r[0]);
#else
    tbb::flow::make_edge( tbb::flow::output_port<0>(a), r[0] );
#endif
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    ASSERT( g.is_cancelled() == false, "task group not canceled" );

    // a total of 4N+1 items should have passed through the node body
    // the local body count should now be 3*N
    // and all of the counting receivers should remain at a count of 2N+1, except r[0] which should be 3N+1
    counting_async_body b5 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == 4*N+1, "global body execution count not 4N+1"  );
    ASSERT( int(b5.my_async_body_exec_count) == 3*N, "local body execution count not 3N"  );
    ASSERT( int(r[0].my_count) == 3*N+1, "counting receiver has not received 3N+1 items" );
    for (int i = 1; i < R; ++i) {
        ASSERT( int(r[i].my_count) == 2*N+1, "counting receiver has not received 2N+1 items" );
    }

    // should clear the async_node queue and keep its local count at N and remove all edges
    REMARK( "N body executions with no edges and new body\n" );
    g.reset(static_cast<tbb::flow::reset_flags>(tbb::flow::rf_reset_bodies|tbb::flow::rf_clear_edges));
    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();
    ASSERT( g.is_cancelled() == false, "task group not canceled" );

    // a total of 4N+1 items should have passed through the node body
    // the local body count should now be 3*N
    // and all of the counting receivers should remain at a count of 2N+1, except r[0] which should be 3N+1
    counting_async_body b6 = tbb::flow::copy_body<counting_async_body>(a);
    ASSERT( int(async_body_exec_count) == 5*N+1, "global body execution count not 5N+1"  );
    ASSERT( int(b6.my_async_body_exec_count) == N, "local body execution count not N"  );
    ASSERT( int(r[0].my_count) == 3*N+1, "counting receiver has not received 3N+1 items" );
    for (int i = 1; i < R; ++i) {
        ASSERT( int(r[i].my_count) == 2*N+1, "counting receiver has not received 2N+1 items" );
    }
}

template< typename Input, typename Output >
class async_activity : NoAssign {
public:
    typedef Input input_type;
    typedef Output output_type;
    typedef tbb::flow::async_node< input_type, output_type > async_node_type;
    typedef typename async_node_type::gateway_type gateway_type;

    struct work_type {
        input_type input;
        gateway_type* gateway;
    };

    class ServiceThreadBody {
    public:
        ServiceThreadBody( async_activity* activity ) : my_activity( activity ) {}

        void operator()() {
            my_activity->process();
        }
    private:
        async_activity* my_activity;
    };

    async_activity(int expected_items, bool deferred = false, int sleep_time = 50)
        : my_expected_items(expected_items), my_sleep_time(sleep_time) {
        is_active = !deferred;
        my_quit = false;
        tbb::tbb_thread( ServiceThreadBody( this ) ).swap( my_service_thread );
    }

private:

    async_activity( const async_activity& )
        : my_expected_items(UNKNOWN_NUMBER_OF_ITEMS), my_sleep_time(0) {
        is_active = true;
    }

public:
    ~async_activity() {
        stop();
        my_service_thread.join();
    }

    void submit( const input_type &input, gateway_type& gateway ) {
        work_type work = { input, &gateway};
        my_work_queue.push( work );
    }

    void process() {
        do {
            work_type work;
            if( is_active && my_work_queue.try_pop( work ) ) {
                Harness::Sleep(my_sleep_time);
                ++async_activity_processed_msg_count;
                output_type output;
                wrapper_helper<output_type, output_type>::copy_value(work.input, output);
                wrapper_helper<output_type, output_type>::check(work.input, output);
                work.gateway->try_put(output);
                if ( my_expected_items == UNKNOWN_NUMBER_OF_ITEMS ||
                     int(async_activity_processed_msg_count) == my_expected_items ) {
                    work.gateway->release_wait();
                }
            }
        } while( my_quit == false || !my_work_queue.empty());
    }

    void stop() {
        my_quit = true;
    }

    void activate() {
        is_active = true;
    }

    bool should_reserve_each_time() {
        if ( my_expected_items == UNKNOWN_NUMBER_OF_ITEMS )
            return true;
        else
            return false;
    }

private:

    const int my_expected_items;
    const int my_sleep_time;
    tbb::atomic< bool > is_active;

    tbb::concurrent_queue< work_type > my_work_queue;

    tbb::atomic< bool > my_quit;

    tbb::tbb_thread my_service_thread;
};

template<typename Input, typename Output>
struct basic_test {
    typedef Input input_type;
    typedef Output output_type;
    typedef tbb::flow::async_node< input_type, output_type > async_node_type;
    typedef typename async_node_type::gateway_type gateway_type;

    class start_body_type {
        typedef Input input_type;
    public:
        input_type operator()( int input ) {
            return input_type(input);
        }
    };

#if !__TBB_CPP11_LAMBDAS_PRESENT
    class async_body_type {
        typedef Input input_type;
        typedef Output output_type;
        typedef tbb::flow::async_node< input_type, output_type > async_node_type;
        typedef typename async_node_type::gateway_type gateway_type;
    public:
        typedef async_activity<input_type, output_type> async_activity_type;

        async_body_type( async_activity_type* aa ) : my_async_activity( aa ) { }

        async_body_type( const async_body_type& other ) : my_async_activity( other.my_async_activity ) { }

        void operator()( const input_type &input, gateway_type& gateway ) {
            ++async_body_exec_count;
            my_async_activity->submit( input, gateway);
            if ( my_async_activity->should_reserve_each_time() )
                gateway.reserve_wait();
        }

    private:
        async_activity_type* my_async_activity;
    };
#endif

    class end_body_type {
        typedef Output output_type;
    public:
        void operator()( const output_type &input ) {
            ++end_body_exec_count;
            output_type output;
            wrapper_helper<output_type, output_type>::check(input, output);
        }
    };

    basic_test() {}

public:

    static int run(int async_expected_items = UNKNOWN_NUMBER_OF_ITEMS) {
        async_activity<input_type, output_type> my_async_activity(async_expected_items);
        tbb::flow::graph g;
        tbb::flow::function_node< int, input_type > start_node( g, tbb::flow::unlimited, start_body_type() );
#if __TBB_CPP11_LAMBDAS_PRESENT
        async_node_type offload_node(g, tbb::flow::unlimited, [&] (const input_type &input, gateway_type& gateway) {
            ++async_body_exec_count;
            my_async_activity.submit(input, gateway);
            if(my_async_activity.should_reserve_each_time())
                gateway.reserve_wait();
        } );
#else
        async_node_type offload_node( g, tbb::flow::unlimited, async_body_type( &my_async_activity ) );
#endif

        tbb::flow::function_node< output_type > end_node( g, tbb::flow::unlimited, end_body_type() );

        tbb::flow::make_edge( start_node, offload_node );
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
        tbb::flow::make_edge( offload_node, end_node );
#else
        tbb::flow::make_edge( tbb::flow::output_port<0>(offload_node), end_node );
#endif
        async_body_exec_count = 0;
        async_activity_processed_msg_count = 0;
        end_body_exec_count = 0;

        if (async_expected_items != UNKNOWN_NUMBER_OF_ITEMS ) {
            offload_node.gateway().reserve_wait();
        }
        for (int i = 0; i < NUMBER_OF_MSGS; ++i) {
            start_node.try_put(i);
        }
        g.wait_for_all();
        ASSERT( async_body_exec_count == NUMBER_OF_MSGS, "AsyncBody processed wrong number of signals" );
        ASSERT( async_activity_processed_msg_count == NUMBER_OF_MSGS, "AsyncActivity processed wrong number of signals" );
        ASSERT( end_body_exec_count == NUMBER_OF_MSGS, "EndBody processed wrong number of signals");
        REMARK("async_body_exec_count == %d == async_activity_processed_msg_count == %d == end_body_exec_count == %d\n",
                int(async_body_exec_count), int(async_activity_processed_msg_count), int(end_body_exec_count));
        return Harness::Done;
    }

};

int test_copy_ctor() {
    const int N = NUMBER_OF_MSGS;
    async_body_exec_count = 0;

    tbb::flow::graph g;

    harness_counting_receiver<int> r1(g);
    harness_counting_receiver<int> r2(g);

    counting_async_node_type a(g, tbb::flow::unlimited, counting_async_body() );
    counting_async_node_type b(a);
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
    tbb::flow::make_edge(a, r1);
    tbb::flow::make_edge(b, r2);
#else
    tbb::flow::make_edge(tbb::flow::output_port<0>(a), r1);
    tbb::flow::make_edge(tbb::flow::output_port<0>(b), r2);
#endif

    for (int i = 0; i < N; ++i) {
       a.try_put(i);
    }
    g.wait_for_all();

    REMARK("async_body_exec_count = %d\n", int(async_body_exec_count));
    REMARK("r1.my_count == %d and r2.my_count = %d\n", int(r1.my_count), int(r2.my_count));
    ASSERT( int(async_body_exec_count) == NUMBER_OF_MSGS, "AsyncBody processed wrong number of signals" );
    ASSERT( int(r1.my_count) == N, "counting receiver r1 has not received N items" );
    ASSERT( int(r2.my_count) == 0, "counting receiver r2 has not received 0 items" );

    for (int i = 0; i < N; ++i) {
       b.try_put(i);
    }
    g.wait_for_all();

    REMARK("async_body_exec_count = %d\n", int(async_body_exec_count));
    REMARK("r1.my_count == %d and r2.my_count = %d\n", int(r1.my_count), int(r2.my_count));
    ASSERT( int(async_body_exec_count) == 2*NUMBER_OF_MSGS, "AsyncBody processed wrong number of signals" );
    ASSERT( int(r1.my_count) == N, "counting receiver r1 has not received N items" );
    ASSERT( int(r2.my_count) == N, "counting receiver r2 has not received N items" );
    return Harness::Done;
}

tbb::atomic<int> main_tid_count;

template<typename Input, typename Output>
struct spin_test {
    typedef Input input_type;
    typedef Output output_type;
    typedef tbb::flow::async_node< input_type, output_type > async_node_type;
    typedef typename async_node_type::gateway_type gateway_type;

    class start_body_type {
        typedef Input input_type;
    public:
        input_type operator()( int input ) {
            return input_type(input);
        }
    };

#if !__TBB_CPP11_LAMBDAS_PRESENT
    class async_body_type {
        typedef Input input_type;
        typedef Output output_type;
        typedef tbb::flow::async_node< input_type, output_type > async_node_type;
        typedef typename async_node_type::gateway_type gateway_type;
    public:
        typedef async_activity<input_type, output_type> async_activity_type;

        async_body_type( async_activity_type* aa ) : my_async_activity( aa ) { }

        async_body_type( const async_body_type& other ) : my_async_activity( other.my_async_activity ) { }

        void operator()(const input_type &input, gateway_type& gateway) {
            ++async_body_exec_count;
            my_async_activity->submit(input, gateway);
            if(my_async_activity->should_reserve_each_time())
                gateway.reserve_wait();
        }

    private:
        async_activity_type* my_async_activity;
    };
#endif

    class end_body_type {
        typedef Output output_type;
        tbb::tbb_thread::id my_main_tid;
        Harness::SpinBarrier *my_barrier;
    public:
        end_body_type(tbb::tbb_thread::id t, Harness::SpinBarrier &b) : my_main_tid(t), my_barrier(&b) { }

        void operator()( const output_type & ) {
            ++end_body_exec_count;
            if (tbb::this_tbb_thread::get_id() == my_main_tid) {
               ++main_tid_count;
            }
            my_barrier->timed_wait_noerror(10);
        }
    };

    spin_test() {}

    static int run(int nthreads, int async_expected_items = UNKNOWN_NUMBER_OF_ITEMS) {
        async_activity<input_type, output_type> my_async_activity(async_expected_items, false, 0);
        Harness::SpinBarrier spin_barrier(nthreads);
        tbb::flow::graph g;
        tbb::flow::function_node< int, input_type > start_node( g, tbb::flow::unlimited, start_body_type() );
#if __TBB_CPP11_LAMBDAS_PRESENT
        async_node_type offload_node(g, tbb::flow::unlimited, [&](const input_type &input, gateway_type& gateway) {
            ++async_body_exec_count;
            my_async_activity.submit(input, gateway);
            if(my_async_activity.should_reserve_each_time())
                gateway.reserve_wait();
        });
#else
        async_node_type offload_node( g, tbb::flow::unlimited, async_body_type( &my_async_activity ) );
#endif
        tbb::flow::function_node< output_type > end_node( g, tbb::flow::unlimited, end_body_type(tbb::this_tbb_thread::get_id(), spin_barrier) );
        tbb::flow::make_edge( start_node, offload_node );
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
        tbb::flow::make_edge( offload_node, end_node );
#else
        tbb::flow::make_edge( tbb::flow::output_port<0>(offload_node), end_node );
#endif
        async_body_exec_count = 0;
        async_activity_processed_msg_count = 0;
        end_body_exec_count = 0;
        main_tid_count = 0;

        if (async_expected_items != UNKNOWN_NUMBER_OF_ITEMS ) {
            offload_node.gateway().reserve_wait();
        }
        for (int i = 0; i < nthreads*NUMBER_OF_MSGS; ++i) {
            start_node.try_put(i);
        }
        g.wait_for_all();
        ASSERT( async_body_exec_count == nthreads*NUMBER_OF_MSGS, "AsyncBody processed wrong number of signals" );
        ASSERT( async_activity_processed_msg_count == nthreads*NUMBER_OF_MSGS, "AsyncActivity processed wrong number of signals" );
        ASSERT( end_body_exec_count == nthreads*NUMBER_OF_MSGS, "EndBody processed wrong number of signals");
        ASSERT_WARNING( main_tid_count != 0, "Main thread did not participate in end_body tasks");
        REMARK("async_body_exec_count == %d == async_activity_processed_msg_count == %d == end_body_exec_count == %d\n",
                int(async_body_exec_count), int(async_activity_processed_msg_count), int(end_body_exec_count));
        return Harness::Done;
    }

};

void test_for_spin_avoidance() {
    spin_test<int, int>::run(4);
}

template< typename Input, typename Output >
int run_tests() {
    basic_test<Input, Output>::run();
    basic_test<Input, Output>::run(NUMBER_OF_MSGS);
    basic_test<place_wrapper<Input>, place_wrapper<Output> >::run();
    basic_test<place_wrapper<Input>, place_wrapper<Output> >::run(NUMBER_OF_MSGS);
    return Harness::Done;
}

#include "tbb/parallel_for.h"
template<typename Input, typename Output>
class equeueing_on_inner_level {
    typedef Input input_type;
    typedef Output output_type;
    typedef async_activity<input_type, output_type> async_activity_type;
    typedef tbb::flow::async_node<Input, Output> async_node_type;
    typedef typename async_node_type::gateway_type gateway_type;

    class start_body_type {
    public:
        input_type operator() ( int input ) {
            return input_type( input);
        }
    };

    class async_body_type {
    public:
        async_body_type( async_activity_type& activity ) : my_async_activity(&activity) {}

        void operator() ( const input_type &input, gateway_type& gateway ) {
            gateway.reserve_wait();
            my_async_activity->submit( input, gateway );
        }
    private:
        async_activity_type* my_async_activity;
    };

    class end_body_type {
    public:
        void operator()( output_type ) {}
    };

    class body_graph_with_async {
    public:
        body_graph_with_async( Harness::SpinBarrier& barrier, async_activity_type& activity )
            : spin_barrier(&barrier), my_async_activity(&activity) {}

        void operator()(int) const {
            tbb::flow::graph g;
            tbb::flow::function_node< int, input_type > start_node( g, tbb::flow::unlimited, start_body_type() );

            async_node_type offload_node( g, tbb::flow::unlimited, async_body_type( *my_async_activity ) );

            tbb::flow::function_node< output_type > end_node( g, tbb::flow::unlimited, end_body_type() );

            tbb::flow::make_edge( start_node, offload_node );
            tbb::flow::make_edge( offload_node, end_node );

            start_node.try_put(1);

            spin_barrier->wait();

            my_async_activity->activate();

            g.wait_for_all();
        }

    private:
        Harness::SpinBarrier* spin_barrier;
        async_activity_type* my_async_activity;
    };


public:
    static int run ()
    {
        const int nthreads = tbb::this_task_arena::max_concurrency();
        Harness::SpinBarrier spin_barrier( nthreads );

        async_activity_type my_async_activity( UNKNOWN_NUMBER_OF_ITEMS, true );

        tbb::parallel_for( 0, nthreads, body_graph_with_async( spin_barrier, my_async_activity ) );
        return Harness::Done;
    }
};

int run_test_equeueing_on_inner_level() {
    equeueing_on_inner_level<int, int>::run();
    return Harness::Done;
}

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#include <array>
#include <thread>

template<typename NodeType>
class AsyncActivity {
public:
    using gateway_t = typename NodeType::gateway_type;

    struct work_type {
        int input;
        gateway_t* gateway;
    };

    AsyncActivity(size_t limit) : thr([this]() {
        while(!end_of_work()) {
            work_type w;
            while( my_q.try_pop(w) ) {
                int res = do_work(w.input);
                w.gateway->try_put(res);
                w.gateway->release_wait();
                ++c;
            }
        }
    }), stop_limit(limit), c(0) {}

    void submit(int i, gateway_t* gateway) {
        work_type w = {i, gateway};
        gateway->reserve_wait();
        my_q.push(w);
    }

    void wait_for_all() { thr.join(); }

private:
    bool end_of_work() { return c >= stop_limit; }

    int do_work(int& i) { return i + i; }

    tbb::concurrent_queue<work_type> my_q;
    tbb::tbb_thread thr;
    size_t stop_limit;
    size_t c;
};

void test_follows() {
    using namespace tbb::flow;

    using input_t = int;
    using output_t = int;
    using node_t = async_node<input_t, output_t>;

    graph g;

    AsyncActivity<node_t> async_activity(3);

    std::array<broadcast_node<input_t>, 3> preds = {
      {
        broadcast_node<input_t>(g),
        broadcast_node<input_t>(g),
        broadcast_node<input_t>(g)
      }
    };

    node_t node(follows(preds[0], preds[1], preds[2]), unlimited, [&](int input, node_t::gateway_type& gtw) {
        async_activity.submit(input, &gtw);
    });

    buffer_node<output_t> buf(g);
    make_edge(node, buf);

    for(auto& pred: preds) {
        pred.try_put(1);
    }

    g.wait_for_all();
    async_activity.wait_for_all();

    output_t storage;
    ASSERT((buf.try_get(storage) && buf.try_get(storage) && buf.try_get(storage) && !buf.try_get(storage)),
            "Not exact edge quantity was made");
}

void test_precedes() {
    using namespace tbb::flow;

    using input_t = int;
    using output_t = int;
    using node_t = async_node<input_t, output_t>;

    graph g;

    AsyncActivity<node_t> async_activity(1);

    std::array<buffer_node<input_t>, 1> successors = { {buffer_node<input_t>(g)} };

    broadcast_node<input_t> start(g);

    node_t node(precedes(successors[0]), unlimited, [&](int input, node_t::gateway_type& gtw) {
        async_activity.submit(input, &gtw);
    });

    make_edge(start, node);

    start.try_put(1);

    g.wait_for_all();
    async_activity.wait_for_all();

    for(auto& successor : successors) {
        output_t storage;
        ASSERT(successor.try_get(storage) && !successor.try_get(storage),
               "Not exact edge quantity was made");
    }
}

void test_follows_and_precedes_api() {
    test_follows();
    test_precedes();
}
#endif // __TBB_PREVIEW_FLOW_GRAPH_NODE_SET

#if TBB_DEPRECATED_FLOW_NODE_ALLOCATOR
typedef tbb::flow::async_node< int, int, tbb::flow::queueing, std::allocator<int> > async_node_type;

struct async_body {
    void operator()( const int&, async_node_type::gateway_type& ) {}
};

void test_node_allocator() {
    tbb::flow::graph g;
    async_node_type tmp(g, tbb::flow::unlimited, async_body());
}
#endif

int TestMain() {
    tbb::task_scheduler_init init(4);
    run_tests<int, int>();
    run_tests<minimal_type, minimal_type>();
    run_tests<int, minimal_type>();

    lightweight_testing::test<tbb::flow::async_node>(NUMBER_OF_MSGS);

    test_reset();
    test_copy_ctor();
    test_for_spin_avoidance();
    run_test_equeueing_on_inner_level();
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    test_follows_and_precedes_api();
#endif
#if TBB_DEPRECATED_FLOW_NODE_ALLOCATOR
    test_node_allocator();
#endif
    return Harness::Done;
}

