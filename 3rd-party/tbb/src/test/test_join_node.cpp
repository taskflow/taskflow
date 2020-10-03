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
#include "test_join_node.h"

static tbb::atomic<int> output_count;

// get the tag from the output tuple and emit it.
// the first tuple component is tag * 2 cast to the type
template<typename OutputTupleType>
class recirc_output_func_body {
public:
    // we only need this to use source_node_helper
    typedef typename tbb::flow::join_node<OutputTupleType, tbb::flow::tag_matching> join_node_type;
    static const int N = tbb::flow::tuple_size<OutputTupleType>::value;
    int operator()(const OutputTupleType &v) {
        int out = int(tbb::flow::get<0>(v))/2;
        source_node_helper<N, join_node_type>::only_check_value(out, v);
        ++output_count;
        return out;
    }
};

template<typename JType>
class tag_recirculation_test {
public:
    typedef typename JType::output_type TType;
    typedef typename tbb::flow::tuple<int, tbb::flow::continue_msg> input_tuple_type;
    typedef tbb::flow::join_node<input_tuple_type, tbb::flow::reserving> input_join_type;
    static const int N = tbb::flow::tuple_size<TType>::value;
    static void test() {
        source_node_helper<N, JType>::print_remark("Recirculation test of tag-matching join");
        REMARK(" >\n");
        for(int maxTag = 1; maxTag <10; maxTag *= 3) {
            for(int i = 0; i < N; ++i) all_source_nodes[i][0] = NULL;

            tbb::flow::graph g;
            // this is the tag-matching join we're testing
            JType * my_join = makeJoin<N, JType, tbb::flow::tag_matching>::create(g);
            // source_node for continue messages
            tbb::flow::input_node<tbb::flow::continue_msg> snode(g, recirc_source_node_body());
            // reserving join that matches recirculating tags with continue messages.
            input_join_type * my_input_join = makeJoin<2, input_join_type, tbb::flow::reserving>::create(g);
            // tbb::flow::make_edge(snode, tbb::flow::input_port<1>(*my_input_join));
            tbb::flow::make_edge(snode, tbb::flow::get<1>(my_input_join->input_ports()));
            // queue to hold the tags
            tbb::flow::queue_node<int> tag_queue(g);
            tbb::flow::make_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
            // add all the function_nodes that are inputs to the tag-matching join
            source_node_helper<N, JType>::add_recirc_func_nodes(*my_join, *my_input_join, g);
            // add the function_node that accepts the output of the join and emits the int tag it was based on
            tbb::flow::function_node<TType, int> recreate_tag(g, tbb::flow::unlimited, recirc_output_func_body<TType>());
            tbb::flow::make_edge(*my_join, recreate_tag);
            // now the recirculating part (output back to the queue)
            tbb::flow::make_edge(recreate_tag, tag_queue);

            // put the tags into the queue
            for(int t = 1; t<=maxTag; ++t) tag_queue.try_put(t);

            input_count = Recirc_count;
            output_count = 0;

            // start up the source node to get things going
            snode.activate();

            // wait for everything to stop
            g.wait_for_all();

            ASSERT(output_count==Recirc_count, "not all instances were received");

            int j;
            // grab the tags from the queue, record them
            std::vector<bool> out_tally(maxTag, false);
            for(int i = 0; i < maxTag; ++i) {
                ASSERT(tag_queue.try_get(j), "not enough tags in queue");
                ASSERT(!out_tally.at(j-1), "duplicate tag from queue");
                out_tally[j-1] = true;
            }
            ASSERT(!tag_queue.try_get(j), "Extra tags in recirculation queue");

            // deconstruct graph
            source_node_helper<N, JType>::remove_recirc_func_nodes(*my_join, *my_input_join);
            tbb::flow::remove_edge(*my_join, recreate_tag);
            makeJoin<N, JType, tbb::flow::tag_matching>::destroy(my_join);
            tbb::flow::remove_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
            tbb::flow::remove_edge(snode, tbb::flow::input_port<1>(*my_input_join));
            makeJoin<2, input_join_type, tbb::flow::reserving>::destroy(my_input_join);
        }
    }
};

template<typename JType>
class generate_recirc_test {
public:
    typedef tbb::flow::join_node<JType, tbb::flow::tag_matching> join_node_type;
    static void do_test() {
        tag_recirculation_test<join_node_type>::test();
    }
};

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#include <array>
#include <vector>
void test_follows_and_precedes_api() {
    using msg_t = tbb::flow::continue_msg;
    using JoinOutputType = tbb::flow::tuple<msg_t, msg_t, msg_t>;

    std::array<msg_t, 3> messages_for_follows = { {msg_t(), msg_t(), msg_t()} };
    std::vector<msg_t> messages_for_precedes = {msg_t(), msg_t(), msg_t()};

    follows_and_precedes_testing::test_follows
        <msg_t, tbb::flow::join_node<JoinOutputType>, tbb::flow::buffer_node<msg_t>>(messages_for_follows);
    follows_and_precedes_testing::test_follows
        <msg_t, tbb::flow::join_node<JoinOutputType, tbb::flow::queueing>>(messages_for_follows);
    follows_and_precedes_testing::test_follows
        <msg_t, tbb::flow::join_node<JoinOutputType, tbb::flow::reserving>, tbb::flow::buffer_node<msg_t>>(messages_for_follows);
    // TODO: add tests for key_matching and message based key matching

    follows_and_precedes_testing::test_precedes
        <msg_t, tbb::flow::join_node<JoinOutputType>>(messages_for_precedes);
    follows_and_precedes_testing::test_precedes
        <msg_t, tbb::flow::join_node<JoinOutputType, tbb::flow::queueing>>(messages_for_precedes);
    follows_and_precedes_testing::test_precedes
        <msg_t, tbb::flow::join_node<JoinOutputType, tbb::flow::reserving>>(messages_for_precedes);
}
#endif

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
void test_deduction_guides() {
    using namespace tbb::flow;

    graph g;
    using tuple_type = std::tuple<int, int, int>;
    broadcast_node<int> b1(g), b2(g), b3(g);
    broadcast_node<tuple_type> b4(g);
    join_node<tuple_type> j0(g);

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    join_node j1(follows(b1, b2, b3));
    static_assert(std::is_same_v<decltype(j1), join_node<tuple_type>>);

    join_node j2(follows(b1, b2, b3), reserving());
    static_assert(std::is_same_v<decltype(j2), join_node<tuple_type, reserving>>);

    join_node j3(precedes(b4));
    static_assert(std::is_same_v<decltype(j3), join_node<tuple_type>>);

    join_node j4(precedes(b4), reserving());
    static_assert(std::is_same_v<decltype(j4), join_node<tuple_type, reserving>>);
#endif

    join_node j5(j0);
    static_assert(std::is_same_v<decltype(j5), join_node<tuple_type>>);
}

#endif

int TestMain() {
#if __TBB_USE_TBB_TUPLE
    REMARK("  Using TBB tuple\n");
#else
    REMARK("  Using platform tuple\n");
#endif
#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    test_follows_and_precedes_api();
#endif
#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    test_deduction_guides();
#endif
    TestTaggedBuffers();
    test_main<tbb::flow::queueing>();
    test_main<tbb::flow::reserving>();
    test_main<tbb::flow::tag_matching>();
    generate_recirc_test<tbb::flow::tuple<int,float> >::do_test();
    return Harness::Done;
}
