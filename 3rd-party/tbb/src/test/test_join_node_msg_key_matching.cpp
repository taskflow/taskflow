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

// Message based key matching is a preview feature
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

// This preview feature depends on
// TBB_PREVIEW_FLOW_GRAPH_FEATURES macro, and should not accidentally be dependent on
// this deprecated feature
#define TBB_DEPRECATED_FLOW_NODE_EXTRACTION 0

#include "test_join_node.h"

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
struct message_key {
    int my_key;
    double my_value;

    int key() const { return my_key; }

    operator size_t() const { return my_key; }

    bool operator==(const message_key& rhs) { return my_value == rhs.my_value; }
};

void test_deduction_guides() {
    using namespace tbb::flow;
    using tuple_type = std::tuple<message_key, message_key>;

    graph g;
    broadcast_node<message_key> bm1(g), bm2(g);
    broadcast_node<tuple_type> bm3(g);
    join_node<tuple_type, key_matching<int> > j0(g);

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    join_node j1(follows(bm1, bm2), key_matching<int>());
    static_assert(std::is_same_v<decltype(j1), join_node<tuple_type, key_matching<int>>>);

    join_node j2(precedes(bm3), key_matching<int>());
    static_assert(std::is_same_v<decltype(j2), join_node<tuple_type, key_matching<int>>>);
#endif

    join_node j3(j0);
    static_assert(std::is_same_v<decltype(j3), join_node<tuple_type, key_matching<int>>>);
}
#endif

int TestMain() {
#if __TBB_USE_TBB_TUPLE
    REMARK("  Using TBB tuple\n");
#else
    REMARK("  Using platform tuple\n");
#endif

#if !__TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN
    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithBrokenKey<int, double>, MyMessageKeyWithoutKey<int, float> >, message_based_key_matching<int> >::do_test();
    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<std::string, double>, MyMessageKeyWithBrokenKey<std::string, float> >, message_based_key_matching<std::string> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithoutKey<std::string, double>, MyMessageKeyWithoutKeyMethod<std::string, float>, MyMessageKeyWithBrokenKey<std::string, int> >, message_based_key_matching<std::string&> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
    generate_test<serial_test, tbb::flow::tuple<
        MyMessageKeyWithoutKey<std::string, double>,
        MyMessageKeyWithoutKeyMethod<std::string, int>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKey<std::string, size_t>,
        MyMessageKeyWithoutKeyMethod<std::string, int>,
        MyMessageKeyWithBrokenKey<std::string, short>,
        MyMessageKeyWithoutKey<std::string, threebyte>
    >, message_based_key_matching<std::string&> >::do_test();
#endif

    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithBrokenKey<int, double>, MyMessageKeyWithoutKey<int, float> >, message_based_key_matching<int> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<int, double>, MyMessageKeyWithBrokenKey<int, float> >, message_based_key_matching<int&> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyMessageKeyWithoutKey<std::string, double>, MyMessageKeyWithoutKeyMethod<std::string, float> >, message_based_key_matching<std::string&> >::do_test();


#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    test_deduction_guides();
#endif

#if MAX_TUPLE_TEST_SIZE >= 10
    generate_test<parallel_test, tbb::flow::tuple<
        MyMessageKeyWithoutKeyMethod<std::string, double>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, size_t>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, short>,
        MyMessageKeyWithoutKeyMethod<std::string, threebyte>,
        MyMessageKeyWithBrokenKey<std::string, int>,
        MyMessageKeyWithoutKeyMethod<std::string, threebyte>,
        MyMessageKeyWithBrokenKey<std::string, size_t>
    >, message_based_key_matching<std::string&> >::do_test();
#endif
#endif /* __TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN */

    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithBrokenKey<int, double>, MyMessageKeyWithoutKey<int, float> >, message_based_key_matching<int> >::do_test();
    generate_test<serial_test, tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<std::string, double>, MyMessageKeyWithBrokenKey<std::string, float> >, message_based_key_matching<std::string> >::do_test();

    return Harness::Done;
}
