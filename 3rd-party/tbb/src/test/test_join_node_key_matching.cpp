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

#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "test_join_node.h"

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
void test_deduction_guides() {
    using namespace tbb::flow;
    using tuple_type = std::tuple<int, int, double>;

    graph g;
    auto body_int = [](const int&)->int { return 1; };
    auto body_double = [](const double&)->int { return 1; };

    join_node j1(g, body_int, body_int, body_double);
    static_assert(std::is_same_v<decltype(j1), join_node<tuple_type, key_matching<int>>>);

#if __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
    broadcast_node<int> b1(g), b2(g);
    broadcast_node<double> b3(g);
    broadcast_node<tuple_type> b4(g);

    join_node j2(follows(b1, b2, b3), body_int, body_int, body_double);
    static_assert(std::is_same_v<decltype(j2), join_node<tuple_type, key_matching<int>>>);

    join_node j3(precedes(b4), body_int, body_int, body_double);
    static_assert(std::is_same_v<decltype(j3), join_node<tuple_type, key_matching<int>>>);
#endif

    join_node j4(j1);
    static_assert(std::is_same_v<decltype(j4), join_node<tuple_type, key_matching<int>>>);
}
#endif

int TestMain() {
#if __TBB_USE_TBB_TUPLE
    REMARK("  Using TBB tuple\n");
#else
    REMARK("  Using platform tuple\n");
#endif

    REMARK("key_matching\n");
    generate_test<serial_test, tbb::flow::tuple<MyKeyFirst<int, double>, MyKeySecond<int, float> >, tbb::flow::key_matching<int> >::do_test();
    generate_test<serial_test, tbb::flow::tuple<MyKeyFirst<std::string, double>, MyKeySecond<std::string, float> >, tbb::flow::key_matching<std::string> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
    generate_test<serial_test, tbb::flow::tuple<MyKeyFirst<std::string, double>, MyKeySecond<std::string, float>, MyKeyWithBrokenMessageKey<std::string, int> >, tbb::flow::key_matching<std::string&> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
    generate_test<serial_test, tbb::flow::tuple<
        MyKeyFirst<std::string, double>,
        MyKeyWithBrokenMessageKey<std::string, int>,
        MyKeyFirst<std::string, int>,
        MyKeySecond<std::string, size_t>,
        MyKeyWithBrokenMessageKey<std::string, int>,
        MyKeySecond<std::string, short>,
        MyKeySecond<std::string, threebyte>
    >, tbb::flow::key_matching<std::string&> >::do_test();
#endif

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
    test_deduction_guides();
#endif

    generate_test<parallel_test, tbb::flow::tuple<MyKeyFirst<int, double>, MyKeySecond<int, float> >, tbb::flow::key_matching<int> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyKeyFirst<int, double>, MyKeySecond<int, float> >, tbb::flow::key_matching<int&> >::do_test();
    generate_test<parallel_test, tbb::flow::tuple<MyKeyFirst<std::string, double>, MyKeySecond<std::string, float> >, tbb::flow::key_matching<std::string&> >::do_test();

#if MAX_TUPLE_TEST_SIZE >= 10
    generate_test<parallel_test, tbb::flow::tuple<
        MyKeyFirst<std::string, double>,
        MyKeySecond<std::string, int>,
        MyKeyFirst<std::string, int>,
        MyKeyWithBrokenMessageKey<std::string, size_t>,
        MyKeyWithBrokenMessageKey<std::string, int>,
        MyKeySecond<std::string, short>,
        MyKeySecond<std::string, threebyte>,
        MyKeyFirst<std::string, int>,
        MyKeySecond<std::string, threebyte>,
        MyKeyWithBrokenMessageKey<std::string, size_t>
    >, tbb::flow::key_matching<std::string&> >::do_test();
#endif

    return Harness::Done;
}
