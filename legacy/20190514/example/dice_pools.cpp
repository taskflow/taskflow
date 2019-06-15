/*

2019/02/25 - contributed by Paolo Bolzoni

cpp-taskflow works on directed acyclic graphs.
And here we want to pass information between the flow elements.

To do so, we see the cpp-taskflow arcs as objects where the functions on the
nodes read from or write to.

The function on every node will *read from* the objects on the incoming arcs
and *write to* the objects on the outcoming arcs.

The cpp-taskflow semantics ensures the synchronization.


Nodes without incoming arcs will require the input from somewhere else; instead
nodes without outcoming arcs have to execute some side effects to be useful.


In this example we fill up (in parallel) two vectors of the same size with the
results of a fair percentile die, once done we pick up the maximum values from
each cell. Finally we output the result.

.----------------.
| fill in vector |----|
'----------------'    |->.-------------.     .-----------------.
                         | pick up max |---->| print in stdout |
.----------------.    |->'-------------'     '-----------------'
| fill in vector |----|
'----------------'

The output will be twenty random integer between 1 and 100, that are clearly
not uniform distributed as they favor larger numbers.

The code assumes the taskflow is executed once, when using the Framework
feature the programmer needs care to keep the invariants.

It is then suggested to use const references (eg., vector<int> const&) for the
objects related to the incoming arcs and non-cost references for outcoming
ones.

*/


#include <taskflow/taskflow.hpp>

//All those includes are just to init the mersenne twister
#include <array>
#include <algorithm>
#include <functional>
#include <random>
//until here

#include <vector>
#include <iostream>

std::mt19937 init_mersenne_twister() {
    std::array<std::uint32_t, std::mt19937::state_size> seed_bits{};
    std::random_device real_random{};
    std::generate(seed_bits.begin(), seed_bits.end(), std::ref(real_random));
    std::seed_seq wrapped_seed_bits(seed_bits.begin(), seed_bits.end());

    return std::mt19937(wrapped_seed_bits);
}


class Fill_in_vector {
public:
    Fill_in_vector(std::vector<int>& v, int length)
      : v_{v}, length_{length} {}

    void operator()() {
        auto rng = init_mersenne_twister();
        std::uniform_int_distribution<int> percentile_die(1, 100);

        //the taskflow is used only once, so we can mess up with length_ value
        while (length_ > 0) {
            --length_;
            v_.push_back( percentile_die(rng) );
        }
    }
private:
    std::vector<int>& v_;
    int length_;
};


class Pick_up_max {
public:
    Pick_up_max(std::vector<int>& in1, std::vector<int>& in2, std::vector<int>& out)
      : in1_{in1}, in2_{in2}, out_{out} {}
    void operator()() {
        for (std::vector<int>::size_type i{}, e = in1_.size(); i < e; ++i) {
            in1_[i] = std::max(in1_[i], in2_[i]);
        }
        // the taskflow is executed once, so we avoid one allocation
        out_.swap(in1_);
    }
private:
    std::vector<int>& in1_;
    std::vector<int>& in2_;
    std::vector<int>& out_;
};


class Print {
public:
    Print(std::vector<int> const& v)
      : v_{v} {}

    void operator()() {
        bool first{ true };
        for (auto i : v_) {
            if (!first)  {
                std::cout << ", ";
            }
            std::cout << i;
            first = false;
        }
        std::cout << "\n";
    }
private:
    std::vector<int> const& v_;
};


int main() {
    // Set up the memory for the arcs
    std::vector<int> in1{}, in2{}, out{};

    // Prepare the functors for taskflow
    tf::Taskflow tf;
    auto [
        fill_in_vector1,
        fill_in_vector2,
        pick_up_max,
        print
    ] = tf.emplace(
        Fill_in_vector(in1, 20),
        Fill_in_vector(in2, 20),
        Pick_up_max(in1, in2, out),
        Print(out)
    );

    // Set up the dependencies
    fill_in_vector1.precede(pick_up_max);
    fill_in_vector2.precede(pick_up_max);
    pick_up_max.precede(print);

    // Execution
    tf.wait_for_all();

    return 0;
}

