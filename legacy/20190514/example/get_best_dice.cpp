/*

This example shows the usage of the dataflow template classes.

Similarly to the percentile dice example, we have multiple dice pools, each
contains 20 values, for each pool we want the maximum result of the position..

So for each pool we throw 20 dice and fill up (in parallel) the vectors with
the results, wait all the dice have been thrown, for each position in the
vectors we pick up the maximum result, finally we write in stdout.

So, with only one pool, on screen we will get 20 percentile results uniformily
distributed. The more pools we use, the more the results will favor higher
results.


.----------------.
| fill in vector |----|
'----------------'    |
...                  ...
.----------------.    |
| fill in vector |----|
'----------------'    |->.-------------.     .-----------------.
...                  ... | pick up max |---->| print in stdout |
.----------------.    |->'-------------'     '-----------------'
| fill in vector |----|
'----------------'    |
...                  ...
.----------------.    |
| fill in vector |----|
'----------------'

*/

#include "dataflow.hpp"

#include <vector>

//All those includes are just to init the mersenne twister
#include <array>
#include <algorithm>
#include <functional>
#include <random>
//until here


// We set up all types
using Dice_pool = std::vector<int>;

using Dataflow_generator = df::Dataflow_generator<Dice_pool>;
using Node = df::Node<Dice_pool>;

using Pad_id = df::Pad_id<Dice_pool>;
using Node_id = df::Node_id<Dice_pool>;
using Transform_f = df::Transform_f<Dice_pool>;


std::mt19937 init_mersenne_twister() {
    std::array<std::uint32_t, std::mt19937::state_size> seed_bits{};
    std::random_device real_random{};
    std::generate(seed_bits.begin(), seed_bits.end(), std::ref(real_random));
    std::seed_seq wrapped_seed_bits(seed_bits.begin(), seed_bits.end());

    return std::mt19937(wrapped_seed_bits);
}


struct Fill_up_vector {
    Fill_up_vector()
      : rng_{ init_mersenne_twister() } {}

    void operator()(Node& node) {
        Node_id opad_id = node.opad_list()[0]; // This kind of nodes have exactly one output pad
        Dice_pool& dice_pool = node.opad(opad_id);

        dice_pool.resize(20);
        std::uniform_int_distribution die(1,100);

        for (auto& res : dice_pool) {
            res = die( rng_ );
        }
    }

    std::mt19937 rng_;
};


void pick_up_max(Node& node) {
    std::vector<Node_id> const& dice_pool_ids = node.ipad_list(); // there can be many input pads
    Node_id opad_id = node.opad_list()[0]; // Output pad

    Dice_pool& best_results_pool = node.opad(opad_id);
    best_results_pool.resize(20);

    for (std::size_t i{}, e{20}; i < e; ++i) {
        int maximum{ -1 };
        for (Node_id dice_pool_id: dice_pool_ids) {
            if (node.ipad(dice_pool_id)[i] > maximum) {
                maximum = node.ipad(dice_pool_id)[i];
            }
        }
        best_results_pool[i] = maximum;
    }
}


void write_out(Node& node) {
    Node_id ipad_id = node.ipad_list()[0]; // The input pad
    Dice_pool const& best_results_pool = node.ipad(ipad_id);

    bool first{ true };
    for (auto const& result : best_results_pool) {
        if (first) {
            first = false;
        } else {
            std::cout << ", ";
        }
        std::cout << result;
    }
    std::cout << "\n";
}


int main() {

  Dataflow_generator gg{};

  auto output = gg.create_node(write_out);
  auto pickup = gg.create_node(pick_up_max);
  gg.create_arc(pickup, output);

  //try to increase or reduce the number of dice pools
  int nr_dice_pools{ 5 };
  while (nr_dice_pools-- > 0) {
      gg.create_arc(gg.create_node(Fill_up_vector{}), pickup);
  }

  int times{ 10 };
  gg.start_flow([&times]() { return times-- == 0; });
}

