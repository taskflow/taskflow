#ifndef DATAFLOW_TPP_HPP
#define DATAFLOW_TPP_HPP 1

#include <taskflow/taskflow.hpp>

#include <utility>
#include <vector>
#include <functional>

namespace df {


// "pads" are the object where functions on nodes can read from or write to;
// pads are all of the same compile time type T.
//
// Class hierarchies and virtual functions can be used to have different types
// of pads. In those cases, T would be probably a std::unique_ptr
template <typename T>
using Pad_id = typename std::vector<T>::size_type;


// "nodes" corresponds to taskflow tasks, but are agumented with input and
// output pads.
template <typename T>
class Node;
template <typename T>
using Node_id = typename std::vector< Node<T> >::size_type;


// each node executes a "functor" of type Tranform_f passing itself as a
// argument; the functor is expected to read from input pads and write to
// output pads.
template <typename T>
using Transform_f = std::function<void(Node<T>& element)>;


// "Dataflow_generator" is the type that keeps track of the relations between
// nodes and their pads. See below for the interface.
template <typename T>
class Dataflow_generator;


// Node interface, gives access to input and output pads.
// For convenience output pads can be accessed with the [] operator
template <typename T>
class Node {
public:
    // get input pad
    T const& ipad(Pad_id<T> id) const;

    // get output pad
    T& opad(Pad_id<T> id);
    T& operator[](Pad_id<T>);

    // get all the input pad ids
    std::vector< Pad_id<T> > const& ipad_list() const;

    // get all the output pad ids
    std::vector< Pad_id<T> > const& opad_list() const;


private:
    std::vector<T>& pads_;
    std::vector< Pad_id<T> > ipads_;
    std::vector< Pad_id<T> > opads_;
    Transform_f<T> compute_;

    explicit Node(std::vector<T>& pads)
      : pads_{pads},
        ipads_{},
        opads_{},
        compute_{ nullptr } {}

    friend class Dataflow_generator<T>;
};


// Dataflow generator interface, allows to:
// . create nodes
// . set up what each node executes
// . create arcs
// . execute the dag
//
// nodes are basically taskflow tasks agumented with input and output pads
// edges are taskflow precedences
// execution is done via taskflow taskflow.run_until
template <typename T>
class Dataflow_generator {
public:
    // creates a new node
    Node_id<T> create_node();

    // creates a new node and set up its functor
    Node_id<T> create_node(Transform_f<T> f);

    // creates a new arc ensuring that source output pads become the target
    // input pads
    Pad_id<T> create_arc(Node_id<T> source, Node_id<T> target);

    // Set up the functor for an existing node
    void set_function(Node_id<T> id, Transform_f<T> f);

    // Peek to a node, it might be useful to see how many input or output pads
    // it has
    Node<T> const& node(Node_id<T> id) const;

    // Executes the flow repeatly until the functor Cond returns true
    template <typename Cond>
    void start_flow(Cond && cond);

    // Executes the flow once
    void start_flow_once();

    Dataflow_generator() = default;
private:
    std::vector<T> pads_;
    std::vector< Node<T> > nodes_;
    std::vector<tf::Task> tasks_;
    tf::Framework executor_;
};



// Implementations

template <typename T>
T const&  Node<T>::
ipad(Pad_id<T> id) const {
    return pads_[id];
}


template <typename T>
T&  Node<T>::
opad(Pad_id<T> id) {
    return pads_[id];
}


template <typename T>
std::vector<Pad_id<T>> const&  Node<T>::
ipad_list() const {
    return ipads_;
}


template <typename T>
std::vector<Pad_id<T>> const&  Node<T>::
opad_list() const {
    return opads_;
}


template <typename T>
T& Node<T>::
operator[](Pad_id<T> id) {
    return pads_[id];
}


template <typename T>
Node_id<T>  Dataflow_generator<T>::
create_node() {
    Node_id<T> node_id { nodes_.size() };

    nodes_.emplace_back( Node(pads_) );
    tasks_.emplace_back( executor_.emplace([this, node_id]() { this->nodes_[node_id].compute_( this->nodes_[node_id] ); }) );

    return node_id;
}


template <typename T>
Node_id<T>  Dataflow_generator<T>::
create_node(Transform_f<T> f) {
    Node_id<T> node_id { create_node() };
    set_function(node_id, f);
    return node_id;
}


template <typename T>
Pad_id<T> Dataflow_generator<T>::
create_arc(Node_id<T> source, Node_id<T> target) {
    Pad_id<T> pad_id { pads_.size() };
    pads_.emplace_back( T{} );

    nodes_[source].opads_.push_back(pad_id);
    nodes_[target].ipads_.push_back(pad_id);

    tasks_[source].precede(tasks_[target]);

    return pad_id;
}


template <typename T>
template <typename Cond>
void Dataflow_generator<T>::
start_flow(Cond && cond) {
    tf::Taskflow taskflow{};
    taskflow.run_until(executor_, cond);
}


template <typename T>
void Dataflow_generator<T>::
start_flow_once() {
    bool leave{true};
    start_flow( [&leave]() { leave = !leave; return leave;} );
}


template <typename T>
Node<T> const&  Dataflow_generator<T> ::
node(Node_id<T> id) const {
    return nodes_[id];
}


template <typename T>
void  Dataflow_generator<T> ::
set_function(Node_id<T> id, Transform_f<T> f) {
    nodes_[id].compute_ = f;
}

}
#endif
