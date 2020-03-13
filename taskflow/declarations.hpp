#pragma once

namespace tf {

// ----------------------------------------------------------------------------
// forward declarations
// ----------------------------------------------------------------------------

// taskflow
class Node;
class Graph;
class FlowBuilder;
class Subflow;
class Task;
class TaskView;
class Taskflow;
class Topology;

template <typename Strategy>
class BasicExecutor;

// cudaflow
class cudaNode;
class cudaGraph;
class cudaTask;
class cudaFlow;

}  // end of namespace tf -----------------------------------------------------

#define TF_FRIEND_EXECUTOR \
template <typename Strategy> friend class BasicExecutor;

#define TF_FRIEND_TASKFLOW \
friend class Taskflow;


