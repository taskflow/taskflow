module;

#include <taskflow/core/async_task.hpp>
#include <taskflow/core/atomic_notifier.hpp>
#include <taskflow/core/environment.hpp>
#include <taskflow/core/error.hpp>
#include <taskflow/core/executor.hpp>
#include <taskflow/core/flow_builder.hpp>
#include <taskflow/core/graph.hpp>
#include <taskflow/core/nonblocking_notifier.hpp>
#include <taskflow/core/observer.hpp>
#include <taskflow/core/runtime.hpp>
#include <taskflow/core/semaphore.hpp>
#include <taskflow/core/task_group.hpp>
#include <taskflow/core/task.hpp>
#include <taskflow/core/taskflow.hpp>
#include <taskflow/core/topology.hpp>
#include <taskflow/core/worker.hpp>
#include <taskflow/core/wsq.hpp>

export module tf:core;

export namespace tf {
    using tf::AsyncTask;
    using tf::AtomicNotifier;
    using tf::NSTATE;
    using tf::nstate_t;
    using tf::ESTATE;
    using tf::estate_t;
    using tf::ASTATE;
    using tf::astate_t;
    using tf::Executor;
    using tf::FlowBuilder;
    using tf::Graph;
    using tf::TaskParams;
    using tf::DefaultTaskParams;
    using tf::NodeBase;
    using tf::Node;
    using tf::ExplicitAnchorGuard;
    using tf::has_graph;
    using tf::NonblockingNotifier;
    using tf::observer_stamp_t;
    using tf::Segment;
    using tf::Timeline;
    using tf::ProfileData;
    using tf::ObserverInterface;
    using tf::ChromeObserver;
    using tf::TFProfObserver;
    using tf::TFProfManager;
    using tf::ObserverType;
    using tf::Runtime;
    using tf::NonpreemptiveRuntime;
    using tf::Semaphore;
    using tf::TaskGroup;
    using tf::TaskType;
    using tf::is_static_task;
    using tf::is_subflow_task;
    using tf::is_runtime_task;
    using tf::Task;
    using tf::TaskView;
    using tf::Taskflow;
    using tf::Future;
    using tf::Topology;
    using tf::DerivedTopology;
    using tf::DefaultNotifier;
    using tf::Worker;
    using tf::WorkerView;
    using tf::WorkerInterface;
    using tf::UnboundedWSQ;
    using tf::BoundedWSQ;
    
    using tf::is_task_params_v;
    using tf::has_graph_v;
    using tf::TASK_TYPES;
    using tf::is_static_task_v;
    using tf::is_subflow_task_v;
    using tf::is_runtime_task_v;
    using tf::is_condition_task_v;
    using tf::is_multi_condition_task_v;

    using tf::throw_re;
    using tf::animate;
    using tf::recycle;
    using tf::to_string;
    using tf::make_worker_interface;

    using tf::operator<<;
}

export namespace std {
    using std::hash;
}
