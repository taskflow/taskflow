#pragma once

namespace tf {

// ----------------------------------------------------------------------------
// taskflow
// ----------------------------------------------------------------------------
class AsyncTopology;
class Node;
class Graph;
class FlowBuilder;
class Semaphore;
class Subflow;
class Runtime;
class Task;
class TaskView;
class Taskflow;
class Topology;
class TopologyBase;
class Executor;
class Worker;
class WorkerView;
class ObserverInterface;
class ChromeTracingObserver;
class TFProfObserver;
class TFProfManager;

template <typename T>
class Future;

template <typename...Fs>
class Pipeline;

// ----------------------------------------------------------------------------
// cudaFlow
// ----------------------------------------------------------------------------
class cudaNode;
class cudaGraph;
class cudaTask;
class cudaFlow;
class cudaFlowCapturer;
class cudaFlowCapturerBase;
class cudaCapturingBase;
class cudaLinearCapturing;
class cudaSequentialCapturing;
class cudaRoundRobinCapturing;

// ----------------------------------------------------------------------------
// syclFlow
// ----------------------------------------------------------------------------
class syclNode;
class syclGraph;
class syclTask;
class syclFlow;


}  // end of namespace tf -----------------------------------------------------




