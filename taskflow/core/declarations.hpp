#pragma once

namespace tf {

// taskflow
class AsyncTopology;
class Node;
class Graph;
class FlowBuilder;
class Semaphore;
class Subflow;
class Task;
class TaskView;
class Taskflow;
class Topology;
class TopologyBase;
class Executor;
class WorkerView;
class ObserverInterface;
class ChromeTracingObserver;
class TFProfObserver;
class TFProfManager;

template <typename T>
class Future;

// cudaFlow
class cudaNode;
class cudaGraph;
class cudaTask;
class cudaFlow;
class cudaFlowCapturer;
class cudaFlowCapturerBase;
class cudaCapturingBase;
class cudaSequentialCapturing;
class cudaRoundRobinCapturing;
class cublasFlowCapturer;

// syclFlow
class syclNode;
class syclGraph;
class syclTask;
class syclFlow;


}  // end of namespace tf -----------------------------------------------------




