#pragma once

#include "../taskflow.hpp"

/** 
@file pipeline.hpp
@brief pipeline include file
*/

namespace tf {

/**
@enum PipeType

@brief enumeration of all pipe types
*/
enum class PipeType : int {
  /** @brief parallel type */
  PARALLEL = 1,  
  /** @brief serial type */
  SERIAL   = 2
};

/**
@class Pipe

@brief class to create a pipe object for a pipeline stage

@tparam C callable type

A pipe represents a stage of a pipeline. A pipe can be either
@em parallel direction or @em serial direction (specified by tf::PipeType)
and is coupled with a callable to invoke by the pipeline scheduler.
The callable is one of the following possible forms:

@code{.cpp}
Pipe{PipeType::SERIAL, [](tf::Pipeflow&){}}
Pipe{PipeType::SERIAL, [](tf::Pipeflow&, tf::Runtime&){}}
@endcode

The first version takes a pipeflow object for user to to query the 
present statistics of a pipeline scheduling token.
The second version takes an additional runtime object for users to
interact with the taskflow scheduler, such as scheduling a task
and spawning a subflow.
*/
template <typename C>
class Pipe {

  template <typename... Ps>
  friend class Pipeline;

  public:
  
  /**
  @brief alias of the callable type
  */
  using callable_t = C;
  
  /**
  @brief constructs the pipe object

  @param d pipe type (tf::PipeType)
  @param callable callable type

  The constructor constructs a pipe with the given direction
  (either tf::PipeType::SERIAL or tf::PipeType::PARALLEL) and the
  given callable. The callable is one of the following possible forms:

  @code{.cpp}
  Pipe{PipeType::SERIAL, [](tf::Pipeflow&){}}
  Pipe{PipeType::SERIAL, [](tf::Pipeflow&, tf::Runtime&){}}
  @endcode

  When creating a pipeline, the direction of the first pipe must be serial 
  (tf::PipeType::SERIAL).
  */
  Pipe(PipeType d, C&& callable) :
    _type{d}, _callable{std::forward<C>(callable)} {
  }

  private:

  PipeType _type;

  C _callable;
};

/**
@class Pipeflow

@brief class to create a pipeflow object used by the pipe callable

Pipeflow represents a <i>scheduling token</i> in the pipeline scheduling 
framework. A pipeflow is created by the pipeline scheduler at runtime to
pass to the pipe callable. Users can query the present statistics
of that scheduling token, including the line identifier, pipe identifier, 
and token identifier, and build their application algorithms based on
these statistics. 
At the first stage, users can explicitly call the stop method 
to stop the pipeline scheduler.

@code{.cpp}
tf::Pipe{tf::PipeType::SERIAL, [](tf::Pipeflow& pf){
  std::cout << "token id=" << pf.token() 
            << " at line=" << pf.line()
            << " at pipe=" << pf.pipe()
            << '\n';
}};
@endcode

Pipeflow can only be created privately by the tf::Pipeline and
be used through the pipe callable.
*/
class Pipeflow {

  template <typename... Ps>
  friend class Pipeline;

  public:
  
  /**
  @brief queries the line identifier of the present token
  */
  size_t line() const {
    return _line;
  }
  
  /**
  @brief queries the pipe identifier of the present token
  */
  size_t pipe() const {
    return _pipe;
  }
  
  /**
  @brief queries the token identifier
  */
  size_t token() const {
    return _token;
  }

  /**
  @brief stops the pipeline scheduling 

  Only the first pipe can call this method to stop the pipeline.
  Others have no effect.
  */
  void stop() {
    _stop = true;
  }
  
  private:

  Pipeflow(size_t line, size_t pipe) :
    _line {line},
    _pipe {pipe} {
  }

  size_t _line;
  size_t _pipe;
  size_t _token;
  bool   _stop;
};

/**
@class Pipeline

@brief class to create a pipeline scheduling framework

@tparam Ps pipe types

A pipeline object is a composable graph object for users to create a 
<i>pipeline scheduling framework</i> using a module task in a taskflow.
Unlike the conventional pipeline programming frameworks (e.g., Intel TBB),
%Taskflow's pipeline algorithm does not provide any data abstraction,
which often restricts users from optimizing data layouts in their applications,
but a flexible framework for users to customize their application data 
atop our pipeline scheduling.
The following code creates a pipeline of up to four concurrent scheduling
tokens flowing through three serial pipes:

@code{.cpp}
tf::Taskflow taskflow;
tf::Executor executor;

const size_t num_lines = 4;
const size_t num_pipes = 3;

// create a custom data buffer
std::array<std::array<int, num_pipes>, num_lines> buffer;

// create a pipeline graph of four concurrent lines and three serial pipes
tf::Pipeline pipeline(num_lines,
  // first pipe must define a serial direction
  tf::Pipe{tf::PipeType::SERIAL, [&buffer](tf::Pipeflow& pf) {
    // generate only 5 scheduling tokens
    if(pf.token() == 5) {
      pf.stop();
    }
    // save the token id into the buffer
    else {
      buffer[pf.line()][pf.pipe()] = pf.token();
    }
  }},
  tf::Pipe{tf::PipeType::SERIAL, [&buffer] (tf::Pipeflow& pf) {
    // propagate the previous result to this pipe by adding one
    buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe()-1] + 1;
  }},
  tf::Pipe{tf::PipeType::SERIAL, [&buffer](tf::Pipeflow& pf){
    // propagate the previous result to this pipe by adding one
    buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe()-1] + 1;
  }}
);

// build the pipeline graph using composition
tf::Task init = taskflow.emplace([](){ std::cout << "ready\n"; })
                        .name("starting pipeline");
tf::Task task = taskflow.composed_of(pipeline)
                        .name("pipeline");
tf::Task stop = taskflow.emplace([](){ std::cout << "stopped\n"; })
                        .name("pipeline stopped");

// create task dependency
init.precede(task);
task.precede(stop);

// run the pipeline
executor.run(taskflow).wait();
@endcode

The above example creates a pipeline scheduling graph that rolls over 
four scheduling tokens in a cyclic fashion, as depicted below:

@code{.txt}
o -> o -> o
|    |    |
v    v    v
o -> o -> o
|    |    |
v    v    v
o -> o -> o
|    |    |
v    v    v
o -> o -> o 
@endcode

At each pipe stage, the program propagates the result to the next pipe
by adding one to the result stored in a custom data storage, @c buffer.
The pipeline scheduler will generate five scheduling tokens and then stop.
*/
template <typename... Ps>
class Pipeline {
  
  static_assert(sizeof...(Ps)>0, "must have at least one pipe");
  
  /**
  @private
  */
  struct Line {
    std::atomic<size_t> join_counter;
  };

  /**
  @private
  */
  struct PipeMeta {
    PipeType type;
  };

  public:

  /**
  @brief constructs a pipeline object

  @param num_lines the number of parallel lines
  @param ps a list of pipes

  Constructs a linear pipeline of up to @c num_lines concurrent
  scheduling tokens flowing through the given linear chain of pipes.
  The first pipe must define a serial direction (tf::PipeType::SERIAL) 
  or an exception will be thrown.
  */
  Pipeline(size_t num_lines, Ps&&... ps);

  /**
  @brief queries the number of parallel lines

  The function returns the number of parallel lines given by the user
  upon the construction of the pipeline.
  The number of lines represents the maximum parallelism this pipeline
  can achieve.
  */
  size_t num_lines() const noexcept;

  /**
  @brief queries the number of pipes

  The Function returns the number of pipes given by the user
  upon the construction of the pipeline.
  */
  constexpr size_t num_pipes() const noexcept;
  
  /**
  @brief resets the pipeline

  Resetting the pipeline to the initial state. After resetting a pipeline,
  its token identifier will start from zero as if the pipeline was just 
  constructed.
  */
  void reset();
  
  /**
  @brief queries the number of generated tokens in the pipeline

  The number represents the total scheduling tokens that has been
  generated by the pipeline so far.
  */
  size_t num_tokens() const noexcept;

  /**
  @brief obtains the graph object associated with the pipeline construct

  This method is primarily used as an opaque data structure for creating 
  a module task of the this pipeline. 
  */
  Graph& graph();

  private:
  
  Graph _graph;
  std::tuple<Ps...> _pipes;
  std::array<PipeMeta, sizeof...(Ps)> _meta;
  std::vector<std::array<Line, sizeof...(Ps)>> _lines; 
  std::vector<tf::Task> _tasks;
  size_t _num_tokens;
    
  void _on_pipe(Pipeflow&, Runtime&);
  void _build();
};

// constructor
template <typename... Ps>
Pipeline<Ps...>::Pipeline(size_t num_lines, Ps&&... ps) :
  _pipes {std::make_tuple(std::forward<Ps>(ps)...)},
  _meta  {PipeMeta{ps._type}...},
  _lines (num_lines),
  _tasks (num_lines + 1) {

  if(std::get<0>(_pipes)._type == PipeType::PARALLEL) {
    TF_THROW("first pipe must be serial");
  }

  reset();
  _build();
}

// Function: num_lines
template <typename... Ps>
size_t Pipeline<Ps...>::num_lines() const noexcept {
  return _lines.size();
}

// Function: num_pipes
template <typename... Ps>
constexpr size_t Pipeline<Ps...>::num_pipes() const noexcept {
  return sizeof...(Ps);
}

// Function: num_tokens
template <typename... Ps>
size_t Pipeline<Ps...>::num_tokens() const noexcept {
  return _num_tokens;
}

// Function: graph
template <typename... Ps>
Graph& Pipeline<Ps...>::graph() {
  return _graph;
}

// Function: reset
template <typename... Ps>
void Pipeline<Ps...>::reset() {

  _num_tokens = 0;

  _lines[0][0].join_counter.store(0, std::memory_order_relaxed);

  for(size_t l=1; l<num_lines(); l++) {
    for(size_t f=1; f<num_pipes(); f++) {
      _lines[l][f].join_counter.store(
        static_cast<size_t>(_meta[f].type), std::memory_order_relaxed
      );
    }
  }

  for(size_t f=1; f<num_pipes(); f++) {
    _lines[0][f].join_counter.store(1, std::memory_order_relaxed);
  }

  for(size_t l=1; l<num_lines(); l++) {
    _lines[l][0].join_counter.store(
      static_cast<size_t>(_meta[0].type) - 1, std::memory_order_relaxed
    );
  }
}
  
// Procedure: _on_pipe
template <typename... Ps>
void Pipeline<Ps...>::_on_pipe(Pipeflow& pf, Runtime& rt) {
  visit_tuple([&](auto&& pipe){ 
    using P = typename std::decay_t<decltype(pipe)>::callable_t;
    if constexpr(std::is_invocable_r_v<void, P, Pipeflow&>) {
      pipe._callable(pf);
    }
    else if constexpr(std::is_invocable_r_v<void, P, Pipeflow&, Runtime&>) {
      pipe._callable(pf, rt);
    }
    else {
      static_assert(dependent_false_v<P>, "unsupported pipe callable");
    }
  }, _pipes, pf._pipe);
}

// Procedure: _build
template <typename... Ps>
void Pipeline<Ps...>::_build() {
  
  using namespace std::literals::string_literals;
 
  FlowBuilder fb(_graph); 

  // init task
  _tasks[0] = fb.emplace([this]() {
    return static_cast<int>(_num_tokens % num_lines());
  }).name("cond");

  // line task
  for(size_t l = 0; l < num_lines(); l++) {

    _tasks[l + 1] = fb.emplace(
    [this, pf = Pipeflow{l, 0}] (tf::Runtime& rt) mutable {

    pipeline:

      _lines[pf._line][pf._pipe].join_counter.store(
        static_cast<size_t>(_meta[pf._pipe].type), std::memory_order_relaxed
      );

      if (pf._pipe == 0) {
        pf._token = _num_tokens;
        if (pf._stop = false, _on_pipe(pf, rt); pf._stop == true) {
          // here, the pipeline is not stopped yet because other
          // lines of tasks may still be running their last stages
          return;
        }
        ++_num_tokens;
      }
      else {
        _on_pipe(pf, rt);
      }

      size_t c_f = pf._pipe;
      size_t n_f = (pf._pipe + 1) % num_pipes();
      size_t n_l = (pf._line + 1) % num_lines();

      pf._pipe = n_f;
      
      // ---- scheduling starts here ----
      // Notice that the shared variable f must not be changed after this
      // point because it can result in data race due to the following 
      // condition:
      //
      // a -> b
      // |    |
      // v    v
      // c -> d
      //
      // d will be spawned by either c or b, so if c changes f but b spawns d
      // then data race on f will happen

      SmallVector<int, 2> retval;

      // downward dependency
      if(_meta[c_f].type == PipeType::SERIAL && 
         _lines[n_l][c_f].join_counter.fetch_sub(
           1, std::memory_order_acq_rel) == 1
        ) {
        retval.push_back(1);
      }
      
      // forward dependency
      if(_lines[pf._line][n_f].join_counter.fetch_sub(
          1, std::memory_order_acq_rel) == 1
        ) {
        retval.push_back(0);
      }
      
      // notice that the index of task starts from 1
      switch(retval.size()) {
        case 2: {
          rt.schedule(_tasks[n_l+1]);
          goto pipeline;
        }
        case 1: {
          if (retval[0] == 0) {
            goto pipeline;
          }
          else {
            rt.schedule(_tasks[n_l+1]);
          }
        }
      }
    }).name("rt-"s + std::to_string(l));

    _tasks[0].precede(_tasks[l+1]);
  }
}

}  // end of namespace tf -----------------------------------------------------





