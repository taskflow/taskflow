#pragma once

#include "../taskflow.hpp"

/**
@file pipeline.hpp
@brief pipeline include file
*/

namespace tf {


// ----------------------------------------------------------------------------
// Structure Definition: DeferredPipeflow
// ----------------------------------------------------------------------------
// For example: 
// 12.defer(7); 12.defer(16);
//        _____
//       |     |
//       v     |
// 7    12    16
// |     ^
// |____ |
//
// DeferredPipeflow dpf of 12 :
// dpf._token = 12;
// dpf._num_deferrals = 1;
// dpf._dependents = std::list<size_t>{7,16};
// dpf._dependent_satellites has following two entries
// {key: 7, value: dpf._dependents.begin()} 
// {key: 16, value: dpf._dependents.begin()+1}
//
/** @private */
class DeferredPipeflow {

  template <typename... Ps>
  friend class Pipeline;
  
  template <typename P>
  friend class ScalablePipeline;
  
  public:
  
    DeferredPipeflow() = default;
    DeferredPipeflow(const DeferredPipeflow&) = delete;
    DeferredPipeflow(DeferredPipeflow&&) = delete;
  
    DeferredPipeflow(size_t t, size_t n, std::unordered_set<size_t>&& dep) : 
      _token{t}, _num_deferrals{n}, _dependents{std::move(dep)} {
    }
  
    DeferredPipeflow& operator = (const DeferredPipeflow&) = delete;
    DeferredPipeflow& operator = (DeferredPipeflow&&) = delete;
  
  private:
  
    // token id
    size_t _token;
  
    // number of deferrals
    size_t _num_deferrals;  
  
    // dependents
    // For example,
    // 12.defer(7); 12.defer(16)
    // _dependents = {7, 16}
    std::unordered_set<size_t> _dependents;
};



// ----------------------------------------------------------------------------
// Class Definition: Pipeflow
// ----------------------------------------------------------------------------

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

  template <typename P>
  friend class ScalablePipeline;

  template <typename... Ps>
  friend class DataPipeline;

  public:

  /**
  @brief default constructor
  */
  Pipeflow() = default;

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
  Calling stop from other pipes will throw exception.
  */
  void stop() {
    if(_pipe != 0) {
      TF_THROW("only the first pipe can stop the token");
    }
    _stop = true;
  }

  /**
  @brief queries the number of deferrals
  */
  size_t num_deferrals() const {
    return _num_deferrals;
  }

  /**
  @brief pushes token in _dependents

  Only the first pipe can call this method to defer the current
  scheduling token to the given token.
  */
  void defer(size_t token) {
    if(_pipe != 0) {
      TF_THROW("only the first pipe can defer the current scheduling token");
    }
    _dependents.insert(token);
  }
  
  private:

  // Regular data
  size_t _line;
  size_t _pipe;
  size_t _token;
  bool   _stop;
  
  // Data field for token dependencies
  size_t _num_deferrals; 
  std::unordered_set<size_t> _dependents; 

};

// ----------------------------------------------------------------------------
// Class Definition: PipeType
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Class Definition: Pipe
// ----------------------------------------------------------------------------

/**
@class Pipe

@brief class to create a pipe object for a pipeline stage

@tparam C callable type

A pipe represents a stage of a pipeline. A pipe can be either
@em parallel direction or @em serial direction (specified by tf::PipeType)
and is coupled with a callable to invoke by the pipeline scheduler.
The callable must take a referenced tf::Pipeflow object in the first argument:

@code{.cpp}
Pipe{PipeType::SERIAL, [](tf::Pipeflow&){}}
@endcode

The pipeflow object is used to query the statistics of a scheduling token
in the pipeline, such as pipe, line, and token numbers.
*/
template <typename C = std::function<void(tf::Pipeflow&)>>
class Pipe {

  template <typename... Ps>
  friend class Pipeline;

  template <typename P>
  friend class ScalablePipeline;

  public:

  /**
  @brief alias of the callable type
  */
  using callable_t = C;

  /**
  @brief default constructor
  */
  Pipe() = default;

  /**
  @brief constructs the pipe object

  @param d pipe type (tf::PipeType)
  @param callable callable type

  The constructor constructs a pipe with the given direction
  (tf::PipeType::SERIAL or tf::PipeType::PARALLEL) and the given callable. 
  The callable must take a referenced tf::Pipeflow object in the first argument.

  @code{.cpp}
  Pipe{PipeType::SERIAL, [](tf::Pipeflow&){}}
  @endcode

  When creating a pipeline, the direction of the first pipe must be serial
  (tf::PipeType::SERIAL).
  */
  Pipe(PipeType d, C&& callable) :
    _type{d}, _callable{std::forward<C>(callable)} {
  }

  /**
  @brief queries the type of the pipe

  Returns the type of the callable.
  */
  PipeType type() const {
    return _type;
  }

  /**
  @brief assigns a new type to the pipe

  @param type a tf::PipeType variable
  */
  void type(PipeType type) {
    _type = type;
  }

  /**
  @brief assigns a new callable to the pipe

  @tparam U callable type
  @param callable a callable object constructible from std::function<void(tf::Pipeflow&)>

  Assigns a new callable to the pipe with universal forwarding.
  */
  template <typename U>
  void callable(U&& callable) {
    _callable = std::forward<U>(callable);
  }

  private:

  PipeType _type;

  C _callable;
};

// ----------------------------------------------------------------------------
// Class Definition: Pipeline
// ----------------------------------------------------------------------------

/**
@class Pipeline

@brief class to create a pipeline scheduling framework

@tparam Ps pipe types

A pipeline is a composable graph object for users to create a
<i>pipeline scheduling framework</i> using a module task in a taskflow.
Unlike the conventional pipeline programming frameworks (e.g., Intel TBB),
%Taskflow's pipeline algorithm does not provide any data abstraction,
which often restricts users from optimizing data layouts in their applications,
but a flexible framework for users to customize their application data
atop our pipeline scheduling.
The following code creates a pipeline of four parallel lines to schedule
tokens through three serial pipes:

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

The above example creates a pipeline graph that schedules five tokens over
four parallel lines in a circular fashion, as depicted below:

@code{.shell-session}
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

Internally, tf::Pipeline uses std::tuple to store the given sequence of pipes.
The definition of each pipe can be different, completely decided by the compiler
to optimize the object layout.
After a pipeline is constructed, it is not possible to change its pipes.
If applications need to change these pipes, please use tf::ScalablePipeline.
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

  Constructs a pipeline of up to @c num_lines parallel lines to schedule
  tokens through the given linear chain of pipes.
  The first pipe must define a serial direction (tf::PipeType::SERIAL)
  or an exception will be thrown.
  */
  Pipeline(size_t num_lines, Ps&&... ps);

  /**
  @brief constructs a pipeline object

  @param num_lines the number of parallel lines
  @param ps a tuple of pipes

  Constructs a pipeline of up to @c num_lines parallel lines to schedule
  tokens through the given linear chain of pipes.
  The first pipe must define a serial direction (tf::PipeType::SERIAL)
  or an exception will be thrown.
  */
  Pipeline(size_t num_lines, std::tuple<Ps...>&& ps);

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

  size_t _num_tokens;

  std::tuple<Ps...> _pipes;
  std::array<PipeMeta, sizeof...(Ps)> _meta;
  std::vector<std::array<Line, sizeof...(Ps)>> _lines;
  std::vector<Task> _tasks;
  std::vector<Pipeflow> _pipeflows;
  
  // queue of ready tokens (paired with their deferral times)
  // For example,
  // when 12 does not have any dependents,
  // we put 12 in _ready_tokens queue
  // Assume num_deferrals of 12 is 1,
  // we push pair{12, 1} in the queue 
  std::queue<std::pair<size_t, size_t>> _ready_tokens;

  // unordered_map of token dependencies
  // For example,
  // 12.defer(16); 13.defer(16);
  // _token_dependencies has the following entry
  // {key: 16, value: std::vector{12, 13}}.
  std::unordered_map<size_t, std::vector<size_t>> _token_dependencies;
  
  // unordered_map of deferred tokens
  // For example,
  // 12.defer(16); 13.defer(16);
  // _deferred_tokens has the following two entries
  // {key: 12, DeferredPipeflow of 12} and
  // {key: 13, DeferredPipeflow of 13}
  std::unordered_map<size_t, DeferredPipeflow> _deferred_tokens;
  
  // variable to keep track of the longest deferred tokens
  // For example,
  // 2.defer(16)
  // 5.defer(19)
  // 5.defer(17),
  // _longest_deferral will be 19 - after token 19 the pipeline
  // has almost zero cost on handling deferred pipeflow
  size_t _longest_deferral = 0;  
  
  template <size_t... I>
  auto _gen_meta(std::tuple<Ps...>&&, std::index_sequence<I...>);

  void _on_pipe(Pipeflow&, Runtime&);
  void _build();
  void _check_dependents(Pipeflow&);
  void _construct_deferred_tokens(Pipeflow&);
  void _resolve_token_dependencies(Pipeflow&); 
};

// constructor
template <typename... Ps>
Pipeline<Ps...>::Pipeline(size_t num_lines, Ps&&... ps) :
  _pipes     {std::make_tuple(std::forward<Ps>(ps)...)},
  _meta      {PipeMeta{ps.type()}...},
  _lines     (num_lines),
  _tasks     (num_lines + 1),
  _pipeflows (num_lines) {

  if(num_lines == 0) {
    TF_THROW("must have at least one line");
  }

  if(std::get<0>(_pipes).type() != PipeType::SERIAL) {
    TF_THROW("first pipe must be serial");
  }

  reset();
  _build();
}

// constructor
template <typename... Ps>
Pipeline<Ps...>::Pipeline(size_t num_lines, std::tuple<Ps...>&& ps) :
  _pipes     {std::forward<std::tuple<Ps...>>(ps)},
  _meta      {_gen_meta(
    std::forward<std::tuple<Ps...>>(ps), std::make_index_sequence<sizeof...(Ps)>{}
  )},
  _lines     (num_lines),
  _tasks     (num_lines + 1),
  _pipeflows (num_lines) {

  if(num_lines == 0) {
    TF_THROW("must have at least one line");
  }

  if(std::get<0>(_pipes).type() != PipeType::SERIAL) {
    TF_THROW("first pipe must be serial");
  }

  reset();
  _build();
}

// Function: _get_meta
template <typename... Ps>
template <size_t... I>
auto Pipeline<Ps...>::_gen_meta(std::tuple<Ps...>&& ps, std::index_sequence<I...>) {
  return std::array{PipeMeta{std::get<I>(ps).type()}...};
}

// Function: num_lines
template <typename... Ps>
size_t Pipeline<Ps...>::num_lines() const noexcept {
  return _pipeflows.size();
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

  for(size_t l = 0; l<num_lines(); l++) {
    _pipeflows[l]._pipe = 0;
    _pipeflows[l]._line = l;
    
    _pipeflows[l]._num_deferrals = 0;
    _pipeflows[l]._dependents.clear();
  }
  
  assert(_ready_tokens.empty() == true);
  _token_dependencies.clear();
  _deferred_tokens.clear();

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
    using callable_t = typename std::decay_t<decltype(pipe)>::callable_t;
    if constexpr (std::is_invocable_v<callable_t, Pipeflow&>) {
      pipe._callable(pf);
    }
    else if constexpr(std::is_invocable_v<callable_t, Pipeflow&, Runtime&>) {
      pipe._callable(pf, rt);
    }
    else {
      static_assert(dependent_false_v<callable_t>, "un-supported pipe callable type");
    }
  }, _pipes, pf._pipe);
}

// Procedure: _check_dependents
// Check and remove invalid dependents after on_pipe
// For example, users may defer a pipeflow to multiple tokens,
// and we need to remove invalid tokens.
//   12.defer(7);   // valid only if 7 is deferred, or invalid otherwise
//   12.defer(16);  // 16 is valid 
template <typename... Ps>
void Pipeline<Ps...>::_check_dependents(Pipeflow& pf) {
  //if (pf._dependents.size()) {
  ++pf._num_deferrals;
  
  for (auto it = pf._dependents.begin(); it != pf._dependents.end();) {
 
    // valid (e.g., 12.defer(16)) 
    if (*it >= _num_tokens) {
      _token_dependencies[*it].push_back(pf._token);
      _longest_deferral = std::max(_longest_deferral, *it);
      ++it;
    }
    // valid or invalid (e.g., 12.defer(7))
    else {
      auto pit = _deferred_tokens.find(*it);
      
      // valid (e.g., 7 is deferred)
      if (pit != _deferred_tokens.end()) {
        _token_dependencies[*it].push_back(pf._token);
        ++it;
      }

      // invalid (e.g., 7 is finished - this this 12.defer(7) is dummy)
      else {
        it = pf._dependents.erase(it);
      }
    }
  }
}

// Procedure: _construct_deferred_tokens
// Construct a data structure for a deferred token
// 
// For example, 
// 12.defer(7); 12.defer(16);
// After _check_dependents, 12 needs to be deferred,
// so we will construct a data structure for 12 using hashmap:
// {key: 12, value: DeferredPipeflow of 12}
template <typename... Ps>
void Pipeline<Ps...>::_construct_deferred_tokens(Pipeflow& pf) {
  
  //auto res = _deferred_tokens.emplace(
  //  pf._token, DeferredPipeflow{pf._token, pf._num_deferrals, std::move(pf._dependents)}
  //);
  
  // construct the deferred pipeflow with zero copy
  //auto res = _deferred_tokens.emplace(
  _deferred_tokens.emplace(
    std::piecewise_construct,
    std::forward_as_tuple(pf._token),
    std::forward_as_tuple(
      pf._token, pf._num_deferrals, std::move(pf._dependents)
    )
  );

  //assert(res.second == true);
}

// Procedure: _resolve_token_dependencies
// Resolve dependencies for tokens that defer to current token
// 
// For example,
// 12.defer(16);
// 13.defer(16);
// _token_dependencies will have the entry
// {key: 16, value: std::vector{12, 13}} 
//
// When 16 finishes, we need to remove 16 from 12's and 13's 
// individual_dependents
template <typename... Ps>
void Pipeline<Ps...>::_resolve_token_dependencies(Pipeflow& pf) {

  if (auto it = _token_dependencies.find(pf._token);
      it != _token_dependencies.end()) {
    
    // iterate tokens that defer to pf._token
    // (e.g., 12 and 13)
    for(size_t target : it->second) {

      auto dpf = _deferred_tokens.find(target);

      assert(dpf != _deferred_tokens.end());

      // erase pf._token from target's _dependents
      // (e.g., remove 16 from 12's dependents)
      dpf->second._dependents.erase(pf._token);
      //  dpf->second._dependent_satellites[pf._token]
      //);

      // target has no dependents
      if (dpf->second._dependents.empty()) {

        // push target into _ready_tokens queue
        _ready_tokens.emplace(dpf->second._token, dpf->second._num_deferrals);
        //_ready_tokens.push(
        //  std::make_pair(dpf->second._token, dpf->second._num_deferrals)
        //);
        
        // erase target from _deferred_tokens
        _deferred_tokens.erase(dpf);
      }
    }

    // remove pf._token from _token_dependencies
    // (e.g., remove the entry
    // {key: 16, value: std::vector{12, 13}} from _token_dependencies)
    _token_dependencies.erase(it);
  }
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

    _tasks[l + 1] = fb.emplace([this, l] (tf::Runtime& rt) mutable {

      auto pf = &_pipeflows[l];

      pipeline:

      _lines[pf->_line][pf->_pipe].join_counter.store(
        static_cast<size_t>(_meta[pf->_pipe].type), std::memory_order_relaxed
      );
      
      // First pipe does all jobs of initialization and token dependencies
      if (pf->_pipe == 0) {
        // _ready_tokens queue is not empty
        // substitute pf with the token at the front of the queue
        if (!_ready_tokens.empty()) {
          pf->_token = _ready_tokens.front().first;
          pf->_num_deferrals = _ready_tokens.front().second;
          _ready_tokens.pop();
        }
        else {
          pf->_token = _num_tokens;
          pf->_num_deferrals = 0;
        }
      
      handle_token_dependency: 

        if (pf->_stop = false, _on_pipe(*pf, rt); pf->_stop == true) {
          // here, the pipeline is not stopped yet because other
          // lines of tasks may still be running their last stages
          return;
        }
        
        if (_num_tokens == pf->_token) {
          ++_num_tokens;
        }
      
        if (pf->_dependents.empty() == false){ 
          // check if the pf->_dependents have valid dependents
          _check_dependents(*pf); 
          
          // tokens in pf->_dependents are all valid dependents 
          if (pf->_dependents.size()) {
            
            // construct a data structure for pf in _deferred_tokens 
            _construct_deferred_tokens(*pf);
            goto pipeline;
          }

          // tokens in pf->_dependents are invalid dependents
          // directly goto on_pipe on the same line
          else {
            goto handle_token_dependency;
          }
        }
        
        // Every token within the deferral range needs to check
        // if it can resolve dependencies on other tokens.
        if (pf->_token <= _longest_deferral) {
          _resolve_token_dependencies(*pf); 
        }
      }
      else {
        _on_pipe(*pf, rt);
      }

      size_t c_f = pf->_pipe;
      size_t n_f = (pf->_pipe + 1) % num_pipes();
      size_t n_l = (pf->_line + 1) % num_lines();

      pf->_pipe = n_f;

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

      std::array<int, 2> retval;
      size_t n = 0;

      // downward dependency
      if(_meta[c_f].type == PipeType::SERIAL &&
         _lines[n_l][c_f].join_counter.fetch_sub(
           1, std::memory_order_acq_rel) == 1
        ) {
        retval[n++] = 1;
      }

      // forward dependency
      if(_lines[pf->_line][n_f].join_counter.fetch_sub(
          1, std::memory_order_acq_rel) == 1
        ) {
        retval[n++] = 0;
      }
      
      // notice that the task index starts from 1
      switch(n) {
        case 2: {
          rt.schedule(_tasks[n_l+1]);
          goto pipeline;
        }
        case 1: {
          // downward dependency 
          if (retval[0] == 1) {
            pf = &_pipeflows[n_l];
          }
          // forward dependency
          goto pipeline;
        }
      }
    }).name("rt-"s + std::to_string(l));

    _tasks[0].precede(_tasks[l+1]);
  }
}

// ----------------------------------------------------------------------------
// Class Definition: ScalablePipeline
// ----------------------------------------------------------------------------

/**
@class ScalablePipeline

@brief class to create a scalable pipeline object

@tparam P type of the iterator to a range of pipes

A scalable pipeline is a composable graph object for users to create a
<i>pipeline scheduling framework</i> using a module task in a taskflow.
Unlike tf::Pipeline that instantiates all pipes upon the construction time,
tf::ScalablePipeline allows variable assignments of pipes using range iterators.
Users can also reset a scalable pipeline to a different range of pipes
between runs. The following code creates a scalable pipeline of four
parallel lines to schedule tokens through three serial pipes in a custom storage,
then resetting the pipeline to a new range of five serial pipes:

@code{.cpp}
tf::Taskflow taskflow("pipeline");
tf::Executor executor;

const size_t num_lines = 4;

// create data storage
std::array<int, num_lines> buffer;

// define the pipe callable
auto pipe_callable = [&buffer] (tf::Pipeflow& pf) mutable {
  switch(pf.pipe()) {
    // first stage generates only 5 scheduling tokens and saves the
    // token number into the buffer.
    case 0: {
      if(pf.token() == 5) {
        pf.stop();
      }
      else {
        printf("stage 1: input token = %zu\n", pf.token());
        buffer[pf.line()] = pf.token();
      }
      return;
    }
    break;

    // other stages propagate the previous result to this pipe and
    // increment it by one
    default: {
      printf(
        "stage %zu: input buffer[%zu] = %d\n", pf.pipe(), pf.line(), buffer[pf.line()]
      );
      buffer[pf.line()] = buffer[pf.line()] + 1;
    }
    break;
  }
};

// create a vector of three pipes
std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;

for(size_t i=0; i<3; i++) {
  pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
}

// create a pipeline of four parallel lines based on the given vector of pipes
tf::ScalablePipeline pl(num_lines, pipes.begin(), pipes.end());

// build the pipeline graph using composition
tf::Task init = taskflow.emplace([](){ std::cout << "ready\n"; })
                        .name("starting pipeline");
tf::Task task = taskflow.composed_of(pl)
                        .name("pipeline");
tf::Task stop = taskflow.emplace([](){ std::cout << "stopped\n"; })
                        .name("pipeline stopped");

// create task dependency
init.precede(task);
task.precede(stop);

// dump the pipeline graph structure (with composition)
taskflow.dump(std::cout);

// run the pipeline
executor.run(taskflow).wait();

// reset the pipeline to a new range of five pipes and starts from
// the initial state (i.e., token counts from zero)
for(size_t i=0; i<2; i++) {
  pipes.emplace_back(tf::PipeType::SERIAL, pipe_callable);
}
pl.reset(pipes.begin(), pipes.end());

executor.run(taskflow).wait();
@endcode

The above example creates a pipeline graph that schedules five tokens over
four parallel lines in a circular fashion, first going through three serial pipes
and then five serial pipes:

@code{.shell-session}
# initial construction of three serial pipes
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

# resetting to a new range of five serial pipes
o -> o -> o -> o -> o
|    |    |    |    |
v    v    v    v    v
o -> o -> o -> o -> o
|    |    |    |    |
v    v    v    v    v
o -> o -> o -> o -> o
|    |    |    |    |
v    v    v    v    v
o -> o -> o -> o -> o
@endcode

Each pipe has the same type of `%tf::Pipe<%std::function<void(%tf::Pipeflow&)>>`
and is kept in a vector that is amenable to change.
We construct the scalable pipeline using two range iterators pointing to the
beginning and the end of the vector.
At each pipe stage, the program propagates the result to the next pipe
by adding one to the result stored in a custom data storage, @c buffer.
The pipeline scheduler will generate five scheduling tokens and then stop.

A scalable pipeline is move-only.
*/
template <typename P>
class ScalablePipeline {

  /**
  @private
  */
  struct Line {
    std::atomic<size_t> join_counter;
  };


  public:

  /**
  @brief pipe type
  */
  using pipe_t = typename std::iterator_traits<P>::value_type;

  /**
  @brief default constructor
  */
  ScalablePipeline() = default;

  /**
  @brief constructs an empty scalable pipeline object

  @param num_lines the number of parallel lines

  An empty scalable pipeline does not have any pipes.
  The pipeline needs to be reset to a valid range of pipes
  before running.
  */
  ScalablePipeline(size_t num_lines);

  /**
  @brief constructs a scalable pipeline object

  @param num_lines the number of parallel lines
  @param first iterator to the beginning of the range
  @param last iterator to the end of the range

  Constructs a pipeline from the given range of pipes specified in
  <tt>[first, last)</tt> using @c num_lines parallel lines.
  The first pipe must define a serial direction (tf::PipeType::SERIAL)
  or an exception will be thrown.

  Internally, the scalable pipeline copies the iterators
  from the specified range. Those pipe callables pointed to by
  these iterators must remain valid during the execution of the pipeline.
  */
  ScalablePipeline(size_t num_lines, P first, P last);

  /**
  @brief disabled copy constructor
  */
  ScalablePipeline(const ScalablePipeline&) = delete;

  /**
  @brief move constructor

  Constructs a pipeline from the given @c rhs using move semantics
  (i.e. the data in @c rhs is moved into this pipeline).
  After the move, @c rhs is in a state as if it is just constructed.
  The behavior is undefined if @c rhs is running during the move.
  */
  ScalablePipeline(ScalablePipeline&& rhs);

  /**
  @brief disabled copy assignment operator
  */
  ScalablePipeline& operator = (const ScalablePipeline&) = delete;

  /**
  @brief move constructor

  Replaces the contents with those of @c rhs using move semantics
  (i.e. the data in @c rhs is moved into this pipeline).
  After the move, @c rhs is in a state as if it is just constructed.
  The behavior is undefined if @c rhs is running during the move.
  */
  ScalablePipeline& operator = (ScalablePipeline&& rhs);

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
  size_t num_pipes() const noexcept;

  /**
  @brief resets the pipeline

  Resets the pipeline to the initial state. After resetting a pipeline,
  its token identifier will start from zero.
  */
  void reset();

  /**
  @brief resets the pipeline with a new range of pipes

  @param first iterator to the beginning of the range
  @param last iterator to the end of the range

  The member function assigns the pipeline to a new range of pipes
  specified in <tt>[first, last)</tt> and resets the pipeline to the
  initial state. After resetting a pipeline, its token identifier will
  start from zero.

  Internally, the scalable pipeline copies the iterators
  from the specified range. Those pipe callables pointed to by
  these iterators must remain valid during the execution of the pipeline.
  */
  void reset(P first, P last);

  /**
  @brief resets the pipeline to a new line number and a
         new range of pipes

  @param num_lines number of parallel lines
  @param first iterator to the beginning of the range
  @param last iterator to the end of the range

  The member function resets the pipeline to a new number of
  parallel lines and a new range of pipes specified in
  <tt>[first, last)</tt>, as if the pipeline is just constructed.
  After resetting a pipeline, its token identifier will start from zero.

  Internally, the scalable pipeline copies the iterators
  from the specified range. Those pipe callables pointed to by
  these iterators must remain valid during the execution of the pipeline.
  */
  void reset(size_t num_lines, P first, P last);

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

  size_t _num_tokens{0};

  std::vector<P> _pipes;
  std::vector<Task> _tasks;
  std::vector<Pipeflow> _pipeflows;
  std::unique_ptr<Line[]> _lines;

  // chchiu
  std::queue<std::pair<size_t, size_t>> _ready_tokens;
  std::unordered_map<size_t, std::vector<size_t>> _token_dependencies;
  std::unordered_map<size_t, DeferredPipeflow> _deferred_tokens;
  size_t _longest_deferral = 0;
  
  void _check_dependents(Pipeflow&);
  void _construct_deferred_tokens(Pipeflow&);
  void _resolve_token_dependencies(Pipeflow&);
  // chchiu

  void _on_pipe(Pipeflow&, Runtime&);
  void _build();

  Line& _line(size_t, size_t);
};

// constructor
template <typename P>
ScalablePipeline<P>::ScalablePipeline(size_t num_lines) :
  _tasks     (num_lines + 1),
  _pipeflows (num_lines) {

  if(num_lines == 0) {
    TF_THROW("must have at least one line");
  }

  _build();
}

// constructor
template <typename P>
ScalablePipeline<P>::ScalablePipeline(size_t num_lines, P first, P last) :
  _tasks     (num_lines + 1),
  _pipeflows (num_lines) {

  if(num_lines == 0) {
    TF_THROW("must have at least one line");
  }

  reset(first, last);
  _build();
}

/*
// move constructor
template <typename P>
ScalablePipeline<P>::ScalablePipeline(ScalablePipeline&& rhs) :
  _graph              {std::move(rhs._graph)},
  _num_tokens         {rhs._num_tokens},
  _pipes              {std::move(rhs._pipes)},
  _tasks              {std::move(rhs._tasks)},
  _pipeflows          {std::move(rhs._pipeflows)},
  _lines              {std::move(rhs._lines)},
  _ready_tokens       {std::move(rhs._ready_tokens)},
  _token_dependencies {std::move(rhs._token_dependencies)},
  _deferred_tokens    {std::move(rhs._deferred_tokens)},
  _longest_deferral   {rhs._longest_deferral}{

  rhs._longest_deferral = 0;
  rhs._num_tokens       = 0;
  std::cout << "scalable move constructor\n";
}
*/

// move constructor
template <typename P>
ScalablePipeline<P>::ScalablePipeline(ScalablePipeline&& rhs):
  _num_tokens           {rhs._num_tokens},
  _pipes                {std::move(rhs._pipes)},
  _pipeflows            {std::move(rhs._pipeflows)},
  _lines                {std::move(rhs._lines)},
  _ready_tokens         {std::move(rhs._ready_tokens)},
  _token_dependencies   {std::move(rhs._token_dependencies)},
  _deferred_tokens      {std::move(rhs._deferred_tokens)},
  _longest_deferral     {rhs._longest_deferral}{


  //_num_tokens = rhs._num_tokens;

  //_pipes.resize(rhs.num_pipes());
  //size_t i=0;
  //for(auto itr = rhs._pipes.begin(); itr != rhs._pipes.end(); itr++) {
  //  _pipes[i++] = *itr;
  //}


  //_pipeflows.resize(rhs.num_lines());
  //for(size_t l = 0; l<rhs.num_lines(); l++) {
  //  _pipeflows[l]._pipe = rhs._pipeflows[l]._pipe;
  //  _pipeflows[l]._line = rhs._pipeflows[l]._line;
  //  _pipeflows[l]._num_deferrals = 0;
  //  _pipeflows[l]._dependents.clear();
  //}

  //_lines = std::make_unique<Line[]>(rhs.num_lines() * rhs._pipes.size());
  //for(size_t l=0; l<num_lines(); l++) {
  //  for(size_t f=0; f<num_pipes(); f++) {
  //    _line(l, f).join_counter.store(
  //      rhs._line(l, f).join_counter, std::memory_order_relaxed
  //    );
  //  }
  //}
 
  //_ready_tokens = std::move(rhs._ready_tokens);
  //_token_dependencies = std::move(rhs._token_dependencies);
  //_deferred_tokens = std::move(rhs._deferred_tokens);

  _graph.clear();
  _tasks.resize(_pipeflows.size()+1);
  rhs._longest_deferral = 0;
  rhs._num_tokens       = 0;
  rhs._tasks.clear();
  _build();
}

//// move assignment operator
//template <typename P>
//ScalablePipeline<P>& ScalablePipeline<P>::operator = (ScalablePipeline&& rhs) {
//  _graph                = std::move(rhs._graph);
//  _num_tokens           = rhs._num_tokens;
//  _pipes                = std::move(rhs._pipes);
//  _tasks                = std::move(rhs._tasks);
//  _pipeflows            = std::move(rhs._pipeflows);
//  _lines                = std::move(rhs._lines);
//  rhs._num_tokens       = 0;
//  _ready_tokens         = std::move(rhs._ready_tokens);
//  _token_dependencies   = std::move(rhs._token_dependencies);
//  _deferred_tokens      = std::move(rhs._deferred_tokens);
//  _longest_deferral     = rhs._longest_deferral;
//  rhs._longest_deferral = 0;
//  std::cout << "scalable move assignment\n";
//  return *this;
//}

// move assignment operator
template <typename P>
ScalablePipeline<P>& ScalablePipeline<P>::operator = (ScalablePipeline&& rhs) {
  _num_tokens         = rhs._num_tokens;
  _pipes              = std::move(rhs._pipes);
  _pipeflows          = std::move(rhs._pipeflows);
  _lines              = std::move(rhs._lines);
  _ready_tokens       = std::move(rhs._ready_tokens);
  _token_dependencies = std::move(rhs._token_dependencies);
  _deferred_tokens    = std::move(rhs._deferred_tokens);
  _longest_deferral   = rhs._longest_deferral;

  _graph.clear();
  _tasks.resize(_pipeflows.size()+1);

  rhs._longest_deferral = 0;
  rhs._num_tokens       = 0;
  rhs._tasks.clear();
  _build();
  return *this;
}

// Function: num_lines
template <typename P>
size_t ScalablePipeline<P>::num_lines() const noexcept {
  return _pipeflows.size();
}

// Function: num_pipes
template <typename P>
size_t ScalablePipeline<P>::num_pipes() const noexcept {
  return _pipes.size();
}

// Function: num_tokens
template <typename P>
size_t ScalablePipeline<P>::num_tokens() const noexcept {
  return _num_tokens;
}

// Function: graph
template <typename P>
Graph& ScalablePipeline<P>::graph() {
  return _graph;
}

// Function: _line
template <typename P>
typename ScalablePipeline<P>::Line& ScalablePipeline<P>::_line(size_t l, size_t p) {
  return _lines[l*num_pipes() + p];
}

template <typename P>
void ScalablePipeline<P>::reset(size_t num_lines, P first, P last) {

  if(num_lines == 0) {
    TF_THROW("must have at least one line");
  }

  _graph.clear();
  _tasks.resize(num_lines + 1);
  _pipeflows.resize(num_lines);

  reset(first, last);

  _build();
}

// Function: reset
template <typename P>
void ScalablePipeline<P>::reset(P first, P last) {

  size_t num_pipes = static_cast<size_t>(std::distance(first, last));

  if(num_pipes == 0) {
    TF_THROW("pipeline cannot be empty");
  }

  if(first->type() != PipeType::SERIAL) {
    TF_THROW("first pipe must be serial");
  }

  _pipes.resize(num_pipes);

  size_t i=0;
  for(auto itr = first; itr != last; itr++) {
    _pipes[i++] = itr;
  }

  _lines = std::make_unique<Line[]>(num_lines() * _pipes.size());

  reset();
}

// Function: reset
template <typename P>
void ScalablePipeline<P>::reset() {

  _num_tokens = 0;

  for(size_t l = 0; l<num_lines(); l++) {
    _pipeflows[l]._pipe = 0;
    _pipeflows[l]._line = l;
    _pipeflows[l]._num_deferrals = 0;
    _pipeflows[l]._dependents.clear();
  }

  _line(0, 0).join_counter.store(0, std::memory_order_relaxed);

  for(size_t l=1; l<num_lines(); l++) {
    for(size_t f=1; f<num_pipes(); f++) {
      _line(l, f).join_counter.store(
        static_cast<size_t>(_pipes[f]->type()), std::memory_order_relaxed
      );
    }
  }

  for(size_t f=1; f<num_pipes(); f++) {
    _line(0, f).join_counter.store(1, std::memory_order_relaxed);
  }

  for(size_t l=1; l<num_lines(); l++) {
    _line(l, 0).join_counter.store(
      static_cast<size_t>(_pipes[0]->type()) - 1, std::memory_order_relaxed
    );
  }
  
  assert(_ready_tokens.empty() == true);
  _token_dependencies.clear();
  _deferred_tokens.clear();
}

// Procedure: _on_pipe
template <typename P>
void ScalablePipeline<P>::_on_pipe(Pipeflow& pf, Runtime& rt) {
    
  using callable_t = typename pipe_t::callable_t;

  if constexpr (std::is_invocable_v<callable_t, Pipeflow&>) {
    _pipes[pf._pipe]->_callable(pf);
  }
  else if constexpr(std::is_invocable_v<callable_t, Pipeflow&, Runtime&>) {
    _pipes[pf._pipe]->_callable(pf, rt);
  }
  else {
    static_assert(dependent_false_v<callable_t>, "un-supported pipe callable type");
  }
}

template <typename P>
void ScalablePipeline<P>::_check_dependents(Pipeflow& pf) {
  ++pf._num_deferrals;
  
  for (auto it = pf._dependents.begin(); it != pf._dependents.end();) {
 
    // valid (e.g., 12.defer(16)) 
    if (*it >= _num_tokens) {
      _token_dependencies[*it].push_back(pf._token);
      _longest_deferral = std::max(_longest_deferral, *it);
      ++it;
    }
    // valid or invalid (e.g., 12.defer(7))
    else {
      auto pit = _deferred_tokens.find(*it);
      
      // valid (e.g., 7 is deferred)
      if (pit != _deferred_tokens.end()) {
        _token_dependencies[*it].push_back(pf._token);
        ++it;
      }

      else {
        it = pf._dependents.erase(it);
      }
    }
  }
}

// Procedure: _construct_deferred_tokens
// Construct a data structure for a deferred token
template <typename P>
void ScalablePipeline<P>::_construct_deferred_tokens(Pipeflow& pf) {
  
  // construct the deferred pipeflow with zero copy
  _deferred_tokens.emplace(
    std::piecewise_construct,
    std::forward_as_tuple(pf._token),
    std::forward_as_tuple(
      pf._token, pf._num_deferrals, std::move(pf._dependents)
    )
  );
}

// Procedure: _resolve_token_dependencies
// Resolve dependencies for tokens that defer to current token
template <typename P>
void ScalablePipeline<P>::_resolve_token_dependencies(Pipeflow& pf) {

  if (auto it = _token_dependencies.find(pf._token);
      it != _token_dependencies.end()) {
    
    // iterate tokens that defer to pf._token
    for(size_t target : it->second) {

      auto dpf = _deferred_tokens.find(target);

      assert(dpf != _deferred_tokens.end());

      // erase pf._token from target's _dependents
      dpf->second._dependents.erase(pf._token);
      
      // target has no dependents
      if (dpf->second._dependents.empty()) {
        _ready_tokens.emplace(dpf->second._token, dpf->second._num_deferrals);
        _deferred_tokens.erase(dpf);
      }
    }

    _token_dependencies.erase(it);
  }
}

// Procedure: _build
template <typename P>
void ScalablePipeline<P>::_build() {

  using namespace std::literals::string_literals;

  FlowBuilder fb(_graph);
  
  // init task
  _tasks[0] = fb.emplace([this]() {
    return static_cast<int>(_num_tokens % num_lines());
  }).name("cond");

  // line task
  for(size_t l = 0; l < num_lines(); l++) {
    
    _tasks[l + 1] = fb.emplace([this, l] (tf::Runtime& rt) mutable {

      auto pf = &_pipeflows[l];

      pipeline:

      _line(pf->_line, pf->_pipe).join_counter.store(
        static_cast<size_t>(_pipes[pf->_pipe]->type()), std::memory_order_relaxed
      );

      // First pipe does all jobs of initialization and token dependencies
      if (pf->_pipe == 0) {
        // _ready_tokens queue is not empty
        // substitute pf with the token at the front of the queue
        if (!_ready_tokens.empty()) {
          pf->_token = _ready_tokens.front().first;
          pf->_num_deferrals = _ready_tokens.front().second;
          _ready_tokens.pop();
        }
        else {
          pf->_token = _num_tokens;
          pf->_num_deferrals = 0;
        }
      
      handle_token_dependency: 

        if (pf->_stop = false, _on_pipe(*pf, rt); pf->_stop == true) {
          // here, the pipeline is not stopped yet because other
          // lines of tasks may still be running their last stages
          return;
        }
        
        if (_num_tokens == pf->_token) {
          ++_num_tokens;
        }
      
        if (pf->_dependents.empty() == false){ 
          // check if the pf->_dependents have valid dependents
          _check_dependents(*pf); 
          
          // tokens in pf->_dependents are all valid dependents 
          if (pf->_dependents.size()) {
            
            // construct a data structure for pf in _deferred_tokens 
            _construct_deferred_tokens(*pf);
            goto pipeline;
          }

          // tokens in pf->_dependents are invalid dependents
          // directly goto on_pipe on the same line
          else {
            goto handle_token_dependency;
          }
        }
        
        // Every token within the deferral range needs to check
        // if it can resolve dependencies on other tokens.
        if (pf->_token <= _longest_deferral) {
          _resolve_token_dependencies(*pf); 
        }
      }
      else {
        _on_pipe(*pf, rt);
      }
      
      size_t c_f = pf->_pipe;
      size_t n_f = (pf->_pipe + 1) % num_pipes();
      size_t n_l = (pf->_line + 1) % num_lines();
      
      pf->_pipe = n_f;

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

      std::array<int, 2> retval;
      size_t n = 0;

      // downward dependency
      if(_pipes[c_f]->type() == PipeType::SERIAL &&
         _line(n_l, c_f).join_counter.fetch_sub(
           1, std::memory_order_acq_rel) == 1
        ) {
        retval[n++] = 1;
      }

      // forward dependency
      if(_line(pf->_line, n_f).join_counter.fetch_sub(
          1, std::memory_order_acq_rel) == 1
        ) {
        retval[n++] = 0;
      }

      // notice that the task index starts from 1
      switch(n) {
        case 2: {
          rt.schedule(_tasks[n_l+1]);
          goto pipeline;
        }
        case 1: {
          if (retval[0] == 1) {
            pf = &_pipeflows[n_l];
          }
          goto pipeline;
        }
      }
    }).name("rt-"s + std::to_string(l));

    _tasks[0].precede(_tasks[l+1]);
  }
}

}  // end of namespace tf -----------------------------------------------------





