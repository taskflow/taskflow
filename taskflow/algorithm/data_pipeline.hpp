#pragma once

#include "../taskflow.hpp"
#include "pipeline.hpp"
#include "../utility/traits.hpp"


namespace tf {

// ----------------------------------------------------------------------------
// Class Definition: DataPipe
// ----------------------------------------------------------------------------

/**
@class DataPipe

@brief class to create a data pipe object for a data pipeline stage

A data pipe represents a stage of a data pipeline. A data pipe can be either
@em parallel direction or @em serial direction (specified by tf::PipeType)
and is coupled with a callable to invoke by the pipeline scheduler.

You must call the make_data_type function to create a DataPipe object.
The input and output type you give to the make_data_pipe function will be decayed to its original form.
The arguments of your lambda can be either a copy or a reference, we will use it by value or by reference as you want.

Usually your lambda function takes only one argument as input(The first pipe must takes a Pipeflow object to control flow).
You can also add an additional pipeflow argument in the lambda function to get token and line position of current pipe.

@code
1: tf::make_data_pipe<int, std::string>(
2:   tf::PipeType::SERIAL, 
3:   [](int input) {return std::to_string(input + 100);}
4: )
@endcode

@code
tf::make_data_pipe<int, std::string>(
  tf::PipeType::SERIAL, 
  [](int input, tf::Pipeflow& pf) {
    std::cout << "token: " << pf.token() << std:: endl;
    std::cout << "line: " << pf.line() << std::endl;
    return std::to_string(input + 100);
  }
)
@endcode

The pipeflow object is used to query the statistics of a scheduling token
in the pipeline, such as pipe, line, and token numbers.
*/
template <typename Input, typename Output, typename C>
class DataPipe {

  template <typename... Ps>
  friend class DataPipeline;

  public:

  /**
  @brief alias of the type
  */
  using callable_t = C;
  using input_t = Input;
  using output_t = Output;

  /**
  @brief default constructor
  */
  DataPipe() = default;

  /**
  @brief
  You must use the make_data_pipe function to create a DataPipe object.
  */
  DataPipe(PipeType d, callable_t&& callable) :
    _type{d}, _callable{std::forward<callable_t>(callable)} {
  }

  /**
  @brief queries the type of the data pipe

  Returns the type of the callable.
  */
  PipeType type() const {
    return _type;
  }

  /**
  @brief assigns a new type to the data pipe

  @param type a tf::PipeType variable
  */
  void type(PipeType type) {
    _type = type;
  }

  /**
  @brief assigns a new callable to the data pipe

  @tparam U callable type
  @param callable a callable object constructible from std::function<void(tf::Pipeflow&)>

  Assigns a new callable to the data pipe with universal forwarding.
  */
  template <typename U>
  void callable(U&& callable) {
    _callable = std::forward<U>(callable);
  }

  private:

  PipeType _type;

  callable_t _callable;
};

/**
@brief constructs the data pipe object

@param d pipe type (tf::PipeType)
@param callable callable type

You must call the make_data_type function to create a DataPipe object.
The input and output type you give to the make_data_pipe function will be decayed to its original form.
The arguments of your lambda can be either a copy or a reference, we will use it by value or by reference as you want.

Usually your lambda function takes only one argument as input(The first pipe must takes a Pipeflow object to control flow).
You can also add an additional pipeflow argument in the lambda function to get token and line position of current pipe.

@code
1: tf::make_data_pipe<int, std::string>(
2:   tf::PipeType::SERIAL, 
3:   [](int input) {return std::to_string(input + 100);}
4: )
@endcode

@code
tf::make_data_pipe<int, std::string>(
  tf::PipeType::SERIAL, 
  [](int input, tf::Pipeflow& pf) {
    std::cout << "token: " << pf.token() << std:: endl;
    std::cout << "line: " << pf.line() << std::endl;
    return std::to_string(input + 100);
  }
)
@endcode

When creating a pipeline, the direction of the first pipe must be serial
(tf::PipeType::SERIAL).
*/
template <typename Input, typename Output, typename C>
auto make_data_pipe(PipeType d, C&& callable) {
  return DataPipe<Input, Output, C>(d, std::forward<C>(callable));
}


// ----------------------------------------------------------------------------
// Class Definition: DataPipeline
// ----------------------------------------------------------------------------

/**
@class DataPipeline

@brief class to create a data pipeline scheduling framework

@tparam Ps pipe types

You can study the @ref ParallelPipeline first, and then you will know the concept of Data Pipeline.
Here we add a layer of data abstraction on top of the original @ref ParallelPipeline, 
so that users can either choose to prepare the data buffer manually and use an efficient pipeline without data abstraction, 
or choose to use a pipeline with data abstraction and let the program allocate the buffer automatically.

The following code is an example:

@code{.cpp}
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline.hpp>

int main() {

  // data flow => void -> int -> std::string -> float -> void 
  tf::Taskflow taskflow("pipeline");
  tf::Executor executor;

  const size_t num_lines = 4;

  tf::DataPipeline pl(num_lines,
    tf::make_data_pipe<void, int>(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) -> int{
      if(pf.token() == 5) {
        pf.stop();
        return 0;
      }
      else {
        return pf.token();
      }
    }),

    tf::make_data_pipe<int, std::string>(tf::PipeType::SERIAL, [](int& input) {
      return std::to_string(input + 100);
    }),

    tf::make_data_pipe<std::string, void>(tf::PipeType::SERIAL, [](std::string& input) {
      std::cout << input << std::endl;
    })
  );

  // build the pipeline graph using composition
  taskflow.composed_of(pl).name("pipeline");

  // dump the pipeline graph structure (with composition)
  taskflow.dump(std::cout);

  // run the pipeline
  executor.run(taskflow).wait();

  return 0;
}
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
*/
template <typename... Ps>
class DataPipeline {

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
  DataPipeline(size_t num_lines, Ps&&... ps);

  /**
  @brief constructs a pipeline object

  @param num_lines the number of parallel lines
  @param ps a tuple of pipes

  Constructs a pipeline of up to @c num_lines parallel lines to schedule
  tokens through the given linear chain of pipes.
  The first pipe must define a serial direction (tf::PipeType::SERIAL)
  or an exception will be thrown.
  */
  DataPipeline(size_t num_lines, std::tuple<Ps...>&& ps);

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
  using variant_t = unique_variant_t<std::variant<std::conditional_t<std::is_void_v<typename Ps::output_t>, std::monostate, std::decay_t<typename Ps::output_t>>...>>;
  // std::vector<variant_t> _buffer;
  alignas (TF_CACHELINE_SIZE << 1) std::vector<padded_t<variant_t> > _buffer;

  template <size_t... I>
  auto _gen_meta(std::tuple<Ps...>&&, std::index_sequence<I...>);

  void _on_pipe(Pipeflow&, Runtime&);
  void _build();
};

// constructor
template <typename... Ps>
DataPipeline<Ps...>::DataPipeline(size_t num_lines, Ps&&... ps) :
  _pipes     {std::make_tuple(std::forward<Ps>(ps)...)},
  _meta      {PipeMeta{ps.type()}...},
  _lines     (num_lines),
  _tasks     (num_lines + 1),
  _pipeflows (num_lines),
  _buffer    (num_lines) {

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
DataPipeline<Ps...>::DataPipeline(size_t num_lines, std::tuple<Ps...>&& ps) :
  _pipes     {std::forward<std::tuple<Ps...>>(ps)},
  _meta      {_gen_meta(
    std::forward<std::tuple<Ps...>>(ps), std::make_index_sequence<sizeof...(Ps)>{}
  )},
  _lines     (num_lines),
  _tasks     (num_lines + 1),
  _pipeflows (num_lines),
  _buffer    (num_lines) {

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
auto DataPipeline<Ps...>::_gen_meta(std::tuple<Ps...>&& ps, std::index_sequence<I...>) {
  return std::array{PipeMeta{std::get<I>(ps).type()}...};
}

// Function: num_lines
template <typename... Ps>
size_t DataPipeline<Ps...>::num_lines() const noexcept {
  return _pipeflows.size();
}

// Function: num_pipes
template <typename... Ps>
constexpr size_t DataPipeline<Ps...>::num_pipes() const noexcept {
  return sizeof...(Ps);
}

// Function: num_tokens
template <typename... Ps>
size_t DataPipeline<Ps...>::num_tokens() const noexcept {
  return _num_tokens;
}

// Function: graph
template <typename... Ps>
Graph& DataPipeline<Ps...>::graph() {
  return _graph;
}

// Function: reset
template <typename... Ps>
void DataPipeline<Ps...>::reset() {

  _num_tokens = 0;

  for(size_t l = 0; l<num_lines(); l++) {
    _pipeflows[l]._pipe = 0;
    _pipeflows[l]._line = l;
  }

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
void DataPipeline<Ps...>::_on_pipe(Pipeflow& pf, Runtime& rt) {
  visit_tuple([&](auto&& pipe){
    using callable_t = typename std::decay_t<decltype(pipe)>::callable_t;
    using input_t = std::decay_t<typename std::decay_t<decltype(pipe)>::input_t>;
    using output_t = std::decay_t<typename std::decay_t<decltype(pipe)>::output_t>;
    
    if constexpr (std::is_invocable_v<callable_t, Pipeflow&>) {
      if constexpr (std::is_void_v<output_t>) {
        pipe._callable(pf);
      } else {
        _buffer[pf._line].v = pipe._callable(pf);
      }
    }
    else if constexpr (std::is_invocable_v<callable_t, input_t&>) {
      if constexpr (std::is_void_v<output_t>) {
        pipe._callable(std::get<input_t>(_buffer[pf._line].v));
      } else {
        _buffer[pf._line].v = pipe._callable(std::get<input_t>(_buffer[pf._line].v));
      }
    }
    else if constexpr (std::is_invocable_v<callable_t, input_t&, Pipeflow&>) {
      if constexpr (std::is_void_v<output_t>) {
        pipe._callable(std::get<input_t>(_buffer[pf._line].v), pf);
      } else {
        _buffer[pf._line].v = pipe._callable(std::get<input_t>(_buffer[pf._line].v), pf);
      }
    }
    else if constexpr(std::is_invocable_v<callable_t, Pipeflow&, Runtime&>) {
      pipe._callable(pf, rt);
    }
    else {
      static_assert(dependent_false_v<callable_t>, "un-supported pipe callable type");
    }
  }, _pipes, pf._pipe);
}

// Procedure: _build
template <typename... Ps>
void DataPipeline<Ps...>::_build() {

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

      if (pf->_pipe == 0) {
        pf->_token = _num_tokens;
        if (pf->_stop = false, _on_pipe(*pf, rt); pf->_stop == true) {
          // here, the pipeline is not stopped yet because other
          // lines of tasks may still be running their last stages
          return;
        }
        ++_num_tokens;
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





