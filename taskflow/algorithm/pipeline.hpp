#pragma once

#include "../taskflow.hpp"

namespace tf {

enum class PipeType : int{
  PARALLEL = 1,  
  SERIAL   = 2
};

template <typename C>
class Pipe {

  template <typename... Fs>
  friend class Pipeline;

  public:

  Pipe(PipeType d, C&& callable) :
    _type{d}, _callable{std::forward<C>(callable)} {
  }

  private:

  PipeType _type;

  C _callable;
};




class Pipeflow {

  template <typename... Fs>
  friend class Pipeline;

  public:

  size_t line() const {
    return _line;
  }

  size_t pipe() const {
    return _pipe;
  }

  size_t token() const {
    return _token;
  }

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
  bool   _stop {false};
};

template <typename... Fs>
class Pipeline {
  
  static_assert(sizeof...(Fs)>0, "must have at least one pipe");


  struct Line {
    std::atomic<size_t> join_counter;
  };

  struct PipeMeta {
    PipeType type;
  };

  public:

  /**
  @brief constructs a pipeline object
  */
  Pipeline(size_t max_lines, Fs&&... fs);

  /**
  @brief queries the number of dataflow lines
  */
  size_t num_lines() const noexcept;

  /**
  @brief queries the number of pipes
  */
  constexpr size_t num_pipes() const noexcept;
  
  /**
  @brief resets the pipeline
  */
  void reset();
  
  /**
  @brief queries the number of tokens in the pipeline
  */
  size_t num_tokens() const noexcept;

  Graph& graph() { return _graph; }

  private:
  
  std::tuple<Fs...> _pipes;

  std::array<PipeMeta, sizeof...(Fs)> _meta;
  
  std::vector<std::array<Line, sizeof...(Fs)>> _lines; 
  
  std::vector<tf::Task> _tasks;
  
  size_t _num_tokens;

  Graph _graph;
    
  void _on_pipe(Pipeflow&);

  void _build();

};

// constructor
template <typename... Fs>
Pipeline<Fs...>::Pipeline(size_t max_lines, Fs&&... fs) :
  _pipes {std::make_tuple(std::forward<Fs>(fs)...)},
  _meta  {PipeMeta{fs._type}...},
  _lines (max_lines),
  _tasks (max_lines + 1) {

  if(std::get<0>(_pipes)._type == PipeType::PARALLEL) {
    TF_THROW("first pipe must be serial");
  }

  reset();
  _build();
}

template <typename... Fs>
size_t Pipeline<Fs...>::num_lines() const noexcept {
  return _lines.size();
}

template <typename... Fs>
constexpr size_t Pipeline<Fs...>::num_pipes() const noexcept {
  return sizeof...(Fs);
}

template <typename... Fs>
size_t Pipeline<Fs...>::num_tokens() const noexcept {
  return _num_tokens;
}

template <typename... Fs>
void Pipeline<Fs...>::reset() {

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
  
template <typename... Fs>
void Pipeline<Fs...>::_on_pipe(Pipeflow& pf) {
  visit_tuple(
    [&](auto&& pipe){ pipe._callable(pf); },
    _pipes, pf._pipe
  );
}

template <typename... Fs>
void Pipeline<Fs...>::_build() {
  
  using namespace std::literals::string_literals;
 
  FlowBuilder fb(_graph); 

  // init task
  _tasks[0] = fb.emplace([this]() {
    return static_cast<int>(_num_tokens % num_lines());
    /*// first pipe is SERIAL
    if (std::get<0>(_pipes)._type == PipeType::SERIAL) {
      return { static_cast<int>(_num_tokens % num_lines()) };
    }

    // first pipe is PARALLEL
    else {
      SmallVector<int> ret;
      ret.reserve(num_lines());
      for(size_t l=0; l<num_lines(); l++) {
        ret.push_back(l);
      }
      return ret;
    }*/
  }).name("p-cond");

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
        if (pf._stop = false, _on_pipe(pf); pf._stop == true) {
          // here, the pipeline is not stopped yet because other
          // lines of tasks may still be running their last stages
          return;
        }
        ++_num_tokens;
      }
      else {
        _on_pipe(pf);
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
          rt.executor().schedule(rt.worker(), _tasks[n_l+1]);
          goto pipeline;
        }
        case 1: {
          if (retval[0] == 0) {
            goto pipeline;
          }
          else {
            rt.executor().schedule(rt.worker(), _tasks[n_l+1]);
          }
        }
      }
    }).name("line-"s + std::to_string(l));

    _tasks[0].precede(_tasks[l+1]);
  }
}

}  // end of namespace tf -----------------------------------------------------





