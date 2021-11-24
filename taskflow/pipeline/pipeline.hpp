#pragma once

namespace tf {

enum class PipeType : int{
  PARALLEL = 1,  
  SERIAL   = 2
};

template <typename C>
class Pipe{

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

  //tf::Subflow& subflow() {
  //  return _sf;   
  //}
  
  private:

  Pipeflow(size_t line, size_t pipe) :
    _line {line},
    _pipe {pipe} {
  }

  size_t _line;
  size_t _pipe;
  size_t _token;
  bool   _stop {false};
  //tf::Subflow& _sf;
};

template <typename... Fs>
class Pipeline {
  
  static_assert(sizeof...(Fs)>0, "must have at least one pipe");

  friend class FlowBuilder;

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



  private:
  
  std::tuple<Fs...> _pipes;

  std::array<PipeMeta, sizeof...(Fs)> _meta;
  
  std::vector<std::array<Line, sizeof...(Fs)>> _lines; 
  
  std::vector<tf::Task> _tasks;
  
  Line& _get_line(size_t l, size_t f);

  size_t _num_tokens;
  
  void _on_pipe(Pipeflow&);

  auto _build(FlowBuilder&);

};

// constructor
template <typename... Fs>
Pipeline<Fs...>::Pipeline(size_t max_lines, Fs&&... fs) :
  _pipes {std::make_tuple(std::forward<Fs>(fs)...)},
  _meta  {PipeMeta{fs._type}...},
  _lines (max_lines),
  _tasks (max_lines + 1) {

  // TODO: throw exception if the first pipe is not serial
  reset();
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

  _get_line(0, 0).join_counter.store(0, std::memory_order_relaxed);

  for(size_t l=1; l<num_lines(); l++) {
    for(size_t f=1; f<num_pipes(); f++) {
      _get_line(l, f).join_counter.store(
        static_cast<size_t>(_meta[f].type), std::memory_order_relaxed
      );
    }
  }

  for(size_t f=1; f<num_pipes(); f++) {
    _get_line(0, f).join_counter.store(1, std::memory_order_relaxed);
  }

  for(size_t l=1; l<num_lines(); l++) {
    _get_line(l, 0).join_counter.store(
      static_cast<size_t>(_meta[0].type) - 1, std::memory_order_relaxed
    );
  }
}
  
template <typename... Fs>
typename Pipeline<Fs...>::Line& 
Pipeline<Fs...>::_get_line(size_t l, size_t f) {
  return _lines[l][f];
}

template <typename... Fs>
void Pipeline<Fs...>::_on_pipe(Pipeflow& pf) {

  //pf._subflow.reset();

  visit_tuple(
    [&](auto&& pipe){ pipe._callable(pf); },
    _pipes, pf._pipe
  );

  //if(pf._subflow._joinable) {
  //  pf._subflow.join();
  //}
}

template <typename... Fs>
auto Pipeline<Fs...>::_build(FlowBuilder& fb) {

  // init task
  _tasks[0] = fb.emplace([this]() {
    return  static_cast<int>(_num_tokens % num_lines());
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
  });

  // create a task for each layer
  for(size_t l = 0; l < num_lines(); l++) {

    _tasks[l + 1] = fb.emplace(
    [this, pf = Pipeflow{l, 0}] (tf::Runtime& rt) mutable {

    pipeline:

      _get_line(pf._line, pf._pipe).join_counter.store(
        static_cast<size_t>(_meta[pf._pipe].type), std::memory_order_relaxed
      );

      if (pf._pipe == 0) {
        pf._token = _num_tokens;
        if (pf._stop = false, _on_pipe(pf); pf._stop == true) {
          //return {};
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
         _get_line(n_l, c_f).join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retval.push_back(1);
      }
      
      // forward dependency
      if(_get_line(pf._line, n_f).join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retval.push_back(0);
      }
      
      // notice that the index of task starts from 1
      switch(retval.size()) {
        case 2: {
          rt.executor().schedule(_tasks[n_l+1]);
          goto pipeline;
        }
        case 1: {
          if (retval[0] == 0) {
            goto pipeline;
          }
          else {
            rt.executor().schedule(_tasks[n_l+1]);
          }
        }
      }
    });
  }

  // Specify the dependencies of tasks
  for (size_t l = 1; l < num_lines() + 1; l++) {
    _tasks[0].precede(_tasks[l]);
    //tasks[l].precede(tasks[l], tasks[l % num_lines() + 1]);
  }

  return _tasks;
}

// ----------------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------------

template <typename... Fs>
auto make_pipeline(size_t max_lines, Fs&&... pipes) {
  return Pipeline<Fs...>{max_lines, std::forward<Fs>(pipes)...};
}

template <typename... Fs>
auto make_unique_pipeline(size_t max_lines, Fs&&... pipes) {
  return std::make_unique<Pipeline<Fs...>>(
    max_lines, std::forward<Fs>(pipes)...
  );
}

template <typename... Fs>
auto make_shared_pipeline(size_t max_lines, Fs&&... pipes) {
  return std::make_shared<Pipeline<Fs...>>(
    max_lines, std::forward<Fs>(pipes)...
  );
}

// ----------------------------------------------------------------------------
// Forward declaration: FlowBuilder::pipeline
// ----------------------------------------------------------------------------

template <typename Pipeline>
auto FlowBuilder::pipeline(Pipeline& p) {
  return p._build(*this);  
}

}  // end of namespace tf -----------------------------------------------------





