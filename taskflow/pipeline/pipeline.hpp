#pragma once

namespace tf {

enum class FilterType : int{
  PARALLEL = 1,  
  SERIAL   = 2
};

template <typename C>
class Filter{

  template <typename D, size_t L, typename... Fs>
  friend class Pipeline;

  public:

  Filter(FilterType d, C&& callable) :
    _type{d}, _callable{std::forward<C>(callable)} {
  }

  private:

  FilterType _type;

  C _callable;
};

template <typename T>
class Dataflow {

  template <typename D, size_t L, typename... Fs>
  friend class Pipeline;

  public:

  void stop() {
    _stop.store(true, std::memory_order_relaxed);
  }

  T& at_input() { 
    if(!_input) {
      throw std::logic_error("input doesn't exist");
    }
    return *_input;  
  }

  T& at_output() { 
    if(!_output) {
      throw std::logic_error("output doesn't exist");
    }
    return *_output; 
  }

  T* input() {
    return _input;
  }

  T* output() {
    return _output;
  }

  private:

  Dataflow(T* input, T* output, std::atomic<bool>& stop) : 
    _input  {input}, 
    _output {output},
    _stop   {stop} {
  }

  T* _input;
  T* _output;
  
  std::atomic<bool>& _stop;
};

template <typename D, size_t L, typename... Fs>
class Pipeline {
  
  static_assert(sizeof...(Fs)>0, "must have at least one filter");
  static_assert(L>0, "must have at least one data line");

  friend class FlowBuilder;

  struct BufferData {
    D data;
    std::atomic<size_t> join_counter;
  };

  struct FilterMeta {
    FilterType type;
  };

  public:

  /**
  @brief constructs a pipeline object
  */
  Pipeline(size_t throttle, Fs&&... fs);

  /**
  @brief queries the number of dataflow lines
  */
  size_t num_lines() const noexcept;

  /**
  @brief queries the number of filters
  */
  constexpr size_t num_filters() const noexcept;

  private:

  std::atomic<bool> _stop;
  
  std::tuple<Fs...> _filters;

  std::array<FilterMeta, sizeof...(Fs)> _meta;
  
  //SmallVector<std::array<BufferData, sizeof...(Fs)>, L> _buffers; 
  std::vector<std::array<BufferData, sizeof...(Fs)>> _buffers; 

  BufferData& _get_buffer(size_t l, size_t f);

  bool _on_filter(size_t f, D* d_in, D* d_out);

  void _set_up_pipeline();
  
  auto _build(FlowBuilder&);

};

// constructor
template <typename D, size_t L, typename... Fs>
Pipeline<D, L, Fs...>::Pipeline(size_t throttle, Fs&&... fs) :
  _filters {std::make_tuple(std::forward<Fs>(fs)...)},
  _meta    {FilterMeta{fs._type}...},
  _buffers (throttle) {
}

template <typename D, size_t L, typename... Fs>
size_t Pipeline<D, L, Fs...>::num_lines() const noexcept {
  return _buffers.size();
}

template <typename D, size_t L, typename... Fs>
constexpr size_t Pipeline<D, L, Fs...>::num_filters() const noexcept {
  return sizeof...(Fs);
}

template <typename D, size_t L, typename... Fs>
void Pipeline<D, L, Fs...>::_set_up_pipeline() {

  _stop.store(false, std::memory_order_relaxed);

  _get_buffer(0, 0).join_counter.store(0, std::memory_order_relaxed);

  for(size_t l=1; l<num_lines(); l++) {
    for(size_t f=1; f<num_filters(); f++) {
      _get_buffer(l, f).join_counter.store(
        static_cast<size_t>(_meta[f].type), std::memory_order_relaxed
      );
    }
  }

  for(size_t f=1; f<num_filters(); f++) {
    _get_buffer(0, f).join_counter.store(1, std::memory_order_relaxed);
  }

  for(size_t l=1; l<num_lines(); l++) {
    _get_buffer(l, 0).join_counter.store(
      static_cast<size_t>(_meta[0].type) - 1, std::memory_order_relaxed
    );
  }
}
  
template <typename D, size_t L, typename... Fs>
typename Pipeline<D, L, Fs...>::BufferData& 
Pipeline<D, L, Fs...>::_get_buffer(size_t l, size_t f) {
  return _buffers[l][f];
}

template <typename D, size_t L, typename... Fs>
bool Pipeline<D, L, Fs...>::_on_filter(size_t f, D* d_in, D* d_out) {
  visit_tuple([&](auto&& filter){
    Dataflow<D> df(d_in, d_out, _stop);
    filter._callable(df);
  }, _filters, f);
  return _stop.load(std::memory_order_relaxed);
}

template <typename D, size_t L, typename... Fs>
auto Pipeline<D, L, Fs...>::_build(FlowBuilder& fb) {

  std::vector<tf::Task> tasks(num_lines() + 1);
  
  // init task
  tasks[0] = fb.emplace([this]() -> SmallVector<int> {

    _set_up_pipeline();

    // first filter is SERIAL
    if (std::get<0>(_filters)._type == FilterType::SERIAL) {
      return {0};
    }

    // first filter is PARALLEL
    else {
      SmallVector<int> ret;
      ret.reserve(num_lines());
      for(size_t l=0; l<num_lines(); l++) {
        ret.push_back(l);
      }
      return ret;
    }
  });

  // create a task for each layer
  for(size_t l = 0; l < num_lines(); l++) {
    tasks[l+1] = fb.emplace(
    [this, l, f = size_t{0}]() mutable -> SmallVector<int> {
      
      _get_buffer(l, f).join_counter.store(
        static_cast<size_t>(_meta[f].type), std::memory_order_relaxed
      );

      if (f == 0) {
        if (_on_filter(0, nullptr, &_get_buffer(l, f).data) == true) {
          return {};
        }
      }
      else {
        _on_filter(
          f, &_get_buffer(l, f - 1).data, &_get_buffer(l, f).data
        );
      }
      
      size_t c_f = f;
      size_t n_f = (f + 1) % num_filters();
      size_t n_l = (l + 1) % num_lines();

      f = n_f;
      
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

      SmallVector<int> retval;

      // downward dependency
      if(_meta[c_f].type == FilterType::SERIAL && 
         _get_buffer(n_l, c_f).join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retval.push_back(1);
      }
      
      // forward dependency
      if(_get_buffer(l, n_f).join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        retval.push_back(0);
      }

      return retval;
    });
  }

  // Specify the dependencies of tasks
  for (size_t l = 1; l < num_lines() + 1; l++) {
    tasks[0].precede(tasks[l]);
    tasks[l].precede(tasks[l], tasks[l % num_lines() + 1]);
  }

  return tasks;
}

// ----------------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------------

template <typename D, size_t L = 8, typename... Fs>
auto make_pipeline(size_t throttle, Fs&&... filters) {
  return Pipeline<D, L, Fs...>{throttle, std::forward<Fs>(filters)...};
}

template <typename D, size_t L = 8, typename... Fs>
auto make_unique_pipeline(size_t throttle, Fs&&... filters) {
  return std::make_unique<Pipeline<D, L, Fs...>>(
    throttle, std::forward<Fs>(filters)...
  );
}

template <typename D, size_t L = 8, typename... Fs>
auto make_shared_pipeline(size_t throttle, Fs&&... filters) {
  return std::make_shared<Pipeline<D, L, Fs...>>(
    throttle, std::forward<Fs>(filters)...
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





