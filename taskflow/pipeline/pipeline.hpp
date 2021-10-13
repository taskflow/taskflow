#pragma once

namespace tf {

enum class FilterType : int{
  SERIAL = 0,
  PARALLEL
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

  using data_type = D;

  // state constants
  constexpr static int EMPTY   = 0;
  constexpr static int ORPHAN  = 1;
  constexpr static int ADOPTED = 2;

  struct BufferData {
    data_type data;
    std::atomic<int> state {EMPTY};
  };

  struct FilterMeta {
    FilterType type;
    std::atomic<size_t> cursor;
    int type_encode;
  };
 
  public:

  /**
  @brief constructs a pipeline object
  */
  Pipeline(Fs&&... fs) :
    _filters {std::make_tuple(std::forward<Fs>(fs)...)},
    _meta    {FilterMeta{fs._type, 0, 0}...} {
    
    for(size_t i = 0; i < num_filters(); i++) {
      auto p_type = static_cast<int>(_meta[(i + num_filters() - 1) % num_filters()].type);
      auto c_type = static_cast<int>(_meta[i].type    );
      auto n_type = static_cast<int>(_meta[(i + 1) % num_filters()].type);
      _meta[i].type_encode = p_type + (c_type << 1) + (n_type << 2);
    }
  }

  /**
  @brief queries the number of dataflow lines
  */
  constexpr size_t num_lines() const {
    return L;
  }

  /**
  @brief queries the number of filters
  */
  constexpr size_t num_filters() const {
    return sizeof...(Fs);
  }
 
  auto make_taskflow(FlowBuilder& taskflow) {

    std::array<tf::Task, L + 1> tasks;
    
    // init task
    tasks[0] = taskflow.emplace([&]() -> SmallVector<int> {
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

    //std::vector<Task> _task_graph;

    // create a task for each layer
    for(size_t l = 0; l < num_lines(); l++) {
      tasks[l+1] = taskflow.emplace(
        [&, l, f = size_t{0}]() mutable -> SmallVector<int> {
          if (f == 0) {
            if (_on_filter(0, nullptr, &_get_buffer(l, f).data) == true) {
              return {};
            }
            else {
              _NTotal.fetch_add(1, std::memory_order_relaxed);
            }
          }
          else {
            _on_filter(
              f, &_get_buffer(l, f - 1).data, &_get_buffer(l, f).data
            );
            _get_buffer(l, f - 1).state.store(EMPTY, std::memory_order_relaxed);
          }
          
          _get_buffer(l, f).state.store(ORPHAN, std::memory_order_release);
          
          if (_meta[f].type == FilterType::SERIAL) {
            _meta[f].cursor.fetch_add(1, std::memory_order_release);  
          }
          
          size_t p_f = (f + num_filters() - 1) % num_filters();
          size_t c_f = f;
          size_t n_f = (f + 1) % num_filters();
          size_t n_l = (l + 1) % num_lines();
          f = n_f;

          if (num_filters() == 1) {
            // SERIAL
            if (_meta[0].type == FilterType::SERIAL) {
              // TODO (10/11): is this redundant?
              if (_meta[0].cursor.load(std::memory_order_acquire) % num_lines() == (l + 1) % num_lines()) {
                return {1};
              }
              return {};
            }
            // PARALLEL
            else {
              return {0};
            }
          }
          
          // general case #filters >= 2
          int orphan = ORPHAN;

          // first batch at filter 0
          if (c_f == 0 && _NTotal < num_lines()) {
            // ss
            // 0        -> 0
            // |           |
            // v           v
            // (l, c_f) -> 0
            // |           |
            // v           v
            // 0        -> 0
            if (_meta[0].type == FilterType::SERIAL &&
                _meta[1].type == FilterType::SERIAL) {
              if ((_meta[1].cursor.load(std::memory_order_acquire) % num_lines() == l) &&
                  (_get_buffer(l, 0).state).compare_exchange_strong(orphan, ADOPTED,
                                            std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return {0, 1};
              }
              return {1};
            }

            // sp
            // 0        -> 0
            // |         
            // v         
            // (l, c_f) -> 0
            // |         
            // v         
            // 0        -> 0
            if (_meta[0].type == FilterType::SERIAL &&
                _meta[1].type == FilterType::PARALLEL) {
              return {0, 1};
            }

            // pp
            // 0        -> 0
            //          
            //          
            // (l, c_f) -> 0
            //          
            //          
            // 0        -> 0
            if (_meta[0].type == FilterType::PARALLEL &&
                _meta[1].type == FilterType::PARALLEL) {
              return {0};
            }

            // ps
            // 0        ->  0
            //              |
            //              v
            // (l, c_f) ->  0
            //              |
            //              v
            // 0        ->  0
            if (_meta[0].type == FilterType::PARALLEL &&
                _meta[1].type == FilterType::SERIAL) {
              if ((_meta[1].cursor.load(std::memory_order_acquire) % num_lines() == l) &&
                  (_get_buffer(l, 0).state).compare_exchange_strong(orphan, ADOPTED,
                                            std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return {0};
              }
              return {-1};
            }

            return {};
          }

          // General case: not first batch at filter 0 or first batch not at filter 0
          else {
          
            SmallVector<int> retval;

            switch(_meta[c_f].type_encode) {
              
              // sss
              // 0 -> 0        -> 0
              // |    |           |
              // v    v           v
              // 0 -> (l, c_f) -> 0
              // |    |           |
              // v    v           v
              // 0 -> 0        -> 0
              // |    |           |
              // v    v           v
              // 0 -> 0        -> 0
              case 0:
                if ((_meta[n_f].cursor.load(std::memory_order_acquire) % num_lines() == l) &&
                    (_get_buffer(l, c_f).state).compare_exchange_strong(orphan, ADOPTED, 
                                                std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  retval.emplace_back(0);
                }
                if ((_get_buffer(n_l, p_f).state).compare_exchange_strong(orphan, ADOPTED,
                                                  std::memory_order_seq_cst, std::memory_order_relaxed)){
                  retval.emplace_back(1);
                }
                return retval;
              break;

              // pss
              // 0 -> 0        -> 0
              //      |           |
              //      v           v
              // 0 -> (l, c_f) -> 0
              //      |           |
              //      v           v
              // 0 -> 0        -> 0
              //      |           |
              //      v           v
              // 0 -> 0        -> 0
              case 1:
                if (_meta[n_f].cursor.load(std::memory_order_acquire) % num_lines() == l &&
                   (_get_buffer(l, c_f).state).compare_exchange_strong(orphan, ADOPTED,
                                               std::memory_order_seq_cst, std::memory_order_relaxed)){
                  retval.emplace_back(0);
                }
                if ((_get_buffer(n_l, p_f).state).compare_exchange_strong(orphan, ADOPTED,
                                                  std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  retval.emplace_back(1);  
                }
                return retval;
              break;

              // sps
              // 0 -> 0        -> 0
              // |                |
              // v                v
              // 0 -> (l, c_f) -> 0
              // |                |
              // v                v 
              // 0 -> 0        -> 0
              // |                |
              // v                v
              // 0 -> 0        -> 0
              case 2:
                if (_meta[n_f].cursor.load(std::memory_order_acquire) % num_lines() == l &&
                   (_get_buffer(l, c_f).state).compare_exchange_strong(orphan, ADOPTED, 
                                               std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  return {0};
                }
                return {};
              break;
                
              // pps
              // 0 -> 0        -> 0
              //                  |
              //                  v
              // 0 -> (l, c_f) -> 0
              //                  |
              //                  v 
              // 0 -> 0        -> 0
              //                  |
              //                  v
              // 0 -> 0        -> 0
              case 3:
                if (_meta[n_f].cursor.load(std::memory_order_acquire) % num_lines() == l &&
                   (_get_buffer(l, c_f).state).compare_exchange_strong(orphan, ADOPTED, 
                                               std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  return {0};
                }
                return {};
              break;
                
              // ssp
              // 0 -> 0        -> 0
              // |    |           
              // v    v           
              // 0 -> (l, c_f) -> 0
              // |    |           
              // v    v           
              // 0 -> 0        -> 0
              // |    |           
              // v    v           
              // 0 -> 0        -> 0
              case 4:
                retval.emplace_back(0);
                if ((_get_buffer(n_l, p_f).state).compare_exchange_strong(orphan, ADOPTED,
                                                  std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  retval.emplace_back(1);
                }
                _get_buffer(l, c_f).state.store(ADOPTED, std::memory_order_relaxed);

                return retval;
              break;

              // psp
              // 0 -> 0        -> 0
              //      |           
              //      v           
              // 0 -> (l, c_f) -> 0
              //      |           
              //      v           
              // 0 -> 0        -> 0
              //      |           
              //      v           
              // 0 -> 0        -> 0
              case 5:
                retval.emplace_back(0);
                if ((_get_buffer(n_l, p_f).state).compare_exchange_strong(orphan, ADOPTED,
                                                  std::memory_order_seq_cst, std::memory_order_relaxed)) {
                  retval.emplace_back(1);
                }
                _get_buffer(l, c_f).state.store(ADOPTED, std::memory_order_relaxed);
                return retval;
              break;

              // spp
              // 0 ->  0        -> 0
              // |                
              // v                
              // 0 -> (l, c_f)  -> 0
              // |                
              // V                
              // 0 ->  0        -> 0
              // |                
              // V                
              // 0 ->  0        -> 0
              case 6:
                _get_buffer(l, c_f).state.store(ADOPTED, std::memory_order_relaxed);
                return {0};
              break;

              // ppp
              // 0 ->  0        -> 0
              //                 
              //                 
              // 0 -> (l, c_f)  -> 0
              //                 
              //                 
              // 0 ->  0        -> 0
              //                 
              //                 
              // 0 ->  0        -> 0
              case 7:
                _get_buffer(l, c_f).state.store(ADOPTED, std::memory_order_relaxed);
                return {0};
              break;


              default:
                assert(false); 
                return {};
              break;
            }
          }

        }
      );
    }

    // Specify the dependencies of tasks
    for (size_t l = 0; l < num_lines(); l++) {
      tasks[0].precede(tasks[l+1]);
      tasks[l+1].precede(tasks[l+1], tasks[(l+2)%num_lines()]);
      //_task_graph[l].precede(_task_graph[l], _task_graph[(l + 1) % num_lines()]);
    }

    return tasks;
  }

  private:

  std::atomic<size_t> _NTotal{0};
  std::atomic<bool> _stop{false};
  
  std::tuple<Fs...> _filters;
  
  std::array<BufferData, L*sizeof...(Fs)> _buffers;
  std::array<FilterMeta, sizeof...(Fs)> _meta;

  BufferData& _get_buffer(size_t l, size_t f) {
    return _buffers[l * sizeof...(Fs) + f];
  }

  bool _on_filter(size_t f, data_type* d_in, data_type* d_out) {
    visit_tuple([&](auto&& filter){
      Dataflow<data_type> df(d_in, d_out, _stop);
      filter._callable(df);
    }, _filters, f);
    return _stop.load(std::memory_order_relaxed);
  }

};

//
template <typename D, size_t L, typename... Fs>
auto make_pipeline(Fs&&... filters) {
  return Pipeline<D, L, Fs...>{std::forward<Fs>(filters)...};
}

// ----------------------------------------------------------------------------
// Forward declaration: FlowBuilder::pipeline
// ----------------------------------------------------------------------------

template <typename Pipeline>
auto FlowBuilder::pipeline(Pipeline& p) {
  return p.make_taskflow(*this);  
}

}  // end of namespace tf -----------------------------------------------------





