#include <taskflow/taskflow.hpp>
#include <taskflow/pipeflow.hpp>

#include <iostream>
#include <variant>
#include <type_traits>
#include <tuple>
#include <functional>


//template<typename T>
//struct type_identity {
//  using type = T;
//};
//
//template <typename T>
//struct void_to_monostate {
//  using type = T;
//};
//
//template <>
//struct void_to_monostate<void> {
//  using type = std::monostate;
//};
//
//template <typename T>
//using void_to_monostate_t = typename void_to_monostate<T>::type;
//
//template <typename T, typename... Ts>
//struct unique_tuple_impl : public type_identity<T> {};
//
//template <
//  template<typename...> typename Tuple, typename... Ts, typename U, typename... Us
//>
//struct unique_tuple_impl<Tuple<Ts...>, U, Us...> : 
//  std::conditional_t<(std::is_same_v<U, Ts> || ...), 
//  unique_tuple_impl<Tuple<Ts...>, Us...>, 
//  unique_tuple_impl<Tuple<Ts..., U>, Us...>> {};
//
//template <typename Tuple>
//struct unique_tuple;
//
//template <template<typename...>typename Tuple, typename... Ts>
//struct unique_tuple<Tuple<Ts...>> : public unique_tuple_impl<Tuple<>, Ts...> {};
//
//template <typename Tuple>
//using unique_tuple_t = typename unique_tuple<Tuple>::type;
//
//// Pipeline filter interface
//
//enum class FilterType : int{
//  SERIAL = 0,
//  PARALLEL
//};
//
//template <typename I, typename O, typename C>
//struct Filter{
//
//  using input_type  = I;
//  using output_type = O;
//
//  Filter(FilterType d, C&& callable) : 
//    direction{d}, callable{std::forward<C>(callable)} {
//  }
//
//  FilterType direction;
//  C callable;
//};
//
//template <typename I, typename O, typename C>
//auto make_filter(FilterType dir, C&& callable) {
//  return Filter<I, O, C>{dir, std::forward<C>(callable)};
//}
//
//// Calls your func with tuple element.
//template <typename Func, typename Tuple, size_t N = 0>
//void visit_tuple(Func func, Tuple& tup, size_t idx) {
//  if (N == idx) {
//    std::invoke(func, std::get<N>(tup));
//    return;
//  }
//  if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
//    return visit_tuple<Func, Tuple, N + 1>(func, tup, idx);
//  }
//}
//  
//struct StreamControl {
//  bool stop {false};
//};
//
//template <typename... Fs>
//struct Pipeline {
//
//  static_assert(sizeof...(Fs)>0, "must have at least one filter");
//
//  // TODO
//  using data_type = unique_tuple_t<
//    std::variant<std::monostate, StreamControl, void_to_monostate_t<typename Fs::output_type> ...>
//  >;
//  
//  // TODO: just for hardcoded debugging
//  static_assert(std::is_same_v<data_type, std::variant<std::monostate, StreamControl, int, float>>);
//
//  // state constants
//  constexpr static int EMPTY   = 0;
//  constexpr static int ORPHAN  = 1;
//  constexpr static int ADOPTED = 2;
//
//  struct BufferData {
//    data_type data;
//    std::atomic<int> state {EMPTY};
//  };
//
//  struct FilterMeta {
//    FilterType type;
//    std::atomic<size_t> cursor;
//  };
//  
//  public:
//
//  /**
//  @brief constructs a pipeline object
//  */
//  Pipeline(size_t L, Fs&&... fs) : 
//    _L {L},
//    _filters {std::make_tuple(std::forward<Fs>(fs)...)},
//    _buffers (L * sizeof...(Fs)) {
//    
//    for(auto& c : _meta) {
//      c.cursor.store(0, std::memory_order_relaxed);
//    }
//
//    // TODO: initialize meta (type)
//    
//    // TODO
//    // construct the taskflow graph
//  }
//
//  /**
//  @brief queries the number of production lines
//  */
//  size_t num_lines() const {
//    return _L;
//  }
//
//  /**
//  @brief queries the number of filters
//  */
//  constexpr size_t num_filters() const {
//    return sizeof...(Fs);
//  }
//  
//
//  void make_taskflow(tf::Taskflow& taskflow) {
//
//    // init task
//    auto init = taskflow.emplace([&]() -> tf::SmallVector<int> { 
//      // first filter is SERIAL
//      if (std::get<0>(_filters).direction == FilterType::SERIAL) {
//        return {0}; 
//      }
//
//      // first filter is PARALLEL
//      else {
//        tf::SmallVector<int> ret; 
//        int i = 0;
//        while (i < _L) {
//          ret.emplace_back(i++);
//        }
//        return ret;
//      }
//    });
//
//    // create a task for each layer
//    for(size_t l = 0; l <_L; l++) {
//      _task_graph.emplace_back(
//        taskflow.emplace([&, l, f = 0]() mutable -> tf::SmallVector<int> {
//          tf::SmallVector<int> retval{}; 
//
//          // pipeline is full of items.
//          if (_N == _L) return {-1};
//
//          _NTotal.fetch_add(1, std::memory_order_release);
//
//          if (f == 0) {
//            // TODO
//            // d = get from input stream; void->nullptr
//            _on_filter(0, void, _buffers[l * num_filters()]);
//            _N.fetch_add(1, std::memory_order_release);  
//          }
//          else {
//            _on_filter(f,
//                       _buffers[l * _filters.size() + (f - 1)],
//                       _buffers[l * _filters.size() + f]);
//            _buffers[l * _filters.size() + f].state = EMPTY;
//          }
//
//          if (f == _filters.size()-1) {
//            _N.fetch_sub(1, std::memory_order_release);  
//          }
//         
//          // TODO : determine the position of f
//          f = (f + 1) % _filters.size();
//           
//          if (std::get<f>(_filters).direction == FilterType::SERIAL) {
//            _cursors[f].fetch_add(1, std::memory_order_release);  
//          }
//          _buffers[l * _filters.size() + f].state = ORPHAN;
//
//          // only one filter exists 
//          if (_filters.size() == 1) {
//            if (std::get<0>(_filters).direction == FilterType::SERIAL) {
//              if (_cursors[0]%_L == (l+1)%_L) {
//                return {0, 1};
//              }
//              return {0};
//            }
//            else {
//              return {0};
//            }
//          }
//
//          // TODO
//          //auto p_type = _meta[f-1].type; 
//          //auto c_type = _meta[f  ].type;
//          //auto n_type = _meta[f+1].type;
//
//          // first batch at filter 0
//          if (f == 0 && _NTotal < _L) {
//            if (std::get<0>(_filters).direction == FilterType::SERIAL && 
//                std::get<1>(_filters).direction == FilterType::SERIAL) {
//              if ((_cursors[1].load(std::memory_order_acquire) % _L == l) && 
//                  (_buffers[l * _filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED))) {
//                return {0, 1};
//              }
//              return {1};
//            }
//
//            if (std::get<0>(_filters).direction == FilterType::SERIAL && 
//                std::get<1>(_filters).direction == FilterType::PARALLEL) {
//              return {0, 1};
//            }
//
//            if (std::get<0>(_filters).direction == FilterType::PARALLEL && 
//                std::get<1>(_filters).direction == FilterType::PARALLEL) {
//              return {0};
//            }
//
//            if (std::get<0>(_filters).direction == FilterType::PARALLEL && 
//                std::get<1>(_filters).direction == FilterType::SERIAL) {
//              if ((_cursors[1].load(std::memory_order_acquire) % _L == l) && 
//                  (_buffers[l * _filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED))) {
//                return {0};
//              }
//              return {-1};
//            }
//          }
//
//          // General case: not first batch at filter 0 or first batch not at filter 0
//          else {
//            // sss
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::SERIAL && 
//                std::get<f>(_filters).direction == FilterType::SERIAL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::SERIAL) {
//              if ((_cursors[(f+1)%_filters.size()].load(std::memory_order_acquire) % _L == l) &&
//                  (_buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED))) {
//                retval.emplace_back(0);
//              }
//              if ((_cursors[(f-1+_filters.size())%_filters.size()].load(std::memory_order_acquire)%_L >= (l+2)%_L) &&
//                  (_buffers[((l+1)%_L) * _filters.size() + (f-1+_filters.size())%_filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED))) {
//                retval.emplace_back(1);
//              }
//
//              if (retval.empty()) {
//                return {-1};
//              }
//              return retval;
//            }
//
//            // pss
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::PARALLEL && 
//                std::get<f>(_filters).direction == FilterType::SERIAL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::SERIAL) {
//              if (_cursors[(f+1)%_filters.size()].load(std::memory_order_acquire)%_L == l &&
//                  _buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(0);
//              }
//              if (_buffers[((l+1)%_L) * _filters.size() + (f-1+_filters.size())%_filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(1);  
//              }
//              if (retval.empty()) {
//                return {-1};
//              }
//              return retval;
//            }
//            
//            // psp
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::PARALLEL && 
//                std::get<f>(_filters).direction == FilterType::SERIAL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::PARALLEL) {
//              if (_buffers[((l+1)%_L) * _filters.size() + (f-1+_filters.size())%_filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(1);
//              }
//              if (_buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(0);
//              }
//              return retval;
//            }
//
//            // ppp
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::PARALLEL && 
//                std::get<f>(_filters).direction == FilterType::PARALLEL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::PARALLEL) {
//              _buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED);
//              return {0};
//            }
//
//            // spp
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::SERIAL && 
//                std::get<f>(_filters).direction == FilterType::PARALLEL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::PARALLEL) {
//              _buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED);
//              return {0};
//            }
//
//            // pps
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::PARALLEL && 
//                std::get<f>(_filters).direction == FilterType::PARALLEL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::SERIAL) {
//              if (_cursors[(f+1)%_filters.size()].load(std::memory_order_acquire)%_L == l &&
//                  _buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                return {0};
//              }
//              return {-1};
//            }
//
//            // sps
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::PARALLEL && 
//                std::get<f>(_filters).direction == FilterType::PARALLEL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::SERIAL) {
//              if (_cursors[(f+1)%_filters.size()].load(std::memory_order_acquire)%_L == l &&
//                  _buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                return {0};
//              }
//              return {-1};
//            }
//
//            // ssp
//            if (std::get<(f-1+_filters.size())%_filters.size()>(_filters).direction == FilterType::SERIAL && 
//                std::get<f>(_filters).direction == FilterType::SERIAL &&
//                std::get<(f+1)%_filters.size()>(_filters).direction == FilterType::PARALLEL) {
//              if (_cursors[(f-1+_filters.size())%_filters.size()].load(std::memory_order_acquire)%_L == (l+2)%_L &&
//                  _buffers[((l+1)%_filters.size()) * _filters.size() + (f-1+_filters.size())%_filters.size()].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(1);
//              }
//              if (_buffers[l * _filters.size() + f].state.compare_exchange_strong(ORPHAN, ADOPTED)) {
//                retval.emplace_back(0);
//              }
//              return retval;
//            }
//          }
//
//        })
//      );
//    }
//
//    // Specify the dependencies of tasks
//    for (size_t l = 0; l < _L; l++) {
//      init.precede(_task_graph[l]);
//      _task_graph[l].precede(_task_graph[l], _task_graph[(l + 1) % _L]);
//    }
//  }
//
//  private:
//  
//  size_t _L;  // must be power of 2
//
//  std::atomic<size_t> _NTotal{0};
//
//  std::atomic<size_t> _N{0};
//
//  std::tuple<Fs...> _filters;
//  std::vector<BufferData> _buffers;
//  
//  std::array<FilterMeta, sizeof...(Fs)> _meta;
//
//  std::vector<tf::Task> _task_graph;
//
//  BufferData& _get_buffer_data(size_t l, size_t f) {
//    return _buffers[l*_filters.size() + f];
//  }
//  
//  // TODO: change to data_type* 
//  void _on_filter(size_t f, data_type& d_in, data_type& d_out) {
//    
//    visit_tuple([&](auto&& filter){
//
//      using F = std::decay_t<decltype(filter)>;
//      using I = typename F::input_type;
//      using O = typename F::output_type;
//
//      constexpr auto void_i = std::is_void_v<I>;
//      constexpr auto void_o = std::is_void_v<O>;
//
//      if constexpr(void_i && void_o) {
//        filter.callable();
//      }
//      else if constexpr(void_i && !void_o) {
//        std::get<O>(d_out) = filter.callable();
//      }
//      else if constexpr(!void_i && void_o) {
//        filter.callable(std::get<I>(d_in));
//      }
//      else {
//        std::get<O>(d_out) = filter.callable(std::get<I>(d_in));
//      }
//    }, _filters, f);
//  }
//
//};
//
////template <typename... Fs>
////auto make_pipeline(Fs&&... filters) {
////
////  using data_type = unique_tuple_t<std::variant<typename Fs::output_type ...>>;
////
////  //static_assert(std::is_same_v<data_type, std::variant<int, float, std::monostate>>);
////}
//
//int main()
//{
//   static_assert(
//     std::is_same_v<
//       unique_tuple_t<std::variant<int, double, double, void>>, std::variant<int, double, void>
//     >,
//     ""
//   );
//
//
//   Pipeline pl(4,
//     make_filter<tf::StreamControl&, int>   (FilterType::SERIAL, [](tf::StreamControl& stop){ 
//
//       if(myprogram_is_stop) {
//         stop = true;
//         return -1;
//       }
//
//       return 1; 
//     }),
//     make_filter<int, float> (FilterType::PARALLEL, [](int){ return 1.2f;}),
//     make_filter<float, void>(FilterType::SERIAL,   [](float){  }            )
//   );
//   
//   tf::Executor executor;
//   tf::Taskflow taskflow;
//   pl.make_taskflow(taskflow);
//   executor.run(taskflow).wait();
//
//
//   //std::tuple<std::string, int> tuple {"123", 4};
//   //visit_tuple([](auto& filter){
//   //  std::cout << filter << std::endl;
//   //}, tuple, 0);
//}
//
int main() {

}
