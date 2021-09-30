#include <taskflow/taskflow.hpp>
#include <taskflow/pipeflow.hpp>

#include <iostream>
#include <variant>
#include <type_traits>
#include <tuple>
#include <functional>

template<typename T>
struct type_identity {
  using type = T;
};

template <typename T>
struct void_to_monostate {
  using type = T;
};

template <>
struct void_to_monostate<void> {
  using type = std::monostate;
};

template <typename T>
using void_to_monostate_t = typename void_to_monostate<T>::type;

template <typename T, typename... Ts>
struct unique_tuple_impl : public type_identity<T> {};

template <
  template<typename...> typename Tuple, typename... Ts, typename U, typename... Us
>
struct unique_tuple_impl<Tuple<Ts...>, U, Us...> : 
  std::conditional_t<(std::is_same_v<U, Ts> || ...), 
  unique_tuple_impl<Tuple<Ts...>, Us...>, 
  unique_tuple_impl<Tuple<Ts..., U>, Us...>> {};

template <typename Tuple>
struct unique_tuple;

template <template<typename...>typename Tuple, typename... Ts>
struct unique_tuple<Tuple<Ts...>> : public unique_tuple_impl<Tuple<>, Ts...> {};

template <typename Tuple>
using unique_tuple_t = typename unique_tuple<Tuple>::type;

// Pipeline filter interface

enum class FilterType : int{
  SERIAL = 0,
  PARALLEL
};

template <typename I, typename O, typename C>
struct Filter{

  using input_type  = I;
  using output_type = O;

  Filter(FilterType d, C&& callable) : 
    direction{d}, callable{std::forward<C>(callable)} {
  }

  FilterType direction;
  C callable;
};

template <typename I, typename O, typename C>
auto make_filter(FilterType dir, C&& callable) {
  return Filter<I, O, C>{dir, std::forward<C>(callable)};
}

// Calls your func with tuple element.
template <typename Func, typename Tuple, size_t N = 0>
void visit_tuple(Func func, Tuple& tup, size_t idx) {
  if (N == idx) {
    std::invoke(func, std::get<N>(tup));
    return;
  }
  if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
    return visit_tuple<Func, Tuple, N + 1>(func, tup, idx);
  }
}

template <typename... Fs>
struct Pipeline {

  static_assert(sizeof...(Fs)>0, "must have at least one filter");

  using data_type = unique_tuple_t<
    std::variant<std::monostate, void_to_monostate_t<typename Fs::output_type> ...>
  >;
  
  // TODO: just for hardcoded debugging
  static_assert(std::is_same_v<data_type, std::variant<std::monostate, int, float>>);

  // state constants
  constexpr static int EMPTY   = 0;
  constexpr static int ORPHAN  = 1;
  constexpr static int ADOPTED = 2;

  struct BufferData {
    data_type data;
    std::atomic<int> state {EMPTY};
  };
  
  public:

  /**
  @brief constructs a pipeline object
  */
  Pipeline(size_t L, Fs&&... fs) : 
    _L {L},
    _filters {std::make_tuple(std::forward<Fs>(fs)...)},
    _buffers (L * sizeof...(Fs)),
    _cursors (sizeof...(Fs)) {
    
    for(auto& c : _cursors) {
      c.store(0, std::memory_order_relaxed);
    }
    
    // TODO
    // construct the taskflow graph
  }

  /**
  @brief queries the number of production lines
  */
  size_t num_lines() const {
    return _L;
  }

  /**
  @brief queries the number of filters
  */
  constexpr size_t num_filters() const {
    return sizeof...(Fs);
  }
  


  void make_taskflow(tf::Taskflow& taskflow) {

    // TODO:
    auto init = taskflow.emplace([](){ 
      
      //return std::vector<int> {0, 1, 2, 3, ... L-1};
    });

    for(size_t l=0; l<_L; l++) {
      taskflow.emplace([l, f=0] () mutable {


        
      });
    }


  }

  private:
  
  size_t _L;  // must be power of 2

  std::atomic<size_t> _N{0};

  std::tuple<Fs...> _filters;
  std::vector<BufferData> _buffers;
  std::vector<std::atomic<size_t>> _cursors;




  BufferData& _get_buffer_data(size_t l, size_t f) {
    return _buffers[l*_filters.size() + f];
  }
  
  void _on_filter(size_t f, data_type& d_in, data_type& d_out) {
    
    visit_tuple([&](auto&& filter){

      using F = std::decay_t<decltype(filter)>;
      using I = typename F::input_type;
      using O = typename F::output_type;

      constexpr auto void_i = std::is_void_v<I>;
      constexpr auto void_o = std::is_void_v<O>;

      if constexpr(void_i && void_o) {
        filter.callable();
      }
      else if constexpr(void_i && !void_o) {
        std::get<O>(d_out) = filter.callable();
      }
      else if constexpr(!void_i && void_o) {
        filter.callable(std::get<I>(d_in));
      }
      else {
        std::get<O>(d_out) = filter.callable(std::get<I>(d_in));
      }
    }, _filters, f);
  }

};

//template <typename... Fs>
//auto make_pipeline(Fs&&... filters) {
//
//  using data_type = unique_tuple_t<std::variant<typename Fs::output_type ...>>;
//
//  //static_assert(std::is_same_v<data_type, std::variant<int, float, std::monostate>>);
//}

int main()
{
   static_assert(
     std::is_same_v<
       unique_tuple_t<std::variant<int, double, double, void>>, std::variant<int, double, void>
     >,
     ""
   );


   Pipeline pl(4,
     make_filter<void, int>   (FilterType::SERIAL,  [](){ 
       return 1; 
     }),
     make_filter<int, float> (FilterType::PARALLEL, [](int){ return 1.2f;}),
     make_filter<float, void>(FilterType::SERIAL,   [](float){  }            )
   );

   //pl.make_taskflow(taskflow);

   //std::tuple<std::string, int> tuple {"123", 4};
   //visit_tuple([](auto& filter){
   //  std::cout << filter << std::endl;
   //}, tuple, 0);
}
