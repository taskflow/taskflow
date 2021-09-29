#pragma once

#include <any>

namespace tf {

enum class FILTER_TYPE {
  PARALLEL,
  SERIAL
};

// from library developers' perspective
class FilterBase {

  public:

    FilterBase() = default; 

    virtual ~FilterBase() = default;

  private:

    virtual std::any operator() (std::any& d) = 0;

};

// from users' perspectives
template <typename I, typename O, typename C>
class Filter : public FilterBase {

  friend class Pipeline;

  public:
    
    // tf::P => 0
    // tf::S => 1
    Filter(int dir, C&& callable) : _filter {std::forward<C>(callable)} {

    }

  private:

  std::any operator () (std::any& d) override final {
    //O output = callable( *static_cast<I*>(data));
    return _filter(std::any_cast<I>(d));
  }

  int _dir;
  C _filter;

};

class FilterBuffer {

  friend class Pipeline;

  private:

    std::atomic<int> _state;
    std::vector<std::any> _data;
};

// scalable pipeline
// 
// # 1
// Pipeline a = taskflow.pipeline(M,  // or auto a = ...
//   Filter<void, int>         {tf::P, []( tf::FC& )     { return 1 }     },
//   Filter<int, std::string>  {tf::S, []( int in)       { return "hi"; } },
//   Filter<std::string, void> {tf::P, [](std::string a) {}               }
// );
//
// # 2
// Pipeline b(M,                      // or auto b = ...
//   Filter<void, int>         {tf::P, []( tf::FC& )     { return 1 }     },
//   Filter<int, std::string>  {tf::S, []( int in)       { return "hi"; } },
//   Filter<std::string, void> {tf::P, [](std::string a) {}               }
// );
// taskflow.pipeline(b);
// 
class Pipeline {

  public:
    
    //template <typename... Fs>
    //Pipeline(size_t M, Fs&&... filters) :
    //  _filters { std::forward<Fs>(filters)... },
    //  _buffers {sizeof...(Fs)},
    //{
    //  for(auto& b : _buffers) {
    //    b._data.resize(M);
    //  }
    //}

  private:

    //std::array<Filter, sizeof...(Fs)> _filters;
    std::vector<FilterBase> _filters;
    std::vector<FilterBuffer> _buffers;
};




}  // end of namespace tf -----------------------------------------------------
