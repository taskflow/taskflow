#pragma once

namespace tf {

enum class FILTER_TYPE {
  PARALLEL,
  SERIAL
};

////template <typename F>
//class FilterBase {
//
//  public:
//    // f = [](int) { return std::string;}
//    //Filter(FILTER_TYPE t, F&& f) {
//    //}
//
//    virtual void operator() (void*);
//
//  private:
//
//};

//class Filter : public FilterBase {
//
//
//
//  void operator() (void*) {
//
//  }
//
//}

template <typename ... Fs>
class Pipeflow {

  public:

    Pipeflow(size_t max_token, Fs&&... filters) {
      // TODO
    }

  private:

    tf::Taskflow _taskflow;

    //std::array<Filter, sizeof...(Fs)> _filters;
};












}
