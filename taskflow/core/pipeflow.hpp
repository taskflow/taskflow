#pragma once
namespace tf {
/*Pipeflow can only be created privately by the tf::Pipeline and
be used through the pipe callable.
*/
class Pipeflow {

  template <typename... Ps>
  friend class Pipeline;
  
  template <typename P>
  friend class ScalablePipeline;

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
  Others have no effect.
  */
  void stop() {
    _stop = true;
  }
  
  Pipeflow(size_t line, size_t pipe) :
    _line {line},
    _pipe {pipe} {
  }
  private:

  size_t _line;
  size_t _pipe;
  size_t _token;
  bool   _stop;
};

}