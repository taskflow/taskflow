  #pragma once

  /**
   * @file task_wrapper.hpp
   * @brief task wrapper include file
   * 
   */

  namespace tf {

  /**
   * @brief performs an identity wrapping of a function object
   * 
   */
  struct TaskWrapperIdent {
    template <typename Task>
    void operator()(Task&& task) const { std::forward<Task>(task)(); }
  };

  }