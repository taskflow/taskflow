#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace tf {
  template <typename InputIt, typename OutputIt>
  OutputIt inclusive_scan(InputIt first, InputIt last, OutputIt dest) {
    if (first == last) {
    *dest = *first;
    return dest;
  }
  
  auto scannedSum = *first;
  *dest = scannedSum;
  tf::Taskflow taskflow;
    tf::Executor executor;
    int next = 1;
    int endVal = static_cast<int> (std::distance(first, last));
    auto pf = taskflow.for_each_index(std::ref(next), std::ref(endVal), 1, 
      [&dest, &first, &scannedSum] (int i) {
        auto currObj  = *(first+i);
        scannedSum += currObj;
        *(dest+i) = scannedSum;
      }
    ); 

    executor.run(taskflow).get(); 

    return dest;
  }

  template <typename InputIt, typename OutputIt, typename T>
  OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt dest, T initialVal) {
    if (first == last) {
      *dest = initialVal;
      return dest;
    }

    tf::Taskflow taskflow;
    tf::Executor executor;

    auto scannedSum = initialVal;
    const int initial = 0;
    const int endVal = (static_cast<int> (std::distance(first, last)));
    auto pf = taskflow.for_each_index(std::ref(initial), std::ref(endVal), 1, 
      [&dest, &first, &scannedSum] (int i) {
        *(dest+i) = scannedSum;
        auto currObj  = *(first+i);
        scannedSum += currObj;
      }
    ); 

    executor.run(taskflow).get(); 

    return dest;
  }
}