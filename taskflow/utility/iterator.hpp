#pragma once

#include <cstddef>
#include <type_traits>

namespace tf {

/**
 * @brief checks if the given index range is invalid
 *
 * @tparam B type of the beginning index
 * @tparam E type of the ending index
 * @tparam S type of the step size
 *
 * @param beg starting index of the range
 * @param end ending index of the range
 * @param step step size to traverse the range
 *
 * @return returns @c true if the range is invalid; @c false otherwise.
 *
 * A range is considered invalid under the following conditions:
 *  + The step is zero and the begin and end values are not equal.
 *  + A positive range (begin < end) with a non-positive step.
 *  + A negative range (begin > end) with a non-negative step.
 */
template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral_v<std::decay_t<B>> && 
                           std::is_integral_v<std::decay_t<E>> && 
                           std::is_integral_v<std::decay_t<S>>, bool>
is_index_range_invalid(B beg, E end, S step) {
  return ((step == 0 && beg != end) ||
          (beg < end && step <=  0) ||  // positive range
          (beg > end && step >=  0));   // negative range
}

/**
 * @brief calculates the number of iterations in the given index range
 *
 * @tparam B type of the beginning index
 * @tparam E type of the ending index
 * @tparam S type of the step size
 *
 * @param beg starting index of the range
 * @param end ending index of the range
 * @param step step size to traverse the range
 *
 * @return returns the number of required iterations to traverse the range
 *
 * The distance of a range represents the number of required iterations to traverse the range
 * from the beginning index to the ending index (exclusive) with the given step size.
 * 
 * Example 1:
 * @code{.cpp}
 * // Range: 0 to 10 with step size 2
 * size_t dist = distance(0, 10, 2);  // Returns 5, the sequence is [0, 2, 4, 6, 8]
 * @endcode
 *
 * Example 2:
 * @code{.cpp}
 * // Range: 10 to 0 with step size -2
 * size_t dist = distance(10, 0, -2);  // Returns 5, the sequence is [10, 8, 6, 4, 2]
 * @endcode
 *
 * Example 3:
 * @code{.cpp}
 * // Range: 5 to 20 with step size 5
 * size_t dist = distance(5, 20, 5);  // Returns 3, the sequence is [5, 10, 15]
 * @endcode
 *
 * @attention
 * It is user's responsibility to ensure the given index range is valid.
 */
template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral_v<std::decay_t<B>> && 
                           std::is_integral_v<std::decay_t<E>> && 
                           std::is_integral_v<std::decay_t<S>>, size_t>
distance(B beg, E end, S step) {
  return (end - beg + step + (step > 0 ? -1 : 1)) / step;
}

/**
 * @class IndexRange
 *
 * @brief class to create an index range of integral indices with a step size
 *
 * This class provides functionality for managing a range of indices, where the range 
 * is defined by a starting index, an ending index, and a step size. The indices must 
 * be of an integral type.
 * For example, the range [0, 10) with a step size 2 represents the five elements,
 * 0, 2, 4, 6, and 8.
 *
 * @tparam T the integral type of the indices
 *
 * @attention
 * It is user's responsibility to ensure the given range is valid.
 */
template <typename T>
class IndexRange {

  static_assert(std::is_integral_v<T>, "index type must be integral");

public:

  /**
  @brief alias for the index type used in the range
  */
  using index_type = T;

  /**
  @brief constructs an index range object without any initialization
  */
  IndexRange() = default;

  /**
   * @brief constructs an IndexRange object
   * @param beg starting index of the range
   * @param end ending index of the range (exclusive)
   * @param step_size step size between consecutive indices in the range
   */
  explicit IndexRange(T beg, T end, T step_size)
    : _beg{beg}, _end{end}, _step_size{step_size} {}

  /**
   * @brief queries the starting index of the range
   */
  T begin() const { return _beg; }

  /**
   * @brief queries the ending index of the range
   */
  T end() const { return _end; }

  /**
   * @brief queries the step size of the range
   */
  T step_size() const { return _step_size; }

  /**
   * @brief updates the range with the new starting index, ending index, and step size
   */
  IndexRange<T>& reset(T begin, T end, T step_size) {
    _beg = begin;
    _end = end;
    _step_size = step_size;
    return *this;
  }

  /**
   * @brief updates the starting index of the range
   */
  IndexRange<T>& begin(T new_begin) { _beg = new_begin; return *this; }

  /**
   * @brief updates the ending index of the range
   */
  IndexRange<T>& end(T new_end) { _end = new_end; return *this; }

  /**
   * @brief updates the step size of the range
   */
  IndexRange<T>& step_size(T new_step_size) { _step_size = new_step_size; return *this; }

  /**
   * @brief queries the number of elements in the range
   *
   * The number of elements is equivalent to the number of iterations in the range.
   * For instance, the range [0, 10) with step size of 2 will iterate five elements,
   * 0, 2, 4, 6, and 8.
   */
  size_t size() const { return distance(_beg, _end, _step_size); }

  /**
   * @brief returns a range from the given discrete domain
   * @param part_beg starting index of the discrete domain
   * @param part_end ending index of the discrete domain
   * @return a new IndexRange object representing the given discrete domain
   * 
   * The discrete domain of a range refers to a counter-based sequence indexed from 0
   * to @c N, where @c N is the size (i.e., number of iterated elements) of the range. 
   * For example, a discrete domain of the range [0, 10) with a step size of 2 corresponds 
   * to the sequence 0, 1, 2, 3, and 4, which map to the range elements 0, 2, 4, 6, and 8.
   *
   * For a partitioned domain [@c part_beg, @c part_end), this function returns
   * the corresponding range. For instance, the partitioned domain [2, 5) for the
   * above example returns the range [4, 10) with the same step size of 2.
   *
   * @attention
   * Users must ensure the specified domain is valid with respect to the range.
   */
  IndexRange discrete_domain(size_t part_beg, size_t part_end) const {
    return IndexRange(
      static_cast<T>(part_beg) * _step_size + _beg,
      static_cast<T>(part_end) * _step_size + _beg,
      _step_size
    );
  }

  private:

  T _beg;
  T _end;
  T _step_size;

};

  

}  // end of namespace tf -----------------------------------------------------
