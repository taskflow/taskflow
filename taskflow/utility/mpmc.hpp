#pragma once

#include <cassert>
#include <atomic>
#include <optional>

namespace tf {

/**
 * A 'lockless' bounded multi-producer, multi-consumer queue
 *
 * Has the caveat that the queue can *appear* empty even if there are
 * returned items within it as a single thread can block progression
 * of the queue.
 */
template<typename T, size_t LogSize = 10>
class MPMC {

  constexpr static uint64_t BufferSize = 1ull << LogSize;
  constexpr static uint64_t BufferMask = (BufferSize - 1);
  
  static_assert((BufferSize >= 2) && ((BufferSize & (BufferSize - 1)) == 0));

public:

  /**
   * Constructs a bounded multi-producer, multi-consumer queue
   *
   * Note: Due to the algorithm used, buffer_size must be a power
   *       of two and must be greater than or equal to two.
   *
   * @param buffer_size Number of spaces available in the queue.
   */
  explicit MPMC() {
    for (size_t i = 0; i < _buffer.size(); i++) {
      _buffer[i].sequence.store(i, std::memory_order_relaxed);
    }
    _enqueue_pos.store(0, std::memory_order_relaxed);
    _dequeue_pos.store(0, std::memory_order_relaxed);
  }


  /**
   * Enqueues an item into the queue
   *
   * @param data Argument to place into the array
   * @return false if the queue was full (and enqueing failed),
   *         true otherwise
   */
  bool try_enqueue(T data) {
    Cell *cell;
    auto pos = _enqueue_pos.load(std::memory_order_relaxed);
    for (; ;) {
      cell = &_buffer[pos & BufferMask];
      auto seq = cell->sequence.load(std::memory_order_acquire);
      if (seq == pos) {
        if (_enqueue_pos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
            break;
        }
      } else if (seq < pos) {
          return false;
      } else {
          pos = _enqueue_pos.load(std::memory_order_relaxed);
      }
    }

    cell->data = data;
    cell->sequence.store(pos + 1, std::memory_order_release);

    return true;
  }
  
  void enqueue(T data) {

    Cell *cell;
    auto pos = _enqueue_pos.load(std::memory_order_relaxed);

    for (; ;) {
      cell = &_buffer[pos & BufferMask];
      auto seq = cell->sequence.load(std::memory_order_acquire);
      if (seq == pos) {
        if (_enqueue_pos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
            break;
        }
      }
      else {
        pos = _enqueue_pos.load(std::memory_order_relaxed);
      }
    }

    cell->data = data;
    cell->sequence.store(pos + 1, std::memory_order_release);
  }

  /**
   * Dequeues an item from the queue
   *
   * @param[out] data Reference to place item into
   * @return false if the queue was empty (and dequeuing failed),
   *         true if successful
   */
  std::optional<T> try_dequeue() {
    Cell *cell;
    auto pos = _dequeue_pos.load(std::memory_order_relaxed);
    for (; ;) {
      cell = &_buffer[pos & BufferMask];
      auto seq = cell->sequence.load(std::memory_order_acquire);
      if (seq == pos + 1) {
        if (_dequeue_pos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
          break;
        }
      } else if (seq < (pos + 1)) {
        return std::nullopt;
      } else {
        pos = _dequeue_pos.load(std::memory_order_relaxed);
      }
    }

    T data = cell->data;
    cell->sequence.store(pos + BufferMask + 1, std::memory_order_release);

    return data;
  }

  bool empty() const {
    auto beg = _dequeue_pos.load(std::memory_order_relaxed);
    auto end = _enqueue_pos.load(std::memory_order_relaxed);
    return beg >= end;
  }

  size_t capacity() const {
    return BufferSize;
  }

private:

  struct Cell {
    T data;
    std::atomic<uint64_t> sequence;
  };

  //static const size_t cacheline_size = 64;

  alignas(2*TF_CACHELINE_SIZE) std::array<Cell, BufferSize> _buffer;
  alignas(2*TF_CACHELINE_SIZE) std::atomic<uint64_t> _enqueue_pos;
  alignas(2*TF_CACHELINE_SIZE) std::atomic<uint64_t> _dequeue_pos;
};

}  // end of namespace tf -----------------------------------------------------
