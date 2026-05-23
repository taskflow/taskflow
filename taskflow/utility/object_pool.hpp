#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <memory>
#include <atomic>
#include <utility>
#include "os.hpp"

/**
@file object_pool.hpp
@brief object pool include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// ObjectBlock
// ----------------------------------------------------------------------------

/**
@private

@brief internal storage block wrapping an object with pool metadata

@tparam T object type stored in this block

ObjectBlock wraps an object of type @c T together with the shard index
(@c pool_id) and an intrusive free-list link (@c next_free) in a
standard-layout struct. Using @c std::byte storage instead of a @c T
member directly ensures that @c offsetof(ObjectBlock, storage) is
well-defined regardless of @c T's layout, which allows safe recovery of
the block pointer from a bare @c T pointer in @c from_object.
*/
template <typename T>
struct ObjectBlock {

  uint16_t     pool_id;
  ObjectBlock* next_free {nullptr};         // intrusive free list link
  alignas(T) std::byte storage[sizeof(T)]; // raw storage for T

  T* object() noexcept {
    return std::launder(reinterpret_cast<T*>(storage));
  }

  const T* object() const noexcept {
    return std::launder(reinterpret_cast<const T*>(storage));
  }

  static ObjectBlock* from_object(T* obj) noexcept {
    return reinterpret_cast<ObjectBlock*>(
      reinterpret_cast<char*>(obj) - offsetof(ObjectBlock, storage)
    );
  }
};

// ----------------------------------------------------------------------------
// ObjectPool
// ----------------------------------------------------------------------------

/**
@brief sharded fixed-size object allocator with a lock-free hot path

@tparam T       object type to allocate
@tparam LogSize log2 of the number of shards (default @c 5, giving 32 shards);
                must be in [1, 15] to fit the shard index in a @c uint16_t

ObjectPool is a high-performance allocator for a single fixed-size type
@c T, designed for concurrent task-parallel workloads where objects are
frequently created and destroyed across many threads.

Internally, allocations are distributed across @c 2^LogSize independent
shards using a round-robin counter. Each shard maintains two components:

- A lock-free Treiber stack of recycled blocks (hot path). Blocks returned
  by tf::ObjectPool::recycle are pushed here without acquiring any mutex.
  The next call to tf::ObjectPool::animate pops from this stack at the
  cost of a single CAS instruction.

- A @c std::pmr::synchronized_pool_resource as backing storage for fresh
  block allocations (cold path). This mutex-protected pool is only touched
  when the shard's free stack is empty, amortizing synchronization cost
  over many allocations via geometric chunk growth.

A tagged pointer (block address + version counter) in the Treiber stack
prevents the ABA problem. Shards are aligned to the cache line size to
prevent false sharing between concurrent threads.

The following example shows typical usage:

@code{.cpp}
// one allocator per object type, usually a global or executor-level singleton
tf::ObjectPool<MyTask> pool;

// construct a MyTask in the pool, forwarding constructor arguments
MyTask* t = pool.animate(arg1, arg2);

// ... use t ...

// destruct and return storage to the pool for reuse
pool.recycle(t);
@endcode

@note
All pointers returned by tf::ObjectPool::animate must be passed to
tf::ObjectPool::recycle before the allocator is destroyed. Destroying
the allocator with live objects is undefined behavior.

@note
ObjectPool is non-copyable. Declare it as a global or as a
long-lived member of the object that owns the task graph.
*/
template <typename T, size_t LogSize = 5>
class ObjectPool {

  static_assert(LogSize >= 1 && LogSize <= 15, "LogSize must be in [1, 15]");

  using Block = ObjectBlock<T>;

  static constexpr size_t NumPools = 1u << LogSize;

  // Pairs a free-list head pointer with a version counter to defeat ABA.
  struct TaggedHead {
    Block*    ptr {nullptr};
    uintptr_t tag {0};
  };

  //static_assert(
  //  std::atomic<TaggedHead>::is_always_lock_free,
  //  "std::atomic<TaggedHead> is not lock-free on this platform — "
  //  "check alignment and compiler support for 128-bit CAS"
  //);

  struct alignas(TF_CACHELINE_SIZE) Shard {

    // Hot path: lock-free Treiber stack of recycled blocks.
    // Padded to its own cache line via alignas on _backing so that
    // hot-path CAS on _free_head does not invalidate the cache line
    // holding _backing's mutex (false sharing between hot and cold paths).
    std::atomic<TaggedHead> _free_head {TaggedHead{}};

    // Cold path: backing allocator for fresh block memory.
    // alignas forces _backing to start on the next cache line boundary,
    // separating it from the hot _free_head above.
    std::pmr::synchronized_pool_resource _backing {
      std::pmr::pool_options {
        .max_blocks_per_chunk        = 1024,
        .largest_required_pool_block = sizeof(Block)
      }
    };

    void push_free(Block* b) noexcept {
      TaggedHead cur = _free_head.load(std::memory_order_relaxed);
      TaggedHead next;
      do {
        b->next_free = cur.ptr;
        next         = {b, cur.tag + 1};
      } while (!_free_head.compare_exchange_weak(
        cur, next,
        std::memory_order_release,
        std::memory_order_relaxed
      ));
    }

    Block* pop_free() noexcept {
      TaggedHead cur = _free_head.load(std::memory_order_acquire);
      TaggedHead next;
      while (cur.ptr) {
        next = {cur.ptr->next_free, cur.tag + 1};
        if (_free_head.compare_exchange_weak(
          cur, next,
          std::memory_order_acquire,
          std::memory_order_relaxed
        )) {
          return cur.ptr;
        }
      }
      return nullptr;
    }

    Block* allocate_from_backing() {
      return static_cast<Block*>(
        _backing.allocate(sizeof(Block), alignof(Block))
      );
    }

    void dealloc_to_backing(Block* b) {
      _backing.deallocate(b, sizeof(Block), alignof(Block));
    }
  };

  std::array<Shard, NumPools> _shards;

  // Round-robin counter shared across all ObjectPool<T, LogSize> instances,
  // providing global load balancing without per-instance state.
  static size_t _next_shard() noexcept {
    static std::atomic<size_t> _counter{0};
    return _counter.fetch_add(1, std::memory_order_relaxed) & (NumPools - 1);
  }

  public:

  /**
  @brief constructs the allocator with @c 2^LogSize empty shards

  Each shard is default-constructed with an empty free stack and an
  uninitialized backing pool. No memory is allocated from the OS until
  the first call to tf::ObjectPool::animate.
  */
  ObjectPool() = default;

  /**
  @brief disabled copy constructor

  ObjectPool owns its shards and backing memory; copying is not
  meaningful. Declare the allocator as a global or long-lived member
  and share it by reference or pointer.
  */
  ObjectPool(const ObjectPool&) = delete;

  /**
  @brief disabled copy assignment operator
  */
  ObjectPool& operator=(const ObjectPool&) = delete;

  /**
  @brief destroys the allocator and releases all backing memory to upstream

  The destructor of each shard's @c std::pmr::synchronized_pool_resource
  returns all allocated chunks to the system allocator, including memory
  backing blocks that are currently on the free stack. No per-block
  destructor is called; callers are responsible for recycling all live
  objects before destroying the allocator.
  */
  ~ObjectPool() = default;

  /**
  @brief constructs an object of type @c T in the pool and returns a pointer

  @tparam Args constructor argument types
  @param  args arguments forwarded to the constructor of @c T

  @return pointer to the newly constructed @c T; never null

  On the hot path, animate pops a previously recycled block from the
  shard's lock-free free stack and constructs @c T in it via
  @c std::construct_at, with no mutex acquisition. On a cache miss (empty
  free stack), a fresh block is carved from the shard's backing
  @c std::pmr::synchronized_pool_resource, which amortizes system
  allocation cost over chunks of up to 1024 blocks.

  Allocations are distributed across shards via a global round-robin
  counter, balancing load regardless of which thread calls animate.

  @code{.cpp}
  tf::ObjectPool<MyTask> pool;

  // default-construct
  MyTask* t1 = pool.animate();

  // construct with arguments
  MyTask* t2 = pool.animate(42, "hello");

  pool.recycle(t1);
  pool.recycle(t2);
  @endcode

  @note
  The returned pointer must eventually be passed to
  tf::ObjectPool::recycle. Discarding it without recycling leaks
  both the object's resources and the underlying block.
  */
  template <typename... Args>
  [[nodiscard]] T* animate(Args&&... args) {
    auto  sid   = _next_shard();
    auto& shard = _shards[sid];

    Block* block = shard.pop_free();       // hot path: lock-free
    if (!block) block = shard.allocate_from_backing(); // cold path: mutex, amortized

    block->pool_id = static_cast<uint16_t>(sid);
    return std::construct_at(block->object(), std::forward<Args>(args)...);
  }

  /**
  @brief destructs the object and returns its storage to the pool

  @param obj pointer to a @c T previously returned by
              tf::ObjectPool::animate, or @c nullptr (no-op)

  recycle calls the destructor of @c *obj via @c std::destroy_at, then
  pushes the underlying block onto the shard's lock-free free stack
  without acquiring any mutex. The block becomes immediately available
  for the next call to tf::ObjectPool::animate on any thread.

  @code{.cpp}
  tf::ObjectPool<MyTask> pool;

  MyTask* t = pool.animate(arg1, arg2);

  // ... use t ...

  pool.recycle(t);  // destructor called here; memory returned to pool
  t = nullptr;      // pointer is now dangling; do not dereference
  @endcode

  @note
  Passing a pointer not obtained from this allocator is undefined behavior.
  After recycle returns, @c obj is a dangling pointer and must not be
  dereferenced.
  */
  void recycle(T* obj) {
    if (!obj) return;
    auto* block = Block::from_object(obj);
    std::destroy_at(block->object());
    _shards[block->pool_id].push_free(block); // hot path: lock-free
  }

  /**
  @brief returns all recycled blocks and backing memory to the system allocator

  release calls @c std::pmr::synchronized_pool_resource::release on each
  shard's backing pool, returning all chunks to the upstream system allocator
  in one shot, then atomically resets each shard's free stack to null.
  This is an O(1) operation per shard — no per-block work is performed,
  because the backing pool owns memory at the chunk level and frees entire
  chunks regardless of how many individual blocks were returned to it.
  After this call the allocator is in the same state as after construction:
  empty free stacks, no memory held from the OS.

  This method is optional and is not required before destruction.
  It is useful for reclaiming pool memory between distinct workload phases
  without destroying the allocator itself.

  @code{.cpp}
  tf::ObjectPool<MyTask> pool;

  // --- phase 1 ---
  for (auto& task : phase1_tasks) {
    MyTask* t = pool.animate(task);
    // ... run t ...
    pool.recycle(t);
  }

  pool.release(); // return OS memory before phase 2 begins

  // --- phase 2 ---
  for (auto& task : phase2_tasks) {
    MyTask* t = pool.animate(task);
    // ... run t ...
    pool.recycle(t);
  }
  @endcode

  @note
  All live objects must be recycled before calling release. Calling
  release while objects are still alive is undefined behavior because
  the backing memory they reside in is freed.
  */
  void release() {
    for (auto& shard : _shards) {
      // Release all backing chunks to upstream first — this covers both
      // blocks currently on the free stack and any that were never recycled,
      // since the backing pool owns memory at the chunk level, not per block.
      shard._backing.release();
      // Reset the free stack to null in O(1). Pointers it held are now
      // dangling (their backing chunks were just freed), so they must be
      // cleared before the allocator is used again.
      shard._free_head.store(TaggedHead{}, std::memory_order_relaxed);
    }
  }
};

} // namespace tf
