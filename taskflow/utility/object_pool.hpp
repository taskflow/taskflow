#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <memory>
#include <atomic>
#include <thread>
#include <utility>
#include "os.hpp"

/**
@file object_pool.hpp
@brief object pool include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// TaggedHead128
// ----------------------------------------------------------------------------

/**
@brief tagged free-list head using a 128-bit (pointer, version) pair

TaggedHead128 stores the free-list head pointer and an ABA version counter
as two independent 64-bit words, yielding a 16-byte representation.
@c std::atomic<TaggedHead128> requires a 128-bit compare-and-swap:
CMPXCHG16B on x86-64 or CASP on ARMv8.1+. On platforms that lack a native
128-bit CAS (e.g. RISC-V, baseline ARMv8.0), the atomic falls back to a
lock-based implementation — still correct, but no longer lock-free.

The full 64-bit version counter makes ABA wrap-around effectively impossible
under any realistic workload.

@code{.cpp}
// explicitly select the 128-bit head (this is also the default)
tf::ObjectPool<MyTask, 5, tf::TaggedHead128> pool;
@endcode

@see TaggedHead64 for an 8-byte alternative that is lock-free on all 64-bit
     platforms at the cost of a narrower version counter.
*/
struct TaggedHead128 {

  /**
  @brief block address representation
  */
  using pointer_type = uintptr_t;

  /**
  @brief ABA version counter representation
  */
  using tag_type = uintptr_t;

  /**
  @brief block address (reinterpret-cast to/from @c ObjectBlock*)
  */
  pointer_type ptr {0};

  /**
  @brief ABA version counter; incremented on every push and pop
  */
  tag_type tag {0};

  /**
  @brief constructs a null, zero-tagged head
  */
  TaggedHead128() = default;

  /**
  @brief constructs a head with an explicit block address and version counter

  @param p block address as @c uintptr_t
  @param t ABA version counter

  @code{.cpp}
  tf::TaggedHead128 head(reinterpret_cast<uintptr_t>(block), 0);
  @endcode
  */
  TaggedHead128(pointer_type p, tag_type t) noexcept : ptr{p}, tag{t} {}

  /**
  @brief returns the stored block address

  @code{.cpp}
  tf::TaggedHead128 head(reinterpret_cast<uintptr_t>(block), 7);
  auto p = head.get_ptr();  // p == reinterpret_cast<uintptr_t>(block)
  @endcode
  */
  pointer_type get_ptr() const noexcept { return ptr; }

  /**
  @brief returns the ABA version counter

  @code{.cpp}
  tf::TaggedHead128 head(reinterpret_cast<uintptr_t>(block), 7);
  auto t = head.get_tag();  // t == 7
  @endcode
  */
  tag_type get_tag() const noexcept { return tag; }
};

// ----------------------------------------------------------------------------
// TaggedHead64
// ----------------------------------------------------------------------------

/**
@brief tagged free-list head packed into a single 64-bit word

@tparam PtrBits number of low bits reserved for the block address; the
                remaining @c (64 - PtrBits) bits hold the ABA version counter.
                Must be at most 48 so that at least 16 tag bits are available.
                Defaults to @c TF_POINTER_BITS (48 on x86-64, AArch64, and
                RISC-V SV48). Override for non-standard VA layouts:
                @li @c 39 for RISC-V SV39, giving 25 tag bits
                @li Values above 48 (e.g. LA57's 57) are rejected by a
                @c static_assert — use TaggedHead128 in those cases

TaggedHead64 fits both the free-list head pointer and an ABA version counter
into a single @c uintptr_t: the low @c PtrBits bits store the block address
and the remaining high bits store the version counter. Because the entire
state is one 64-bit word, @c std::atomic<TaggedHead64<>> is always lock-free
on every 64-bit platform — including those without a native 128-bit CAS
instruction (RISC-V, baseline ARMv8.0).

The default @c PtrBits of 48 assumes user-space virtual addresses occupy at
most 48 bits, which holds for:
@li x86-64 with 4-level paging — bits 48–63 are zero in user space
@li AArch64 with 48-bit VA space — TTBR0 range
@li RISC-V SV39 / SV48 — provides even more headroom (25 / 16 free bits)

The resulting @c (64 - PtrBits) -bit version counter wraps at
@c 2^(64-PtrBits). With the default of 48, the counter is 16 bits and wraps
at 65 536. ABA would require a competing thread to complete exactly that many
push/pop cycles on the same shard between another thread's load and its CAS —
negligible probability in any task-parallel workload.

@code{.cpp}
// default PtrBits (48) — correct for x86-64, AArch64, RISC-V SV48
tf::ObjectPool<MyTask, 5, tf::TaggedHead64<>> pool;

// RISC-V SV39: only 39 address bits, 25 bits available for the tag
tf::ObjectPool<MyTask, 5, tf::TaggedHead64<39>> pool_sv39;
@endcode

@see TaggedHead128 for a variant with a full 64-bit version counter that may
     use a lock-based fallback on platforms without 128-bit CAS.
*/
template <int PtrBits = TF_POINTER_BITS>
struct TaggedHead64 {
  static_assert(64 - PtrBits >= 16,
    "TaggedHead64 requires at least 16 tag bits for ABA safety "
    "(PtrBits must be <= 48); use TaggedHead128 instead");

  /**
  @brief block address representation
  */
  using pointer_type = uintptr_t;

  /**
  @brief ABA version counter representation
  */
  using tag_type = uint16_t;

  /**
  @brief bits reserved for the block address
  */
  static constexpr int PTR_BITS = PtrBits;

  /**
  @brief bits reserved for the version counter
  */
  static constexpr int TAG_BITS = 64 - PtrBits;

  /**
  @brief mask isolating the address field
  */
  static constexpr pointer_type PTR_MASK = (pointer_type{1} << PTR_BITS) - 1;

  /**
  @brief packed word: high @c TAG_BITS bits hold the version tag, low
         @c PTR_BITS bits hold the address
  */
  uintptr_t bits {0};

  /**
  @brief constructs a null, zero-tagged head
  */
  TaggedHead64() = default;

  /**
  @brief constructs a head with an explicit block address and version counter

  @param p block address as @c uintptr_t; only the low @c PtrBits bits are stored
  @param t ABA version counter; only the low @c TAG_BITS bits are stored

  @code{.cpp}
  tf::TaggedHead64<> head(reinterpret_cast<uintptr_t>(block), 0);
  @endcode
  */
  TaggedHead64(pointer_type p, tag_type t) noexcept
    : bits{ (p & PTR_MASK) | (static_cast<uintptr_t>(t) << PTR_BITS) } {}

  /**
  @brief returns the block address

  @code{.cpp}
  tf::TaggedHead64<> head(reinterpret_cast<uintptr_t>(block), 7);
  auto p = head.get_ptr();  // p == reinterpret_cast<uintptr_t>(block)
  @endcode
  */
  pointer_type get_ptr() const noexcept { return bits & PTR_MASK; }

  /**
  @brief returns the 16-bit ABA version counter

  @code{.cpp}
  tf::TaggedHead64<> head(reinterpret_cast<uintptr_t>(block), 7);
  auto t = head.get_tag();  // t == 7
  @endcode
  */
  tag_type get_tag() const noexcept { return static_cast<tag_type>(bits >> PTR_BITS); }
};

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

@c next_free is @c std::atomic<ObjectBlock*> rather than a plain pointer.
C++20 [atomics.types.generic.general] guarantees that @c std::atomic<T> is
standard-layout, so ObjectBlock remains standard-layout and @c offsetof
stays well-defined. See the @c next_free field declaration below for the
race scenario that motivates the atomic type.
*/
template <typename T>
struct ObjectBlock {

  /**
  @brief index of the shard that owns this block
  */
  uint16_t pool_id;

  // Intrusive free-list link. Must be atomic to avoid a formal data race
  // between push_free writing next_free and a concurrent pop_free reading it.
  //
  // The race arises from a *stale pointer* held by a thread that loses a CAS.
  // Consider this interleaving (free list: [b -> null]):
  //
  //   Thread A (pop_free):  loads _free_head = {b, 5}  <-- cur.ptr = b
  //   Thread B (pop_free):  also loads {b, 5}, wins CAS first,
  //                         pops b, returns it to the caller
  //   Thread C (push_free): caller recycles b;
  //                         C writes b->next_free       <-- non-atomic WRITE
  //   Thread A (pop_free):  reads cur.ptr->next_free    <-- non-atomic READ
  //                         (cur.ptr is the stale b!)
  //
  // Thread A's CAS will ultimately fail (the version tag changed), so there
  // is no algorithmic corruption — but the concurrent non-atomic read + write
  // on the same memory location is a formal data race and undefined behavior
  // per the C++ memory model. Making next_free atomic eliminates the race by
  // definition: two concurrent atomic accesses are never a data race.
  std::atomic<ObjectBlock*> next_free {nullptr};

  /**
  @brief raw storage for one @c T
  */
  alignas(T) std::byte storage[sizeof(T)];

  /**
  @brief returns a pointer to the stored object
  */
  T* object() noexcept {
    return std::launder(reinterpret_cast<T*>(storage));
  }

  /**
  @brief returns a const pointer to the stored object
  */
  const T* object() const noexcept {
    return std::launder(reinterpret_cast<const T*>(storage));
  }

  /**
  @brief recovers the enclosing ObjectBlock from a bare object pointer

  Uses @c offsetof(ObjectBlock, storage), which is well-defined because
  ObjectBlock is standard-layout (guaranteed by C++20 for @c std::atomic<T>
  member types).
  */
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
@tparam H       tagged-pointer policy that controls the free-stack head
                representation. The type must provide:
                @li @c pointer_type — typedef for the address representation
                @li @c tag_type     — typedef for the ABA counter representation
                @li @c get_ptr()    — returns the stored address as @c pointer_type
                @li @c get_tag()    — returns the ABA counter as @c tag_type
                @li @c H(pointer_type, tag_type) — two-argument constructor
                Built-in choices:
                @li tf::TaggedHead128 (default) — 16-byte head, 64-bit counter;
                    lock-free on x86-64 and ARMv8.1+, mutex-based elsewhere.
                @li tf::TaggedHead64<>          — 8-byte head, 16-bit counter;
                    lock-free on all 64-bit platforms; requires pointers to
                    fit in 48 bits.
@tparam LogSize log2 of the number of shards (default @c 5, giving 32 shards);
                must be in [1, 15] to fit the shard index in a @c uint16_t

%ObjectPool is a high-performance allocator for a single fixed-size type @c T,
designed for concurrent task-parallel workloads where objects are frequently
created and destroyed across many threads.

@dotfile images/object_pool_flow.dot

Internally, allocations are distributed across @c 2^LogSize independent
shards. Each shard maintains two independent components (separated by cache
lines to prevent false sharing):

<b>Hot Path (99% of operations):</b> A lock-free Treiber stack of recycled blocks.
When tf::ObjectPool::animate is called, it tries to pop a recycled block from
this stack with a single atomic CAS. On success, the block is reused with
no mutex acquisition. Blocks returned by tf::ObjectPool::recycle are pushed
back onto this stack without acquiring any mutex.

<b>Cold Path (1% of operations):</b> A @c std::pmr::synchronized_pool_resource
as backing storage for fresh block allocations. This mutex-protected pool is
only touched when the shard's hot-path stack is empty. When accessed, it
allocates a whole <b>chunk</b> (configured to hold up to 1024 blocks via
@c max_blocks_per_chunk = 1024), amortizing the synchronization cost: one
mutex acquisition yields ~1024 blocks for the hot path.

The tagged-pointer policy @c H attaches a version counter to each free-stack
head to prevent the ABA problem. This counter increments on every push and pop,
making ABA wrap-around effectively impossible under realistic workloads.
Shards are aligned to the cache line size to eliminate false sharing between
concurrent threads accessing different shards' hot-path stacks.

@code{.cpp}
// default: TaggedHead128, 32 shards
tf::ObjectPool<MyTask> pool;

// lock-free on all 64-bit platforms (default PtrBits=48), 32 shards
tf::ObjectPool<MyTask, 5, tf::TaggedHead64<>> pool64;

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

@par Two-Level Freelist Design

The combination of lock-free and mutex-protected freelists is deliberate:
recycled blocks remain on the lock-free stack indefinitely, avoiding mutex
costs on every allocation. The backing pool's internal freelist is rarely
used directly because blocks do <b>not</b> call @c deallocate() in the normal
hot path — they stay on the lock-free stack for immediate reuse. This design
trades chunk-level memory reuse efficiency for atomic-fast allocation on the
hot path, which is the right trade-off for task-parallel workloads where the
hot path is hit millions of times.

@par Chunk Amortization

When the hot-path stack is empty, a single @c std::pmr::synchronized_pool_resource::allocate
call acquires a mutex and either reuses a chunk or allocates a new one from
the system allocator. With @c max_blocks_per_chunk = 1024, one mutex acquisition
amortizes to ~1024 subsequent lock-free pops, yielding negligible mutex overhead
(roughly 0.001 mutex cost per allocation).

*/
template <typename T, typename H = TaggedHead128, size_t LogSize = 5>
class ObjectPool {

  static_assert(LogSize >= 1 && LogSize <= 15,
    "LogSize must be in [1, 15]");

  using Block = ObjectBlock<T>;

  static constexpr size_t NumPools = 1u << LogSize;

  struct alignas(TF_CACHELINE_SIZE) Shard {

    // Hot path: lock-free Treiber stack of recycled blocks.
    // _free_head sits on its own cache line (via the Shard alignas) so that
    // hot-path CAS does not invalidate the line holding _backing's mutex.
    std::atomic<H> _free_head {H{}};

    // Cold path: backing allocator for fresh block memory.
    // alignas pushes _backing to the next cache line, separating it from the
    // hot _free_head above and preventing hot/cold false sharing.
    alignas(TF_CACHELINE_SIZE) std::pmr::synchronized_pool_resource _backing {
      std::pmr::pool_options {
        .max_blocks_per_chunk        = 1024,
        .largest_required_pool_block = sizeof(Block)
      }
    };

    void push_free(Block* b) noexcept {
      H cur = _free_head.load(std::memory_order_relaxed);
      H next;
      do {
        // relaxed: the release CAS below synchronises-with pop_free's acquire,
        // making this store visible to any thread that subsequently observes b
        // at the head of the list.
        b->next_free.store(
          reinterpret_cast<Block*>(cur.get_ptr()), std::memory_order_relaxed);
        next = H(
          reinterpret_cast<typename H::pointer_type>(b),
          static_cast<typename H::tag_type>(cur.get_tag() + 1)
        );
      } while (!_free_head.compare_exchange_weak(
        cur, next,
        std::memory_order_release,   // publish next_free write to pop_free
        std::memory_order_relaxed
      ));
    }

    Block* pop_free() noexcept {
      H cur = _free_head.load(std::memory_order_acquire);
      while (cur.get_ptr()) {
        auto*  p  = reinterpret_cast<Block*>(cur.get_ptr());
        // relaxed on next_free: the acquire on _free_head (either the load
        // above or the acquire failure ordering below) synchronises-with the
        // release CAS in push_free, so the next_free store that preceded it
        // is already visible to this thread.
        Block* nx = p->next_free.load(std::memory_order_relaxed);
        H next(
          reinterpret_cast<typename H::pointer_type>(nx),
          static_cast<typename H::tag_type>(cur.get_tag() + 1)
        );
        if (_free_head.compare_exchange_weak(
              cur, next,
              std::memory_order_acquire,  // success: synchronise with push_free
              std::memory_order_acquire   // failure: fresh cur must also synchronise
        )) {                              //   before the next next_free read
          return p;
        }
      }
      return nullptr;
    }

    Block* allocate_from_backing() {
      return static_cast<Block*>(
        _backing.allocate(sizeof(Block), alignof(Block))
      );
    }

    void deallocate_to_backing(Block* b) {
      _backing.deallocate(b, sizeof(Block), alignof(Block));
    }
  };

  std::array<Shard, NumPools> _shards;

  // Returns the next shard index for this thread. The thread_local counter
  // is seeded once from the calling thread's ID hash, spreading different
  // threads across different starting shards with zero shared state.
  // Subsequent calls are a bare local increment — no atomic, no cache-line
  // traffic.
  //
  // If thread_local is broken (e.g. MSVC DLL with improper TLS), the counter
  // degrades to a single shared value and causes contention, but shard
  // selection remains correct.
  static size_t _next_shard() noexcept {
    thread_local size_t counter =
      std::hash<std::thread::id>{}(std::this_thread::get_id());
    return counter++ & (NumPools - 1);
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

  ObjectPool owns its shards and backing memory; copying is not meaningful.
  Declare the allocator as a global or long-lived member and share it by
  reference or pointer.
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
  backing blocks that are currently on the free stack. No per-block destructor
  is called; callers are responsible for recycling all live objects before
  destroying the allocator.
  */
  ~ObjectPool() = default;

  /**
  @brief constructs an object of type @c T in the pool and returns a pointer

  @tparam Args constructor argument types
  @param  args arguments forwarded to the constructor of @c T

  @return pointer to the newly constructed @c T; never null

  On the hot path, animate pops a previously recycled block from the shard's
  lock-free free stack and constructs @c T in it via @c std::construct_at,
  with no mutex acquisition. On a cache miss (empty free stack), a fresh block
  is carved from the shard's backing @c std::pmr::synchronized_pool_resource,
  which amortizes system allocation cost over chunks of up to 1024 blocks.

  Allocations are distributed across shards via a per-thread round-robin
  counter seeded from the thread ID hash, balancing load with zero shared
  state after initialization.

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
  The returned pointer must eventually be passed to tf::ObjectPool::recycle.
  Discarding it without recycling leaks both the object's resources and the
  underlying block.
  */
  template <typename... Args>
  [[nodiscard]] T* animate(Args&&... args) {
    auto  sid   = _next_shard();
    auto& shard = _shards[sid];

    Block* block = shard.pop_free();                      // hot path: lock-free
    if (!block) block = shard.allocate_from_backing();    // cold path: mutex, amortized

    block->pool_id = static_cast<uint16_t>(sid);
    return std::construct_at(block->object(), std::forward<Args>(args)...);
  }

  /**
  @brief destructs the object and returns its storage to the pool

  @param obj pointer to a @c T previously returned by tf::ObjectPool::animate,
             or @c nullptr (no-op)

  recycle calls the destructor of @c *obj via @c std::destroy_at, then pushes
  the underlying block onto its shard's lock-free free stack without acquiring
  any mutex. The block becomes immediately available for the next call to
  tf::ObjectPool::animate on any thread.

  The correct shard is identified via the @c pool_id stored in the block
  header, so recycle may be called from any thread regardless of which thread
  called animate.

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
    _shards[block->pool_id].push_free(block);             // hot path: lock-free
  }

  /**
  @brief returns all recycled blocks and backing memory to the system allocator

  release calls @c std::pmr::synchronized_pool_resource::release on each
  shard's backing pool, returning all chunks to the upstream system allocator
  in one shot, then atomically resets each shard's free stack to null. This
  is an O(1) operation per shard — no per-block work is performed because the
  backing pool owns memory at the chunk level and frees entire chunks
  regardless of how many individual blocks were returned to it.

  After this call the allocator is in the same state as after construction:
  empty free stacks, no memory held from the OS.

  This method is optional and is not required before destruction. It is useful
  for reclaiming pool memory between distinct workload phases without
  destroying the allocator itself.

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
  All live objects must be recycled before calling release. Calling release
  while objects are still alive is undefined behavior because the backing
  memory they reside in is freed.
  */
  void release() {
    for (auto& shard : _shards) {
      // Release all backing chunks to upstream first — this covers both blocks
      // on the free stack and any that were never recycled, since the backing
      // pool owns memory at the chunk level, not per block.
      shard._backing.release();
      // Reset the free stack to null in O(1). Pointers it held are now
      // dangling (their backing chunks were just freed), so they must be
      // cleared before the allocator is used again.
      shard._free_head.store(H{}, std::memory_order_relaxed);
    }
  }
};

} // namespace tf
