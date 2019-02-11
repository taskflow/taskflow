// 2019/02/08 - created by Chun-Xun Lin
//  - added a singular memory allocator
//  - refactored by Tsung-Wei Huang

#pragma once

#include <new>       
#include <mutex>
#include <vector>
#include <thread>
#include <cassert>

namespace tf {

// Class: Mempool
template <typename T>
struct Mempool { 
  
  // FreeList uses Node array to store Node*
  struct FreeList {
    typedef T*       pointer;
    typedef pointer* ppointer;
  
    FreeList() = default;
  
    void push(pointer ptr) {
      *(reinterpret_cast<ppointer>(ptr)) = top;
      top = ptr;
    }
  
    // Must use empty() before calling pop
    pointer pop()  {
      assert(!empty());
      pointer retval = top;
      top = *(reinterpret_cast<ppointer>(top));
      return retval;
    }
  
    bool empty() const { return top == nullptr; }
  
    pointer top {nullptr};
  };

  struct MemBlock{
    T* block;                            // Block memory.
    size_t size;                         // Size of the block (count).
    struct MemBlock* next;               // Pointer to the next block.
  };

  // Ctor
  Mempool() {
    head = allocate_memblock(1024);
    tail = head;
  }

  // Dtor
  ~Mempool() {
    for(auto* prev = head; head!=nullptr;) {
      head = head->next;
      std::free(prev->block);
      std::free(prev);
      prev = head;
    }
  }

  MemBlock* allocate_memblock(const size_t n) {
    MemBlock* ptr = static_cast<MemBlock*>(std::malloc(sizeof(MemBlock)));
    ptr->block = static_cast<T*>(std::malloc(n*sizeof(T)));
    ptr->size = n;
    ptr->next = nullptr;
    return ptr;
  }

  T* allocate(const size_t n) {
    // TODO: we only deal with single element
    assert(n == 1);

    if(!free_list.empty()) {
      return free_list.pop();
    }

    if(tail->size == used) {
      auto next = allocate_memblock(tail->size << 1);
      tail->next = next;
      tail = next;
      used = 0;
    }
    return &(tail->block[used++]);
  }

  void deallocate(T* ptr, const size_t n) {
    // TODO: we only deal with single element
    assert(n == 1);
    free_list.push(ptr);
  }

  FreeList free_list;
  
  MemBlock *head {nullptr};
  MemBlock *tail {nullptr};
  size_t used {0};
};

// Class: MempoolManager
template <typename T>  
struct MempoolManager {

  struct Handle {

    Handle(MempoolManager<T> &mgr) : manager {mgr} {
      std::scoped_lock lock(mgr.mtx);
      if(mgr.pools.empty()) {
        mempool = new Mempool<T>();
      }
      else {
        mempool = mgr.pools.back();
        mgr.pools.pop_back();
      }
    }
  
    ~Handle() {
      // Return the memory pool to MempoolManager
      std::scoped_lock lock(manager.mtx);
      manager.pools.emplace_back(mempool);
    }
  
    MempoolManager<T>& manager;
    Mempool<T>* mempool {nullptr};
  };

  // Ctor 
  MempoolManager() { 
    pools.reserve(std::thread::hardware_concurrency()); 
  }

  // Dtor
  ~MempoolManager() {
    std::scoped_lock lock(mtx);
    for(auto& p : pools) {
      delete p;
    }
  }

  Mempool<T>* get_per_thread_mempool() {
    thread_local Handle handle {*this};
    return handle.mempool;
  }

  std::mutex mtx;
  std::vector<Mempool<T>*> pools;
};
  
// The singleton allocator
template <typename T> 
auto& get_mempool_manager() {
  static MempoolManager<T> manager;
  return manager;
}

// Class: SingularAllocator
template <typename T>
class SingularAllocator {

  public:

    // Allocator traits
    typedef T              value_type     ;
    typedef T*             pointer        ;
    typedef const T*       const_pointer  ;
    typedef T&             reference      ;
    typedef const T&       const_reference;
    typedef std::size_t    size_type      ;
    typedef std::ptrdiff_t difference_type;

    template <typename U> 
    struct rebind {
      typedef SingularAllocator<U> other;
    };

    SingularAllocator() = default;                          // Constructor.
    ~SingularAllocator() = default;                         // Destructor.

    inline T* allocate(const size_t n=1) ;                  // Allocate an entry of type T.
    inline void deallocate(T*, const size_t n=1);           // Deallocate an entry of type T.
    
    template <typename... ArgsT>               
    inline void construct(T*, ArgsT&&...);                  // Construct an item.
    inline void destroy(T*);                                // Destroy an item.  

    SingularAllocator & operator = (const SingularAllocator &) {} 

    bool operator == (const SingularAllocator &) const { return true; }
    bool operator != (const SingularAllocator &) const { return false; }
};

// Procedure: construct
// Construct an item with placement new.
template <typename T>
template <typename... ArgsT>
inline void SingularAllocator<T>::construct(T* ptr, ArgsT&&... args) {
  new (ptr) T(std::forward<ArgsT>(args)...); 
}

// Procedure: destroy
// Destroy an item
template <typename T>
inline void SingularAllocator<T>::destroy(T* ptr) {
  ptr->~T();
}

// Function: allocate
// Allocate a memory piece of type T from the memory pool and return the T* to that memory.
template <typename T>
inline T* SingularAllocator<T>::allocate(const size_t n) {
  return get_mempool_manager<T>().get_per_thread_mempool()->allocate(n);
}

// Function: deallocate
// Deallocate given memory piece of type T.
template <typename T>
inline void SingularAllocator<T>::deallocate(T* ptr, const size_t n) {
  get_mempool_manager<T>().get_per_thread_mempool()->deallocate(ptr, n); 
}

};  // End of namespace tf. ---------------------------------------------------

