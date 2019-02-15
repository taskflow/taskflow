// 2019/02/15 - modified by Tsung-Wei Huang
//   - refactored the code
//
// 2019/02/10 - modified by Tsung-Wei Huang
//   - fixed the compilation error on MS platform
//
// 2019/02/08 - created by Chun-Xun Lin
//   - added a singular memory allocator
//   - refactored by Tsung-Wei Huang

#pragma once

#include <new>       
#include <mutex>
#include <vector>
#include <thread>
#include <cassert>

namespace tf {

// Class: SingularMempool
template <typename T>
struct SingularMempool { 
  
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
  SingularMempool() {
    head = allocate_memblock(1024);
    tail = head;
  }

  // Dtor
  ~SingularMempool() {
    for(auto* prev = head; head!=nullptr;) {
      head = head->next;
      std::free(prev->block);
      std::free(prev);
      prev = head;
    }
  }

  MemBlock* allocate_memblock(size_t n) {
    MemBlock* ptr = static_cast<MemBlock*>(std::malloc(sizeof(MemBlock)));
    ptr->block = static_cast<T*>(std::malloc(n*sizeof(T)));
    ptr->size = n;
    ptr->next = nullptr;
    return ptr;
  }

  T* allocate() {

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

  void deallocate(T* ptr) {
    free_list.push(ptr);
  }

  FreeList free_list;
  
  MemBlock *head {nullptr};
  MemBlock *tail {nullptr};
  size_t used {0};
};

// Class: SingularMempoolManager
template <typename T>  
struct SingularMempoolManager {

  struct Handle {

    Handle(SingularMempoolManager<T> &mgr) : manager {mgr} {
      std::scoped_lock lock(mgr.mtx);
      if(mgr.pools.empty()) {
        mempool = new SingularMempool<T>();
      }
      else {
        mempool = mgr.pools.back();
        mgr.pools.pop_back();
      }
    }
  
    ~Handle() {
      // Return the memory pool to SingularMempoolManager
      std::scoped_lock lock(manager.mtx);
      manager.pools.emplace_back(mempool);
    }
  
    SingularMempoolManager<T>& manager;
    SingularMempool<T>* mempool {nullptr};
  };

  // Ctor 
  SingularMempoolManager() { 
    pools.reserve(std::thread::hardware_concurrency()); 
  }

  // Dtor
  ~SingularMempoolManager() {
    std::scoped_lock lock(mtx);
    for(auto& p : pools) {
      delete p;
    }
  }

  SingularMempool<T>* get_per_thread_mempool() {
    thread_local Handle handle {*this};
    return handle.mempool;
  }

  std::mutex mtx;
  std::vector<SingularMempool<T>*> pools;
};
  
// The singleton allocator
template <typename T> 
auto& get_singular_mempool_manager() {
  static SingularMempoolManager<T> manager;
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

    explicit SingularAllocator() {}
    ~SingularAllocator() {}

    explicit SingularAllocator(const SingularAllocator&) {}

    template<typename U>
    explicit SingularAllocator(const SingularAllocator<U>&) {}

    T* allocate(size_t n=1) ;          
    void deallocate(T*, size_t n=1); 
    
    template <typename... ArgsT>               
    void construct(T*, ArgsT&&...);
    void destroy(T*);

    //SingularAllocator & operator = (const SingularAllocator &) {} 

    bool operator == (const SingularAllocator &) const { return true; }
    bool operator != (const SingularAllocator &) const { return false; }
};

// Procedure: construct
// Construct an item with placement new.
template <typename T>
template <typename... ArgsT>
void SingularAllocator<T>::construct(T* ptr, ArgsT&&... args) {
  new (ptr) T(std::forward<ArgsT>(args)...); 
}

// Procedure: destroy
// Destroy an item
template <typename T>
void SingularAllocator<T>::destroy(T* ptr) {
  ptr->~T();
}

// Function: allocate
// Allocate a memory piece of type T from the memory pool and return the T* to that memory.
template <typename T>
T* SingularAllocator<T>::allocate(size_t n) {
  assert(n == 1);
  return get_singular_mempool_manager<T>().get_per_thread_mempool()->allocate();
}

// Function: deallocate
// Deallocate given memory piece of type T.
template <typename T>
void SingularAllocator<T>::deallocate(T* ptr, size_t n) {
  assert(n == 1);
  get_singular_mempool_manager<T>().get_per_thread_mempool()->deallocate(ptr); 
}

}  // End of namespace tf. ----------------------------------------------------

