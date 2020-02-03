// 2019/02/10 - modified by Tsung-Wei Huang
//  - fixed the compilation error on MS platform
//
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

  size_t rank(const uint64_t v) {
    assert(v != 0 && (v&(v-1)) == 0);
#ifdef __GNUC__  
    return __builtin_ffs(v) - 1; // GCC
#endif
#ifdef _MSC_VER
    return CHAR_BIT * sizeof(x)-__lzcnt( x ); // Visual studio
#endif
  }

  void deposit() {
    auto left = tail->size - used;
    size_t pos = rank(left & (~left+1));  // get rightmost set bit
    size_t num {0};
    left >>= pos;
    for(; left; left>>=1, pos++) {
      if(left & 1) {
        free_list[pos].push(&tail->block[used+num]);
        num += (1 << pos);
      }
    }
    assert(num + used == tail->size);
  }

  MemBlock* allocate_memblock(size_t n) {
    assert(n && (n & (n-1)) == 0);
    MemBlock* ptr = static_cast<MemBlock*>(std::malloc(sizeof(MemBlock)));
    ptr->block = static_cast<T*>(std::malloc(n*sizeof(T)));
    ptr->size = n;
    ptr->next = nullptr;
    return ptr;
  }

  T* allocate(size_t n) {
    // TODO: we only deal with n which is power of 2
    assert(n && (n & (n-1)) == 0);

    auto r = rank(n);

    if(!free_list[r].empty())  {
      return free_list[r].pop();
    }

    if(tail->size < n + used) {
      auto next = allocate_memblock(std::max(n, tail->size) << 1);
      if(tail->size > used) deposit();
      tail->next = next;
      tail = next;
      used = 0;
    }
    used += n;
    return &(tail->block[used-n]);
  }

  void deallocate(T* ptr, size_t n) {
    // TODO: we only deal with n which is power of 2
    assert(n && (n & (n-1)) == 0);
    auto pos = rank(n);
    free_list[pos].push(ptr);
  }

  FreeList free_list [64];
  
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

// Class: GenericAllocator
template <typename T>
class GenericAllocator {

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
      typedef GenericAllocator<U> other;
    };

    explicit GenericAllocator() {}
    ~GenericAllocator() {}

    explicit GenericAllocator(const GenericAllocator&) {}

    template<typename U>
    explicit GenericAllocator(const GenericAllocator<U>&) {}

    T* allocate(size_t n=1) ;          
    void deallocate(T*, size_t n=1); 
    
    template <typename... ArgsT>               
    void construct(T*, ArgsT&&...);
    void destroy(T*);

    //GenericAllocator & operator = (const GenericAllocator &) {} 

    bool operator == (const GenericAllocator &) const { return true; }
    bool operator != (const GenericAllocator &) const { return false; }
};

// Procedure: construct
// Construct an item with placement new.
template <typename T>
template <typename... ArgsT>
void GenericAllocator<T>::construct(T* ptr, ArgsT&&... args) {
  new (ptr) T(std::forward<ArgsT>(args)...); 
}

// Procedure: destroy
// Destroy an item
template <typename T>
void GenericAllocator<T>::destroy(T* ptr) {
  ptr->~T();
}

// Function: allocate
// Allocate a memory piece of type T from the memory pool and return the T* to that memory.
template <typename T>
T* GenericAllocator<T>::allocate(size_t n) {
  return get_mempool_manager<T>().get_per_thread_mempool()->allocate(n);
}

// Function: deallocate
// Deallocate given memory piece of type T.
template <typename T>
void GenericAllocator<T>::deallocate(T* ptr, size_t n) {
  get_mempool_manager<T>().get_per_thread_mempool()->deallocate(ptr, n); 
}

}  // End of namespace tf. ----------------------------------------------------

