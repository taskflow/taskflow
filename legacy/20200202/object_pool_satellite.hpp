// 2020/02/02 - modified by Tsung-Wei Huang
//  - new implementation motivated by Hoard
// 
// 2019/07/10 - modified by Tsung-Wei Huang
//  - replace raw pointer with smart pointer
//
// 2019/06/13 - created by Tsung-Wei Huang
//  - implemented an object pool class

#pragma once

#include <iostream>
#include <thread>
#include <new>       
#include <mutex>
#include <vector>
#include <list>
#include <array>
#include <cassert>

// ----------------------------------------------------------------------------
// aligned_alloc definition
// ----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: MSVC in general has no aligned alloc function that is
// compatible with free and it is not trivial to implement a version
// which is. Therefore, to remain portable, end user code needs to
// use `aligned_free` which is not part of C11 but defined in this header.
// 
// The same issue is present on some Unix systems not providing
// posix_memalign.
// 
// Note that clang and gcc with -std=c11 or -std=c99 will not define
// _POSIX_C_SOURCE and thus posix_memalign cannot be detected but
// aligned_alloc is not necessarily available either. We assume
// that clang always has posix_memalign although it is not strictly
// correct. For gcc, use -std=gnu99 or -std=gnu11 or don't use -std in
// order to enable posix_memalign, or live with the fallback until using
// a system where glibc has a version that supports aligned_alloc.
// 
// For C11 compliant compilers and compilers with posix_memalign,
// it is valid to use free instead of aligned_free with the above
// caveats.
//
// source: https://github.com/dvidelabs/flatcc

//Define this to see which version is used so the fallback is not
//enganged unnecessarily:
//
//#define TF_DEBUG_ALIGNED_ALLOC


#if 0
#define TF_DEBUG_ALIGNED_ALLOC
#endif

#if !defined(TF_C11_ALIGNED_ALLOC)

  // glibc aligned_alloc detection.
  #if defined (_ISOC11_SOURCE)
    #define TF_C11_ALIGNED_ALLOC 1
  // aligned_alloc is not available in glibc just because 
  // __STDC_VERSION__ >= 201112L.
  #elif defined (__GLIBC__)
    #define TF_C11_ALIGNED_ALLOC 0
  #elif defined (__clang__)
    #define TF_C11_ALIGNED_ALLOC 0
  #elif defined(__IBMC__)
    #define TF_C11_ALIGNED_ALLOC 0
  #elif (defined(__STDC__) && __STDC__ && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    #define TF_C11_ALIGNED_ALLOC 1
  #else
    #define TF_C11_ALIGNED_ALLOC 0
  #endif

#endif // TF_C11_ALIGNED_ALLOC

// https://linux.die.net/man/3/posix_memalign
#if !defined(TF_POSIX_MEMALIGN)

  // https://forum.kde.org/viewtopic.php?p=66274
  #if (defined _GNU_SOURCE) || ((_XOPEN_SOURCE + 0) >= 600) || (_POSIX_C_SOURCE + 0) >= 200112L 
    #define TF_POSIX_MEMALIGN 1
  #elif defined (__clang__)
    #define TF_POSIX_MEMALIGN 1
  #else
    #define TF_POSIX_MEMALIGN 0
  #endif
#endif // TF_POSIX_MEMALIGN

// https://forum.kde.org/viewtopic.php?p=66274 
#if (defined(__STDC__) && __STDC__ && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
  // C11 or newer
  #include <stdalign.h>
#endif

// C11 or newer
#if !defined(aligned_alloc) && !defined(__aligned_alloc_is_defined)

  #if TF_C11_ALIGNED_ALLOC
    #ifdef TF_DEBUG_ALIGNED_ALLOC
    #error "DEBUG: c11 aligned_alloc configured"
    #endif
  // Aligned _aligned_malloc is not compatible with free.
  #elif defined(_MSC_VER)
    #ifdef TF_DEBUG_ALIGNED_ALLOC
    #error "DEBUG: MS _aligned_malloc and _aligned_free configured"
    #endif

    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(p) _aligned_free(p)
    #define __aligned_alloc_is_defined 1
    #define __aligned_free_is_defined 1
  #elif TF_POSIX_MEMALIGN

    #ifdef TF_DEBUG_ALIGNED_ALLOC
    #error "DEBUG: POSIX posix_memalign configured"
    #endif

    #if defined(__GNUC__) && __GNUCC__ < 5
    extern int posix_memalign (void **, size_t, size_t);
    #endif

    static inline void *__portable_aligned_alloc(size_t align, size_t size)
    {
      int err;
      void *p = 0;
    
      if (align < sizeof(void *)) {
        align = sizeof(void *);
      }
      err = posix_memalign(&p, align, size);
      if (err && p) {
        free(p);
        p = 0;
      }
      return p;
    }

    #define aligned_alloc(align, size) __portable_aligned_alloc(align, size)
    #define aligned_free(p) free(p)
    #define __aligned_alloc_is_defined 1
    #define __aligned_free_is_defined 1

  #else
    
    #ifdef TF_DEBUG_ALIGNED_ALLOC
    #error "DEBUG: malloc fallback configured"
    #endif

    static inline void *__portable_aligned_alloc(size_t align, size_t size)
    {
      char *raw;
      void *buf;
      size_t total_size = (size + align - 1 + sizeof(void *));
    
      if (align < sizeof(void *)) {
          align = sizeof(void *);
      }
      raw = (char *)(size_t)malloc(total_size);
      buf = raw + align - 1 + sizeof(void *);
      buf = (void *)(((size_t)buf) & ~(align - 1));
      ((void **)buf)[-1] = raw;
      return buf;
    }
    
    static inline void __portable_aligned_free(void *p)
    {
      char *raw;
      
      if (p) {
        raw = (char*)((void **)p)[-1];
        free(raw);
      }
    }

    #define aligned_alloc(align, size) __portable_aligned_alloc(align, size)
    #define aligned_free(p) __portable_aligned_free(p)
    #define __aligned_alloc_is_defined 1
    #define __aligned_free_is_defined 1
    
  #endif

#endif // aligned_alloc

#if !defined(aligned_free) && !defined(__aligned_free_is_defined)
  #define aligned_free(p) free(p)
  #define __aligned_free_is_defined 1
#endif

#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------
// ObjectPool definition
// ----------------------------------------------------------------------------

namespace tf {

// Class: ObjectPool
template <typename T, typename MutexT=std::mutex>
class ObjectPool { 

  struct GlobalHeap;
  struct LocalHeap;
  struct Block;

  public:

  constexpr static size_t S = 8192;
  
  constexpr static size_t F = 4;   
  constexpr static size_t B = F + 1;
  constexpr static size_t M = 32;  
  constexpr static size_t W = 8;
  constexpr static size_t K = 4;

  static_assert(M/F == W && M%F == 0);

  using satellite_t = Block*;

  private:
  
  struct Blocklist {
    Blocklist* prev;
    Blocklist* next;
  };

  struct Block {
    LocalHeap* heap;
    Blocklist list_node;
    size_t i;
    size_t u;
    T* top;
    char data[M*sizeof(T)];

    Block(const Block&) = delete;
    Block(Block&&) = delete;

    Block& operator = (const Block&) = delete;
    Block& operator = (Block&&) = delete;

    bool empty() const;
    void push(T*);
    T* pop();
    T* allocate();
  };

  class GlobalHeap {
    friend class ObjectPool;
    MutexT mutex;
    Blocklist list;
  };

  class LocalHeap {
    friend class ObjectPool;
    MutexT mutex;
    Blocklist lists[B];
    size_t u {0};
    size_t a {0};
  };

  public:

    explicit ObjectPool(unsigned = std::thread::hardware_concurrency());
    ~ObjectPool();

    template <typename... ArgsT>
    T* construct(ArgsT&&... args);

    void destruct(T*);

    size_t num_local_heaps() const;
    size_t num_global_heaps() const;
    size_t num_heaps() const;

  private:

    GlobalHeap _gheap;

    std::vector<LocalHeap> _lheaps;

    LocalHeap& _this_heap();

    unsigned _next_power_of_two(unsigned n) const;

    template <class P, class Q>
    constexpr size_t _offset_in_class(const Q P::*member);
    
    template <class P, class Q>
    constexpr P* _parent_class_of(Q* ptr, const Q P::*member);

    constexpr Block* _block_of(Blocklist*);

    void _blocklist_init_head(Blocklist*);
    void _blocklist_add_impl(Blocklist*, Blocklist*, Blocklist*);
    void _blocklist_push_front(Blocklist*, Blocklist*);
    void _blocklist_push_back(Blocklist*, Blocklist*);
    void _blocklist_del_impl(Blocklist*, Blocklist*);
    void _blocklist_del(Blocklist*);
    void _blocklist_replace(Blocklist*, Blocklist*);
    void _blocklist_move_front(Blocklist*, Blocklist*);
    void _blocklist_move_back(Blocklist*, Blocklist*);
    bool _blocklist_is_first(const Blocklist*, const Blocklist*);
    bool _blocklist_is_last(const Blocklist*, const Blocklist*);
    bool _blocklist_is_empty(const Blocklist*);
    bool _blocklist_is_singular(const Blocklist*);

    template <typename C>
    void _for_each_block_safe(Blocklist*, C&&);

    template <typename C>
    void _for_each_block(Blocklist*, C&&);

};
    
// ----------------------------------------------------------------------------
// Block definition
// ----------------------------------------------------------------------------

// Function: allocate
template <typename T, typename MutexT>
T* ObjectPool<T, MutexT>::Block::allocate() {
  return empty() ? reinterpret_cast<T*>(data) + i++ : pop();
}

// Function: empty
template <typename T, typename MutexT>
bool ObjectPool<T, MutexT>::Block::empty() const { 
  return top == nullptr; 
}

// Procedure: push    
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::Block::push(T* ptr) {
  *(reinterpret_cast<T**>(ptr)) = top;
  top = ptr;
}
  
// Function: pop
// Must use empty() before calling pop
template <typename T, typename MutexT>
T* ObjectPool<T, MutexT>::Block::pop() {
  assert(!empty());
  T* retval = top;
  top = *(reinterpret_cast<T**>(top));
  return retval;
}

// ----------------------------------------------------------------------------
// ObjectPool definition
// ----------------------------------------------------------------------------

// Constructor
template <typename T, typename MutexT>
ObjectPool<T, MutexT>::ObjectPool(unsigned t) :
  //_heap_mask   {(_next_power_of_two(t) << 1) - 1u},
  //_heap_mask   { _next_power_of_two(t<<1) - 1u },
  //_heap_mask   {(t << 1) - 1},
  _lheaps { (t+1) << 1 } {

  _blocklist_init_head(&_gheap.list);

  for(auto& h : _lheaps) {
    for(size_t i=0; i<B; ++i) {
      _blocklist_init_head(&h.lists[i]);
    }
  }
}

template <typename T, typename MutexT>
ObjectPool<T, MutexT>::~ObjectPool() {

  // clear local heaps
  for(auto& h : _lheaps) {
    for(size_t i=0; i<B; ++i) {
      _for_each_block_safe(&h.lists[i], [] (Block* b) { std::free(b); });
    }
  }
  
  // clear global heap
  _for_each_block_safe(&_gheap.list, [] (Block* b) { std::free(b); } );

}

// Function: num_global_heaps
template <typename T, typename MutexT>
size_t ObjectPool<T, MutexT>::num_global_heaps() const {
  return 1;
}

// Function: num_lheaps
template <typename T, typename MutexT>
size_t ObjectPool<T, MutexT>::num_local_heaps() const {
  return _lheaps.size();
}

// Function: num_heaps
template <typename T, typename MutexT>
size_t ObjectPool<T, MutexT>::num_heaps() const {
  return _lheaps.size() + 1;
}
    
template <typename T, typename MutexT>
template <class P, class Q>
constexpr size_t ObjectPool<T, MutexT>::_offset_in_class(const Q P::*member) {
  return (size_t) &( reinterpret_cast<P*>(0)->*member);
}

// C macro: parent_class_of(list_pointer, Block, list)
// C++: parent_class_of(list_pointer,  &Block::list)
template <typename T, typename MutexT>
template <class P, class Q>
constexpr P* ObjectPool<T, MutexT>::_parent_class_of(Q* ptr, const Q P::*member) {
  return (P*)( (char*)ptr - _offset_in_class(member));
}

template <typename T, typename MutexT>
constexpr typename ObjectPool<T, MutexT>::Block* 
ObjectPool<T, MutexT>::_block_of(Blocklist* list) {
  return _parent_class_of(list, &Block::list_node);
}

// Procedure: initialize a list head
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_init_head(Blocklist *list) {
  list->next = list;
  list->prev = list;
}

// Procedure: _blocklist_add_impl
// Insert a new entry between two known consecutive entries.
// 
// This is only for internal list manipulation where we know
// the prev/next entries already!
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_add_impl(
  Blocklist *curr, Blocklist *prev, Blocklist *next
) {
  next->prev = curr;
  curr->next = next;
  curr->prev = prev;
  prev->next = curr;
}

// list_push_front - add a new entry
// @curr: curr entry to be added
// @head: list head to add it after
// 
// Insert a new entry after the specified head.
// This is good for implementing stacks.
// 
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_push_front(
  Blocklist *curr, Blocklist *head
) {
  _blocklist_add_impl(curr, head, head->next);
}

// list_add_tail - add a new entry
// @curr: curr entry to be added
// @head: list head to add it before
// 
// Insert a new entry before the specified head.
// This is useful for implementing queues.
// 
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_push_back(
  Blocklist *curr, Blocklist *head
) {
  _blocklist_add_impl(curr, head->prev, head);
}

// Delete a list entry by making the prev/next entries
// point to each other.
// 
// This is only for internal list manipulation where we know
// the prev/next entries already!
// 
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_del_impl(
  Blocklist * prev, Blocklist * next
) {
  next->prev = prev;
  prev->next = next;
}

// _blocklist_del - deletes entry from list.
// @entry: the element to delete from the list.
// Note: list_empty() on entry does not return true after this, the entry is
// in an undefined state.
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_del(Blocklist *entry) {
  _blocklist_del_impl(entry->prev, entry->next);
  entry->next = nullptr;
  entry->prev = nullptr;
}

// list_replace - replace old entry by new one
// @old : the element to be replaced
// @curr : the new element to insert
// 
// If @old was empty, it will be overwritten.
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_replace(
  Blocklist *old, Blocklist *curr
) {
  curr->next = old->next;
  curr->next->prev = curr;
  curr->prev = old->prev;
  curr->prev->next = curr;
}

// list_move - delete from one list and add as another's head
// @list: the entry to move
// @head: the head that will precede our entry
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_move_front(
  Blocklist *list, Blocklist *head
) {
  _blocklist_del_impl(list->prev, list->next);
  _blocklist_push_front(list, head);
}

// list_move_tail - delete from one list and add as another's tail
// @list: the entry to move
// @head: the head that will follow our entry
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::_blocklist_move_back(
  Blocklist *list, Blocklist *head
) {
  _blocklist_del_impl(list->prev, list->next);
  _blocklist_push_back(list, head);
}

// list_is_first - tests whether @list is the last entry in list @head
// @list: the entry to test
// @head: the head of the list
template <typename T, typename MutexT>
bool ObjectPool<T, MutexT>::_blocklist_is_first(
  const Blocklist *list, const Blocklist *head
) {
  return list->prev == head;
}

// list_is_last - tests whether @list is the last entry in list @head
// @list: the entry to test
// @head: the head of the list
template <typename T, typename MutexT>
bool ObjectPool<T, MutexT>::_blocklist_is_last(
  const Blocklist *list, const Blocklist *head
) {
  return list->next == head;
}

// list_empty - tests whether a list is empty
// @head: the list to test.
template <typename T, typename MutexT>
bool ObjectPool<T, MutexT>::_blocklist_is_empty(const Blocklist *head) {
  return head->next == head;
}

// list_is_singular - tests whether a list has just one entry.
// @head: the list to test.
template <typename T, typename MutexT>
bool ObjectPool<T, MutexT>::_blocklist_is_singular(const Blocklist *head) {
  return !_blocklist_is_empty(head) && (head->next == head->prev);
}

// Procedure: _for_each_block
template <typename T, typename MutexT>
template <typename C>
void ObjectPool<T, MutexT>::_for_each_block(Blocklist* head, C&& c) {
  Blocklist* p;
  for(p=head->next; p!=head; p=p->next) {
    c(_block_of(p));
  }
}
      
// Procedure: _for_each_block_safe
// Iterate each item of a list - safe to free
template <typename T, typename MutexT>
template <typename C>
void ObjectPool<T, MutexT>::_for_each_block_safe(Blocklist* head, C&& c) {
  Blocklist* p;
  Blocklist* t;
  for(p=head->next, t=p->next; p!=head; p=t, t=p->next) {
    c(_block_of(p));
  }
}

// Function: allocate
template <typename T, typename MutexT>
template <typename... ArgsT>
T* ObjectPool<T, MutexT>::construct(ArgsT&&... args) {

  //std::cout << "construct a new item\n";
    
  // my logically mapped heap
  LocalHeap& h = _this_heap(); 
  
  satellite_t s;

  h.mutex.lock();
  
  // scan the list of superblocks from most full to least
  int f = F-1;
  for(; f>=0; f--) {
    if(!_blocklist_is_empty(&h.lists[f])) {
      //s = h.lists[f].begin();
      s = _block_of(h.lists[f].next);
      break;
    }
  }
  
  // no superblock found
  if(f == -1) {

    // check heap 0 for a superblock
    _gheap.mutex.lock();
    if(!_blocklist_is_empty(&_gheap.list)) {
      
      //s = _gheap.list.begin();
      s = _block_of(_gheap.list.next);
      
      //printf("get a superblock from global heap %lu\n", s->u);
      assert(s->u < M && s->heap == nullptr);
      f = (s->u + 1) / W;
      //h.lists[f].splice(h.lists[f].begin(), _gheap.list, s);

      _blocklist_move_front(&s->list_node, &h.lists[f]);

      s->heap = &h;  // must be wihin the global heap lock
      _gheap.mutex.unlock();

      h.u = h.u + s->u;
      h.a = h.a + M;
    }
    // create a new block
    else {
      //printf("create a new superblock\n");
      _gheap.mutex.unlock();
      f = 0;
      //h.lists[f].emplace_front(&h);
      //s = h.lists[f].begin();

      s = static_cast<Block*>(std::malloc(sizeof(Block)));
      s->heap = &h; 
      s->i = 0;
      s->u = 0;
      s->top = nullptr;
      _blocklist_push_front(&s->list_node, &h.lists[f]);

      h.a = h.a + M;
    }
  }
  
  // the superblock must have at least one space
  assert(s->u < M);
  //printf("%lu %lu %lu\n", h.u, h.a, s->u);
  assert(h.u < h.a);

  h.u = h.u + 1;
  s->u = s->u + 1;

  // take one item from the superblock
  //T* mem;
  //if(s->freelist.empty()) {
  //  assert(s->i < M);
  //  mem = s->block + s->i;
  //  s->i = s->i + 1;
  //}
  //else {
  //  mem = s->freelist.pop();
  //}
  T* mem = s->allocate();
  
  int b = s->u / W;
  
  if(b != f) {
    //printf("move superblock from list[%d] to list[%d]\n", f, b);
    //h.lists[b].splice(h.lists[b].begin(), h.lists[f], s);

    _blocklist_move_front(&s->list_node, &h.lists[b]);
  }

  //std::cout << "s.i " << s->i << '\n'
  //          << "s.u " << s->u << '\n'
  //          << "h.u " << h.u  << '\n'
  //          << "h.a " << h.a  << '\n';

  h.mutex.unlock();

  // set the satellite 
  new (mem) T(std::forward<ArgsT>(args)...);

  mem->satellite = s;

  return mem;
}
  
// Function: destruct
template <typename T, typename MutexT>
void ObjectPool<T, MutexT>::destruct(T* mem) {

  //printf("destruct %p\n", mem);

  auto s = mem->satellite;

  // destruct the item
  mem->~T();
  
  // here we need a loop because when we lock the heap,
  // other threads may have removed the superblock to another heap
  bool sync = false;

  while(!sync) {
    
    auto h = s->heap;    
    
    // the block is in global heap
    if(h == nullptr) {
      std::lock_guard<MutexT> glock(_gheap.mutex);
      if(s->heap == h) {
        sync = true;
        s->push(mem);
        s->u = s->u - 1;
      }
    }
    else {
      std::lock_guard<MutexT> llock(h->mutex);
      if(s->heap == h) {
        sync = true;
        // deallocate the item from the superblock
        int f = s->u / W;

        s->push(mem);
        s->u = s->u - 1;
        h->u = h->u - 1;

        int b = s->u / W;

        if(b != f) {
          //printf("move superblock from list[%d] to list[%d]\n", f, b);
          //h->lists[b].splice(h->lists[b].begin(), h->lists[f], s);
          _blocklist_move_front(&s->list_node, &h->lists[b]);
        }

        // transfer a mostly-empty superblock to global heap
        if((h->u + K*M < h->a) && (h->u < ((F-1) * h->a / F))) {
          for(size_t i=0; i<F; i++) {
            //if(!h->lists[i].empty()) {
            if(!_blocklist_is_empty(&h->lists[i])) {
              //auto x = h->lists[i].begin();
              Block* x = _block_of(h->lists[i].next);
              //printf("transfer a block (x.u=%lu/x.i=%lu) to the global heap\n", x->u, x->i);
              assert(h->u > x->u && h->a > M);
              h->u = h->u - x->u;
              h->a = h->a - M;
              x->heap = nullptr;
              std::lock_guard<MutexT> glock(_gheap.mutex);
              //_gheap.list.splice(_gheap.list.begin(), h->lists[i], x);
              _blocklist_move_front(&x->list_node, &_gheap.list);
              break;
            }
          }
        }
      }
    }
  }
  
  //std::cout << "s.i " << s->i << '\n'
  //          << "s.u " << s->u << '\n'
  //          << "h.u " << h->u  << '\n'
  //          << "h.a " << h->a  << '\n';
}
    
// Function: _this_heap
template <typename T, typename MutexT>
typename ObjectPool<T, MutexT>::LocalHeap& 
ObjectPool<T, MutexT>::_this_heap() {
  thread_local LocalHeap& heap = _lheaps[
    //std::hash<std::thread::id>()(std::this_thread::get_id()) & _heap_mask
    std::hash<std::thread::id>()(std::this_thread::get_id()) % _lheaps.size()
  ];
  return heap;
}

// Function: _next_power_of_two
template <typename T, typename MutexT>
unsigned ObjectPool<T, MutexT>::_next_power_of_two(unsigned n) const { 
  n--; 
  n |= n >> 1; 
  n |= n >> 2; 
  n |= n >> 4; 
  n |= n >> 8; 
  n |= n >> 16; 
  n++; 
  return n; 
}  

}  // end of namespace tf -----------------------------------------------------








