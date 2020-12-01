// 2020/03/13 - modified by Tsung-Wei Huang
//  - fixed bug in aligning memory
//
// 2020/02/02 - modified by Tsung-Wei Huang
//  - new implementation motivated by Hoard
// 
// 2019/07/10 - modified by Tsung-Wei Huang
//  - replace raw pointer with smart pointer
//
// 2019/06/13 - created by Tsung-Wei Huang
//  - implemented an object pool class

#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cassert>
#include <cstddef>

namespace tf {

#define TF_ENABLE_POOLABLE_ON_THIS                          \
  template <typename T, size_t S> friend class ObjectPool;  \
  void* _object_pool_block

// Class: ObjectPool
//
// The class implements an efficient thread-safe object pool motivated
// by the Hoard memory allocator algorithm. 
// Different from the normal memory allocator, object pool allocates
// only one object at a time.
//
// Internall, we use the following variables to maintain blocks and heaps:
// X: size in byte of a item slot
// M: number of items per block
// F: emptiness threshold
// B: number of bins per local heap (bin[B-1] is the full list)
// W: number of items per bin
// K: shrinkness constant
//
// Example scenario 1:
// M = 30
// F = 4
// W = (30+4-1)/4 = 8
// 
// b0: 0, 1, 2, 3, 4, 5, 6, 7
// b1: 8, 9, 10, 11, 12, 13, 14, 15
// b2: 16, 17, 18, 19, 20, 21, 22, 23
// b3: 24, 25, 26, 27, 28, 29
// b4: 30 (anything equal to M)
// 
// Example scenario 2:
// M = 32
// F = 4
// W = (32+4-1)/4 = 8
// b0: 0, 1, 2, 3, 4, 5, 6, 7
// b1: 8, 9, 10, 11, 12, 13, 14, 15
// b2: 16, 17, 18, 19, 20, 21, 22, 23
// b3: 24, 25, 26, 27, 28, 29, 30, 31
// b4: 32 (anything equal to M)
//
template <typename T, size_t S = 65536>
class ObjectPool { 
  
  // the data column must be sufficient to hold the pointer in freelist  
  constexpr static size_t X = (std::max)(sizeof(T*), sizeof(T));
  //constexpr static size_t X = sizeof(long double) + std::max(sizeof(T*), sizeof(T));
  //constexpr static size_t M = (S - offsetof(Block, data)) / X;
  constexpr static size_t M = S / X;
  constexpr static size_t F = 4;   
  constexpr static size_t B = F + 1;
  constexpr static size_t W = (M + F - 1) / F;
  constexpr static size_t K = 4;

  static_assert(
    S && (!(S & (S-1))), "block size S must be a power of two"
  );

  static_assert(
    M >= 128, "block size S must be larger enough to pool at least 128 objects"
  );
  
  struct Blocklist {
    Blocklist* prev;
    Blocklist* next;
  };

  class GlobalHeap {
    friend class ObjectPool;
    std::mutex mutex;
    Blocklist list;
  };

  class LocalHeap {
    friend class ObjectPool;
    std::mutex mutex;
    Blocklist lists[B];
    size_t u {0};
    size_t a {0};
  };

  struct Block {
    LocalHeap* heap;
    Blocklist list_node;
    size_t i;
    size_t u;
    T* top;
    // long double padding;
    char data[S];
  };

  public:
    
    /**
    @brief constructs an object pool from a number of anticipated threads
    */
    explicit ObjectPool(unsigned = std::thread::hardware_concurrency());

    /**
    @brief destructs the object pool
    */
    ~ObjectPool();
    
    /**
    @brief acquires a pointer to a object constructed from a given argument list
    */
    template <typename... ArgsT>
    T* animate(ArgsT&&... args);
    
    /**
    @brief recycles a object pointed by @c ptr and destroys it
    */
    void recycle(T* ptr);
    
    size_t num_bins_per_local_heap() const;
    size_t num_objects_per_bin() const;
    size_t num_objects_per_block() const;
    size_t num_available_objects() const;
    size_t num_allocated_objects() const;
    size_t capacity() const;
    size_t num_local_heaps() const;
    size_t num_global_heaps() const;
    size_t num_heaps() const;
    
    float emptiness_threshold() const;

  private:

    const size_t _lheap_mask;

    GlobalHeap _gheap;

    std::vector<LocalHeap> _lheaps;

    LocalHeap& _this_heap();

    constexpr unsigned _next_pow2(unsigned n) const;

    template <class P, class Q>
    constexpr size_t _offset_in_class(const Q P::*member) const;
    
    template <class P, class Q>
    constexpr P* _parent_class_of(Q*, const Q P::*member);

    template <class P, class Q>
    constexpr P* _parent_class_of(const Q*, const Q P::*member) const;

    constexpr Block* _block_of(Blocklist*);
    constexpr Block* _block_of(const Blocklist*) const;

    size_t _bin(size_t) const;

    T* _allocate(Block*);

    void _deallocate(Block*, T*);
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
// ObjectPool definition
// ----------------------------------------------------------------------------

// Constructor
template <typename T, size_t S>
ObjectPool<T, S>::ObjectPool(unsigned t) :
  //_heap_mask   {(_next_pow2(t) << 1) - 1u},
  //_heap_mask   { _next_pow2(t<<1) - 1u },
  //_heap_mask   {(t << 1) - 1},
  _lheap_mask { _next_pow2((t+1) << 1) - 1 },
  _lheaps     { _lheap_mask + 1 } {

  _blocklist_init_head(&_gheap.list);

  for(auto& h : _lheaps) {
    for(size_t i=0; i<B; ++i) {
      _blocklist_init_head(&h.lists[i]);
    }
  }
}

// Destructor
template <typename T, size_t S>
ObjectPool<T, S>::~ObjectPool() {

  // clear local heaps
  for(auto& h : _lheaps) {
    for(size_t i=0; i<B; ++i) {
      _for_each_block_safe(&h.lists[i], [] (Block* b) { 
        std::free(b); 
      });
    }
  }
  
  // clear global heap
  _for_each_block_safe(&_gheap.list, [] (Block* b) { 
    std::free(b);
  });
}
    
// Function: num_bins_per_local_heap
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_bins_per_local_heap() const {
  return B;
}

// Function: num_objects_per_bin
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_objects_per_bin() const {
  return W;
}

// Function: num_objects_per_block
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_objects_per_block() const {
  return M;
}

// Function: emptiness_threshold
template <typename T, size_t S>
float ObjectPool<T, S>::emptiness_threshold() const {
  return 1.0f/F;
}

// Function: num_global_heaps
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_global_heaps() const {
  return 1;
}

// Function: num_lheaps
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_local_heaps() const {
  return _lheaps.size();
}

// Function: num_heaps
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_heaps() const {
  return _lheaps.size() + 1;
}

// Function: capacity
template <typename T, size_t S>
size_t ObjectPool<T, S>::capacity() const {
  
  size_t n = 0;
  
  // global heap
  for(auto p=_gheap.list.next; p!=&_gheap.list; p=p->next) {  
    n += M;
  };

  // local heap
  for(auto& h : _lheaps) {
    n += h.a;
  }

  return n;
}

// Function: num_available_objects
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_available_objects() const {

  size_t n = 0;
  
  // global heap
  for(auto p=_gheap.list.next; p!=&_gheap.list; p=p->next) {  
    n += (M - _block_of(p)->u);
  };

  // local heap
  for(auto& h : _lheaps) {
    n += (h.a - h.u);
  }
  return n;
}

// Function: num_allocated_objects
template <typename T, size_t S>
size_t ObjectPool<T, S>::num_allocated_objects() const {
  
  size_t n = 0;
  
  // global heap
  for(auto p=_gheap.list.next; p!=&_gheap.list; p=p->next) {  
    n += _block_of(p)->u;
  };

  // local heap
  for(auto& h : _lheaps) {
    n += h.u;
  }
  return n;
}

// Function: _bin
template <typename T, size_t S>
size_t ObjectPool<T, S>::_bin(size_t u) const {
  return u == M ? F : u/W;
}

// Function: _offset_in_class
template <typename T, size_t S>
template <class P, class Q>
constexpr size_t ObjectPool<T, S>::_offset_in_class(
  const Q P::*member) const {
  return (size_t) &( reinterpret_cast<P*>(0)->*member);
}

// C macro: parent_class_of(list_pointer, Block, list)
// C++: parent_class_of(list_pointer,  &Block::list)
template <typename T, size_t S>
template <class P, class Q>
constexpr P* ObjectPool<T, S>::_parent_class_of(
  Q* ptr, const Q P::*member
) {
  return (P*)( (char*)ptr - _offset_in_class(member));
}

// Function: _parent_class_of
template <typename T, size_t S>
template <class P, class Q>
constexpr P* ObjectPool<T, S>::_parent_class_of(
  const Q* ptr, const Q P::*member
) const {
  return (P*)( (char*)ptr - _offset_in_class(member));
}

// Function: _block_of
template <typename T, size_t S>
constexpr typename ObjectPool<T, S>::Block* 
ObjectPool<T, S>::_block_of(Blocklist* list) {
  return _parent_class_of(list, &Block::list_node);
}

// Function: _block_of
template <typename T, size_t S>
constexpr typename ObjectPool<T, S>::Block* 
ObjectPool<T, S>::_block_of(const Blocklist* list) const {
  return _parent_class_of(list, &Block::list_node);
}

// Procedure: initialize a list head
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_init_head(Blocklist *list) {
  list->next = list;
  list->prev = list;
}

// Procedure: _blocklist_add_impl
// Insert a new entry between two known consecutive entries.
// 
// This is only for internal list manipulation where we know
// the prev/next entries already!
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_add_impl(
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
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_push_front(
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
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_push_back(
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
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_del_impl(
  Blocklist * prev, Blocklist * next
) {
  next->prev = prev;
  prev->next = next;
}

// _blocklist_del - deletes entry from list.
// @entry: the element to delete from the list.
// Note: list_empty() on entry does not return true after this, the entry is
// in an undefined state.
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_del(Blocklist *entry) {
  _blocklist_del_impl(entry->prev, entry->next);
  entry->next = nullptr;
  entry->prev = nullptr;
}

// list_replace - replace old entry by new one
// @old : the element to be replaced
// @curr : the new element to insert
// 
// If @old was empty, it will be overwritten.
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_replace(
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
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_move_front(
  Blocklist *list, Blocklist *head
) {
  _blocklist_del_impl(list->prev, list->next);
  _blocklist_push_front(list, head);
}

// list_move_tail - delete from one list and add as another's tail
// @list: the entry to move
// @head: the head that will follow our entry
template <typename T, size_t S>
void ObjectPool<T, S>::_blocklist_move_back(
  Blocklist *list, Blocklist *head
) {
  _blocklist_del_impl(list->prev, list->next);
  _blocklist_push_back(list, head);
}

// list_is_first - tests whether @list is the last entry in list @head
// @list: the entry to test
// @head: the head of the list
template <typename T, size_t S>
bool ObjectPool<T, S>::_blocklist_is_first(
  const Blocklist *list, const Blocklist *head
) {
  return list->prev == head;
}

// list_is_last - tests whether @list is the last entry in list @head
// @list: the entry to test
// @head: the head of the list
template <typename T, size_t S>
bool ObjectPool<T, S>::_blocklist_is_last(
  const Blocklist *list, const Blocklist *head
) {
  return list->next == head;
}

// list_empty - tests whether a list is empty
// @head: the list to test.
template <typename T, size_t S>
bool ObjectPool<T, S>::_blocklist_is_empty(const Blocklist *head) {
  return head->next == head;
}

// list_is_singular - tests whether a list has just one entry.
// @head: the list to test.
template <typename T, size_t S>
bool ObjectPool<T, S>::_blocklist_is_singular(
  const Blocklist *head
) {
  return !_blocklist_is_empty(head) && (head->next == head->prev);
}

// Procedure: _for_each_block
template <typename T, size_t S>
template <typename C>
void ObjectPool<T, S>::_for_each_block(Blocklist* head, C&& c) {
  Blocklist* p;
  for(p=head->next; p!=head; p=p->next) {
    c(_block_of(p));
  }
}
      
// Procedure: _for_each_block_safe
// Iterate each item of a list - safe to free
template <typename T, size_t S>
template <typename C>
void ObjectPool<T, S>::_for_each_block_safe(Blocklist* head, C&& c) {
  Blocklist* p;
  Blocklist* t;
  for(p=head->next, t=p->next; p!=head; p=t, t=p->next) {
    c(_block_of(p));
  }
}

// Function: _allocate
// allocate a spot from the block
template <typename T, size_t S>
T* ObjectPool<T, S>::_allocate(Block* s) {
  if(s->top == nullptr) {
    return reinterpret_cast<T*>(s->data + s->i++ * X);
  }
  else {
    T* retval = s->top;
    s->top = *(reinterpret_cast<T**>(s->top));
    return retval;
  }
}

// Procedure: _deallocate
template <typename T, size_t S>
void ObjectPool<T, S>::_deallocate(Block* s, T* ptr) {
  *(reinterpret_cast<T**>(ptr)) = s->top;
  s->top = ptr;
}

// Function: allocate
template <typename T, size_t S>
template <typename... ArgsT>
T* ObjectPool<T, S>::animate(ArgsT&&... args) {

  //std::cout << "construct a new item\n";
    
  // my logically mapped heap
  LocalHeap& h = _this_heap(); 
  
  Block* s {nullptr};

  h.mutex.lock();
  
  // scan the list of superblocks from most full to least
  int f = static_cast<int>(F-1);
  for(; f>=0; f--) {
    if(!_blocklist_is_empty(&h.lists[f])) {
      s = _block_of(h.lists[f].next);
      break;
    }
  }
  
  // no superblock found
  if(f == -1) {

    // check heap 0 for a superblock
    _gheap.mutex.lock();
    if(!_blocklist_is_empty(&_gheap.list)) {
      
      s = _block_of(_gheap.list.next);
      
      //printf("get a superblock from global heap %lu\n", s->u);
      assert(s->u < M && s->heap == nullptr);
      f = static_cast<int>(_bin(s->u + 1));

      _blocklist_move_front(&s->list_node, &h.lists[f]);

      s->heap = &h;  // must be within the global heap lock
      _gheap.mutex.unlock();

      h.u = h.u + s->u;
      h.a = h.a + M;
    }
    // create a new block
    else {
      //printf("create a new superblock\n");
      _gheap.mutex.unlock();
      f = 0;
      s = static_cast<Block*>(std::malloc(sizeof(Block)));

      if(s == nullptr) {
        throw std::bad_alloc();
      }

      s->heap = &h;
      s->i = 0;
      s->u = 0;
      s->top = nullptr;

      _blocklist_push_front(&s->list_node, &h.lists[f]);

      h.a = h.a + M;
    }
  }
  
  // the superblock must have at least one space
  //assert(s->u < M);
  //printf("%lu %lu %lu\n", h.u, h.a, s->u);
  //assert(h.u < h.a);

  h.u = h.u + 1;
  s->u = s->u + 1;

  // take one item from the superblock
  T* mem = _allocate(s);
  
  int b = static_cast<int>(_bin(s->u));
  
  if(b != f) {
    //printf("move superblock from list[%d] to list[%d]\n", f, b);
    _blocklist_move_front(&s->list_node, &h.lists[b]);
  }

  //std::cout << "s.i " << s->i << '\n'
  //          << "s.u " << s->u << '\n'
  //          << "h.u " << h.u  << '\n'
  //          << "h.a " << h.a  << '\n';

  h.mutex.unlock();

  //printf("allocate %p (s=%p)\n", mem, s);

  new (mem) T(std::forward<ArgsT>(args)...);

  mem->_object_pool_block = s;

  return mem;
}
  
// Function: destruct
template <typename T, size_t S>
void ObjectPool<T, S>::recycle(T* mem) {

  //Block* s = *reinterpret_cast<Block**>(
  //  reinterpret_cast<char*>(mem) - sizeof(Block**)
  //);

  //Block* s= *(reinterpret_cast<Block**>(mem) - O); //  (mem) - 1

  Block* s = static_cast<Block*>(mem->_object_pool_block);

  mem->~T();
  
  //printf("deallocate %p (s=%p) M=%lu W=%lu X=%lu\n", mem, s, M, W, X);

  // here we need a loop because when we lock the heap,
  // other threads may have removed the superblock to another heap
  bool sync = false;

  do {
    auto h = s->heap;    
    
    // the block is in global heap
    if(h == nullptr) {
      std::lock_guard<std::mutex> glock(_gheap.mutex);
      if(s->heap == h) {
        sync = true;
        _deallocate(s, mem);
        s->u = s->u - 1;
      }
    }
    else {
      std::lock_guard<std::mutex> llock(h->mutex);
      if(s->heap == h) {
        sync = true;
        // deallocate the item from the superblock
        size_t f = _bin(s->u);
        _deallocate(s, mem);
        s->u = s->u - 1;
        h->u = h->u - 1;

        size_t b = _bin(s->u);

        if(b != f) {
          //printf("move superblock from list[%d] to list[%d]\n", f, b);
          _blocklist_move_front(&s->list_node, &h->lists[b]);
        }

        // transfer a mostly-empty superblock to global heap
        if((h->u + K*M < h->a) && (h->u < ((F-1) * h->a / F))) {
          for(size_t i=0; i<F; i++) {
            if(!_blocklist_is_empty(&h->lists[i])) {
              Block* x = _block_of(h->lists[i].next);
              //printf("transfer a block (x.u=%lu/x.i=%lu) to the global heap\n", x->u, x->i);
              assert(h->u > x->u && h->a > M);
              h->u = h->u - x->u;
              h->a = h->a - M;
              x->heap = nullptr;
              std::lock_guard<std::mutex> glock(_gheap.mutex);
              _blocklist_move_front(&x->list_node, &_gheap.list);
              break;
            }
          }
        }
      }
    }
  } while(!sync);
  
  //std::cout << "s.i " << s->i << '\n'
  //          << "s.u " << s->u << '\n';
}
    
// Function: _this_heap
template <typename T, size_t S>
typename ObjectPool<T, S>::LocalHeap& 
ObjectPool<T, S>::_this_heap() {
  // here we don't use thread local since object pool might be
  // created and destroyed multiple times
  //thread_local auto hv = std::hash<std::thread::id>()(std::this_thread::get_id());
  //return _lheaps[hv & _lheap_mask];

  return _lheaps[
    std::hash<std::thread::id>()(std::this_thread::get_id()) & _lheap_mask
  ];
}

// Function: _next_pow2
template <typename T, size_t S>
constexpr unsigned ObjectPool<T, S>::_next_pow2(unsigned n) const { 
  if(n == 0) return 1;
  n--; 
  n |= n >> 1; 
  n |= n >> 2; 
  n |= n >> 4; 
  n |= n >> 8; 
  n |= n >> 16; 
  n++; 
  return n; 
}  

}  // end namespace tf --------------------------------------------------------
