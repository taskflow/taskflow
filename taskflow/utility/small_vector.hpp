// small vector modified from llvm

#pragma once

#include "macros.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>


/**
@file small_vector.hpp
@brief small vector include file
*/

namespace tf { namespace detail {

/**
@private
@brief NextCapacity - Returns the next power of two (in 64-bits)
       that is strictly greater than A.  Returns zero on overflow.
       this function assumes A to be positive
*/
inline uint64_t NextCapacity(uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

}}  // end of namespace tf::detail --------------------------------------------


namespace tf {

/**
@private
*/
template <typename T>
struct IsPod : std::integral_constant<bool, std::is_standard_layout<T>::value &&
                                            std::is_trivial<T>::value> {};

/**
@private
*/
class SmallVectorBase {
protected:
  void *BeginX, *EndX, *CapacityX;

protected:
  SmallVectorBase(void *FirstEl, size_t Size)
    : BeginX(FirstEl), EndX(FirstEl), CapacityX((char*)FirstEl+Size) {}

  /// This is an implementation of the grow() method which only works
  /// on POD-like data types and is out of line to reduce code duplication.
  void grow_pod(void *FirstEl, size_t MinSizeInBytes, size_t TSize){
    size_t CurSizeBytes = size_in_bytes();
    size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.
    if (NewCapacityInBytes < MinSizeInBytes) {
      NewCapacityInBytes = MinSizeInBytes;
    }

    void *NewElts;
    if (BeginX == FirstEl) {
      NewElts = std::malloc(NewCapacityInBytes);

      // Copy the elements over.  No need to run dtors on PODs.
      memcpy(NewElts, this->BeginX, CurSizeBytes);
    } else {
      // If this wasn't grown from the inline copy, grow the allocated space.
      NewElts = realloc(this->BeginX, NewCapacityInBytes);
    }
    //assert(NewElts && "Out of memory");

    this->EndX = (char*)NewElts+CurSizeBytes;
    this->BeginX = NewElts;
    this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
  }

public:
  /// This returns size()*sizeof(T).
  size_t size_in_bytes() const {
    return size_t((char*)EndX - (char*)BeginX);
  }

  /// capacity_in_bytes - This returns capacity()*sizeof(T).
  size_t capacity_in_bytes() const {
    return size_t((char*)CapacityX - (char*)BeginX);
  }

  bool empty() const { return BeginX == EndX; }
};

/**
@private
*/
template <typename T, unsigned N> struct SmallVectorStorage;

/**
@private
*/
template <typename T, typename = void>
class SmallVectorTemplateCommon : public SmallVectorBase {

  private:
  template <typename, unsigned> friend struct SmallVectorStorage;

  template <typename X>
  struct AlignedUnionType {
    alignas(X) std::byte buff[std::max(sizeof(std::byte), sizeof(X))];
  };

  // Allocate raw space for N elements of type T.  If T has a ctor or dtor, we
  // don't want it to be automatically run, so we need to represent the space as
  // something else.  Use an array of char of sufficient alignment.
  
  // deprecated in c++23
  //typedef typename std::aligned_union<1, T>::type U;
  typedef AlignedUnionType<T> U;

  U FirstEl;
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

  protected:
  SmallVectorTemplateCommon(size_t Size) : SmallVectorBase(&FirstEl, Size) {}

  void grow_pod(size_t MinSizeInBytes, size_t TSize) {
    SmallVectorBase::grow_pod(&FirstEl, MinSizeInBytes, TSize);
  }

  /// Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return BeginX == static_cast<const void*>(&FirstEl);
  }

  /// Put this vector in a state of being small.
  void resetToSmall() {
    BeginX = EndX = CapacityX = &FirstEl;
  }

  void setEnd(T *P) { this->EndX = P; }

  public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef T *iterator;
  typedef const T *const_iterator;

  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;

  typedef T &reference;
  typedef const T &const_reference;
  typedef T *pointer;
  typedef const T *const_pointer;

  // forward iterator creation methods.
  inline iterator begin() { return (iterator)this->BeginX; }
  inline const_iterator begin() const { return (const_iterator)this->BeginX; }
  inline iterator end() { return (iterator)this->EndX; }
  inline const_iterator end() const { return (const_iterator)this->EndX; }

  protected:

  iterator capacity_ptr() { return (iterator)this->CapacityX; }
  const_iterator capacity_ptr() const { return (const_iterator)this->CapacityX;}

  public:

  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin());}

  inline size_type size() const { return end()-begin(); }
  inline size_type max_size() const { return size_type(-1) / sizeof(T); }

  /// Return the total number of elements in the currently allocated buffer.
  size_t capacity() const { return capacity_ptr() - begin(); }

  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() { return pointer(begin()); }
  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const { return const_pointer(begin()); }

  inline reference operator[](size_type idx) {
    //assert(idx < size());
    return begin()[idx];
  }

  inline const_reference operator[](size_type idx) const {
    //assert(idx < size());
    return begin()[idx];
  }

  reference front() {
    //assert(!empty());
    return begin()[0];
  }

  const_reference front() const {
    //assert(!empty());
    return begin()[0];
  }

  reference back() {
    //assert(!empty());
    return end()[-1];
  }

  const_reference back() const {
    //assert(!empty());
    return end()[-1];
  }
};

/**
@private
*/
template <typename T, bool isPodLike>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {

protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Move the range [I, E) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(std::make_move_iterator(I),
                            std::make_move_iterator(E), Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  /// Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

public:
  void push_back(const T &Elt) {
    if (TF_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(Elt);
    this->setEnd(this->end()+1);
  }

  void push_back(T &&Elt) {
    if (TF_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(::std::move(Elt));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
    this->end()->~T();
  }
};

/**
@private
*/
template <typename T, bool isPodLike>
void SmallVectorTemplateBase<T, isPodLike>::grow(size_t MinSize) {
  size_t CurCapacity = this->capacity();
  size_t CurSize = this->size();
  // Always grow, even from zero.
  size_t NewCapacity = size_t(tf::detail::NextCapacity(CurCapacity+2));
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;
  T *NewElts = static_cast<T*>(std::malloc(NewCapacity*sizeof(T)));

  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());

  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall())
    std::free(this->begin());

  this->setEnd(NewElts+CurSize);
  this->BeginX = NewElts;
  this->CapacityX = this->begin()+NewCapacity;
}

/**
@private
*/
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T *, T *) {}

  /// Move the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1 *I, T1 *E, T2 *Dest,
      typename std::enable_if<std::is_same<typename std::remove_const<T1>::type,
                                           T2>::value>::type * = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes SmallVector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that I and E are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (I != E)
      memcpy(Dest, I, (E - I) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) {
    this->grow_pod(MinSize*sizeof(T), sizeof(T));
  }
public:
  void push_back(const T &Elt) {
    if (TF_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    memcpy(this->end(), &Elt, sizeof(T));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
  }
};

/**
@private
*/
template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T, IsPod<T>::value> {
  typedef SmallVectorTemplateBase<T, IsPod<T>::value> SuperClass;

  SmallVectorImpl(const SmallVectorImpl&) = delete;

public:
  typedef typename SuperClass::iterator iterator;
  typedef typename SuperClass::const_iterator const_iterator;
  typedef typename SuperClass::size_type size_type;

protected:
  // Default ctor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N)
    : SmallVectorTemplateBase<T, IsPod<T>::value>(N*sizeof(T)) {
  }

public:
  ~SmallVectorImpl() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());

    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall())
      std::free(this->begin());
  }


  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->EndX = this->BeginX;
  }

  void resize(size_type N) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        new (&*I) T();
      this->setEnd(this->begin()+N);
    }
  }

  void resize(size_type N, const T &NV) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      std::uninitialized_fill(this->end(), this->begin()+N, NV);
      this->setEnd(this->begin()+N);
    }
  }

  void reserve(size_type N) {
    if (this->capacity() < N)
      this->grow(N);
  }

  T pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(SmallVectorImpl &RHS);

  /// Add the specified range to the end of the SmallVector.
  template<typename in_iter>
  void append(in_iter in_start, in_iter in_end) {
    size_type NumInputs = std::distance(in_start, in_end);
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    this->uninitialized_copy(in_start, in_end, this->end());
    this->setEnd(this->end() + NumInputs);
  }

  /// Add the specified range to the end of the SmallVector.
  void append(size_type NumInputs, const T &Elt) {
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    std::uninitialized_fill_n(this->end(), NumInputs, Elt);
    this->setEnd(this->end() + NumInputs);
  }

  void append(std::initializer_list<T> IL) {
    append(IL.begin(), IL.end());
  }

  void assign(size_type NumElts, const T &Elt) {
    clear();
    if (this->capacity() < NumElts)
      this->grow(NumElts);
    this->setEnd(this->begin()+NumElts);
    std::uninitialized_fill(this->begin(), this->end(), Elt);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  iterator erase(const_iterator CI) {
    // Just cast away constness because this is a non-const member function.
    iterator I = const_cast<iterator>(CI);

    //assert(I >= this->begin() && "Iterator to erase is out of bounds.");
    //assert(I < this->end() && "Erasing at past-the-end iterator.");

    iterator N = I;
    // Shift all elts down one.
    std::move(I+1, this->end(), I);
    // Drop the last elt.
    this->pop_back();
    return(N);
  }

  iterator erase(const_iterator CS, const_iterator CE) {
    // Just cast away constness because this is a non-const member function.
    iterator S = const_cast<iterator>(CS);
    iterator E = const_cast<iterator>(CE);

    //assert(S >= this->begin() && "Range to erase is out of bounds.");
    //assert(S <= E && "Trying to erase invalid range.");
    //assert(E <= this->end() && "Trying to erase past the end.");

    iterator N = S;
    // Shift all elts down.
    iterator I = std::move(E, this->end(), S);
    // Drop the last elts.
    this->destroy_range(I, this->end());
    this->setEnd(I);
    return(N);
  }

  iterator insert(iterator I, T &&Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(::std::move(Elt));
      return this->end()-1;
    }

    //assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    //assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }

    ::new ((void*) this->end()) T(::std::move(this->back()));
    // Push everything else over.
    std::move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = ::std::move(*EltPtr);
    return I;
  }

  iterator insert(iterator I, const T &Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(Elt);
      return this->end()-1;
    }

    //assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    //assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }
    ::new ((void*) this->end()) T(std::move(this->back()));
    // Push everything else over.
    std::move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    const T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = *EltPtr;
    return I;
  }

  iterator insert(iterator I, size_type NumToInsert, const T &Elt) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(NumToInsert, Elt);
      return this->begin()+InsertElt;
    }

    //assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    //assert(I <= this->end() && "Inserting past the end of the vector.");

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::fill_n(I, NumToInsert, Elt);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    std::fill_n(I, NumOverwritten, Elt);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert-NumOverwritten, Elt);
    return I;
  }

  template<typename ItTy>
  iterator insert(iterator I, ItTy From, ItTy To) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(From, To);
      return this->begin()+InsertElt;
    }

    //assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    //assert(I <= this->end() && "Inserting past the end of the vector.");

    size_t NumToInsert = std::distance(From, To);

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::copy(From, To, I);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    for (T *J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J; ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  template <typename... ArgTypes> void emplace_back(ArgTypes &&... Args) {
    if (TF_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void *)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->setEnd(this->end() + 1);
  }

  SmallVectorImpl &operator=(const SmallVectorImpl &RHS);

  SmallVectorImpl &operator=(SmallVectorImpl &&RHS);

  bool operator==(const SmallVectorImpl &RHS) const {
    if (this->size() != RHS.size()) return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }
  bool operator!=(const SmallVectorImpl &RHS) const {
    return !(*this == RHS);
  }

  bool operator<(const SmallVectorImpl &RHS) const {
    return std::lexicographical_compare(this->begin(), this->end(),
                                        RHS.begin(), RHS.end());
  }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing elements
  /// which will only be overwritten.
  void set_size(size_type N) {
    //assert(N <= this->capacity());
    this->setEnd(this->begin() + N);
  }
};


template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T> &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither vector is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->EndX, RHS.EndX);
    std::swap(this->CapacityX, RHS.CapacityX);
    return;
  }
  if (RHS.size() > this->capacity())
    this->grow(RHS.size());
  if (this->size() > RHS.capacity())
    RHS.grow(this->size());

  // Swap the shared elements.
  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i)
    std::swap((*this)[i], RHS[i]);

  // Copy over the extra elts.
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin()+NumShared, this->end(), RHS.end());
    RHS.setEnd(RHS.end()+EltDiff);
    this->destroy_range(this->begin()+NumShared, this->end());
    this->setEnd(this->begin()+NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin()+NumShared, RHS.end(), this->end());
    this->setEnd(this->end() + EltDiff);
    this->destroy_range(RHS.begin()+NumShared, RHS.end());
    RHS.setEnd(RHS.begin()+NumShared);
  }
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::
  operator=(const SmallVectorImpl<T> &RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin()+RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // Destroy excess elements.
    this->destroy_range(NewEnd, this->end());

    // Trim.
    this->setEnd(NewEnd);
    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: don't do this if they're efficiently moveable.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::copy(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Copy construct the new elements in place.
  this->uninitialized_copy(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::operator=(SmallVectorImpl<T> &&RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If the RHS isn't small, clear this vector and then steal its buffer.
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall()) std::free(this->begin());
    this->BeginX = RHS.BeginX;
    this->EndX = RHS.EndX;
    this->CapacityX = RHS.CapacityX;
    RHS.resetToSmall();
    return *this;
  }

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = this->begin();
    if (RHSSize)
      NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    // Destroy excess elements and trim the bounds.
    this->destroy_range(NewEnd, this->end());
    this->setEnd(NewEnd);

    // Clear the RHS.
    RHS.clear();

    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: this may not actually make any sense if we can efficiently move
  // elements.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::move(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Move-construct the new elements in place.
  this->uninitialized_move(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);

  RHS.clear();
  return *this;
}

/**
@private
*/
template <typename T, unsigned N>
struct SmallVectorStorage {
  /**
  @private
  */
  typename SmallVectorTemplateCommon<T>::U InlineElts[N - 1];
};

/**
@private
*/
template <typename T> struct SmallVectorStorage<T, 1> {};

/**
@private
*/
template <typename T> struct SmallVectorStorage<T, 0> {};

/**
@brief class to define a vector optimized for small array

@tparam T data type
@tparam N threshold of the number of elements in the initial storage

The class defines a C++ STL-styled vector (a variable-sized array)
optimized for the case when the array is small.
It contains some number of elements in-place,
which allows it to avoid heap allocation when the actual number of
elements is below that threshold. This allows normal @em small cases to be
fast without losing generality for large inputs.
All the methods in [std::vector](https://en.cppreference.com/w/cpp/container/vector)
can apply to this class.

The class is stripped from the LLVM codebase.
*/
template <typename T, unsigned N = 2>
class SmallVector : public SmallVectorImpl<T> {
  /// Inline space for elements which aren't stored in the base class.
  SmallVectorStorage<T, N> Storage;

public:

  /**
  @brief constructs an empty vector
  */
  SmallVector() : SmallVectorImpl<T>(N) {
  }

  /**
  @brief constructs a vector with @c Size copies of elements with value @c value
  */
  explicit SmallVector(size_t Size, const T &Value = T())
    : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  /**
  @brief constructs a vector with the contents of the range
         <tt>[S, E)</tt>
   */
  template<typename ItTy>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  //template <typename RangeTy>
  //explicit SmallVector(const tf::iterator_range<RangeTy> &R)
  //    : SmallVectorImpl<T>(N) {
  //  this->append(R.begin(), R.end());
  //}

  /**
  @brief constructs a vector with the contents of the initializer list @c IL
  */
  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  /**
  @brief constructs the vector with the copy of the contents of @c RHS
  */
  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }

  /**
  @brief constructs the vector with the contents of @c RHS using move semantics
  */
  SmallVector(SmallVector &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  /**
  @brief replaces the contents with a copy of the contents of @c RHS
  */
  const SmallVector &operator=(const SmallVector &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  /**
  @brief replaces the contents with the contents of @c RHS using move semantics
  */
  const SmallVector &operator=(SmallVector &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  /**
  @brief constructs a vector with the contents of @c RHS using move semantics
  */
  SmallVector(SmallVectorImpl<T> &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  /**
  @brief replaces the contents with the contents of @c RHS using move semantics
   */
  const SmallVector &operator=(SmallVectorImpl<T> &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  /**
  @brief replaces the contents with the copy of the contents of an initializer list @c IL
   */
  const SmallVector &operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template<typename T, unsigned N>
static inline size_t capacity_in_bytes(const SmallVector<T, N> &X) {
  return X.capacity_in_bytes();
}

} // end tf namespace ---------------------------------------------------------

namespace std {
  /// Implement std::swap in terms of SmallVector swap.
  template<typename T>
  inline void
  swap(tf::SmallVectorImpl<T> &LHS, tf::SmallVectorImpl<T> &RHS) {
    LHS.swap(RHS);
  }

  /// Implement std::swap in terms of SmallVector swap.
  template<typename T, unsigned N>
  inline void
  swap(tf::SmallVector<T, N> &LHS, tf::SmallVector<T, N> &RHS) {
    LHS.swap(RHS);
  }
}  // end of namespace std ----------------------------------------------------


