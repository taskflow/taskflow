/* Massimo: 
 *  This is a slightly modified version of the linux kernel file 
 *   /<source-dir>/include/asm-x86/atomic_64.h
 *
 */
#ifndef FF_ARCH_X86_64_ATOMIC_H
#define FF_ARCH_X86_64_ATOMIC_H


/* atomic_t should be 32 bit signed type */


#if !defined(likely)
#define likely(x)	__builtin_expect(!!(x), 1)
#endif
#if !defined(unlikely)
#define unlikely(x)	__builtin_expect(!!(x), 0)
#endif


/*
 * Atomic operations that C can't guarantee us.  Useful for
 * resource counting etc..
 */

#define LOCK_PREFIX "lock ; "


/*
 * Make sure gcc doesn't try to be clever and move things around
 * on us. We need to use _exactly_ the address the user gave us,
 * not some alias that contains the same information.
 */
typedef struct { volatile unsigned long counter; } atomic_t;

#define ATOMIC_INIT(i)	{ (i) }

/**
 * atomic_read - read atomic variable
 * @v: pointer of type atomic_t
 * 
 * Atomically reads the value of @v.
 */ 
#define atomic_read(v)		((v)->counter)

/**
 * atomic_set - set atomic variable
 * @v: pointer of type atomic_t
 * @i: required value
 * 
 * Atomically sets the value of @v to @i.
 */ 
#define atomic_set(v,i)		(((v)->counter) = (i))


/**
 * atomic_inc - increment atomic variable
 * @v: pointer of type atomic_t
 * 
 * Atomically increments @v by 1.
 */ 
static __inline__ void atomic_inc(atomic_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "incl %0"
		:"=m" (v->counter)
		:"m" (v->counter));
}

/**
 * atomic64_dec - decrement atomic64 variable
 * @v: pointer to type atomic64_t
 *
 * Atomically decrements @v by 1.
 */
static __inline__ void atomic_dec(atomic_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "decq %0"
		: "=m" (v->counter)
		: "m" (v->counter));
}


/* An 64bit atomic type */

typedef struct { volatile unsigned long counter; } atomic64_t;

#define ATOMIC64_INIT(i)	{ (i) }

/**
 * atomic64_read - read atomic64 variable
 * @v: pointer of type atomic64_t
 *
 * Atomically reads the value of @v.
 * Doesn't imply a read memory barrier.
 */
#define atomic64_read(v)		((v)->counter)

/**
 * atomic64_set - set atomic64 variable
 * @v: pointer to type atomic64_t
 * @i: required value
 *
 * Atomically sets the value of @v to @i.
 */
#define atomic64_set(v,i)		(((v)->counter) = (i))


/**
 * atomic64_inc - increment atomic64 variable
 * @v: pointer to type atomic64_t
 *
 * Atomically increments @v by 1.
 */
static __inline__ void atomic64_inc(atomic64_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "incq %0"
		:"=m" (v->counter)
		:"m" (v->counter));
}

#define atomic64_inc_return(v)  (atomic64_add_return(1, (v)))
#define atomic64_dec_return(v)  (atomic64_sub_return(1, (v)))

/**
 * atomic64_dec - decrement atomic64 variable
 * @v: pointer to type atomic64_t
 *
 * Atomically decrements @v by 1.
 */
static __inline__ void atomic64_dec(atomic64_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "decq %0"
		: "=m" (v->counter)
		: "m" (v->counter));
}

static __inline__ void atomic64_add(unsigned long i, atomic64_t *v)
{
       __asm__ __volatile__(
	       LOCK_PREFIX "addq %1,%0"
	       : "=m" (v->counter)
	       : "ir" (i), "m" (v->counter));
}


/**
 * atomic64_add_return - add and return
 * @i: integer value to add
 * @v: pointer to type atomic64_t
 *
 * Atomically adds @i to @v and returns @i + @v
 */
static __inline__ unsigned long atomic64_add_return(unsigned long i, atomic64_t *v)
{
	unsigned long __i = i;
	asm volatile(LOCK_PREFIX "xaddq %0, %1;"
		     : "+r" (i), "+m" (v->counter)
		     : : "memory");
	return i + __i;
}

static __inline__ unsigned long atomic64_sub_return(unsigned long i, atomic64_t *v)
{
	return atomic64_add_return(-i, v);
}

/**
 * atomic64_sub - subtract the atomic64 variable
 * @i: integer value to subtract
 * @v: pointer to type atomic64_t
 *
 * Atomically subtracts @i from @v.
 */
static __inline__ void atomic64_sub(unsigned long i, atomic64_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "subq %1,%0"
		: "=m" (v->counter)
		: "ir" (i), "m" (v->counter));
}


#define atomic64_cmpxchg(v, old, New) (cmpxchg8(&((v)->counter), (old), (New)))


/*
 * Atomic compare and exchange.  Compare OLD with MEM, if identical,
 * store NEW in MEM.  Return the initial value in MEM.  Success is
 * indicated by comparing RETURN with OLD.
 */

#define __xg(x) ((volatile unsigned long *)(x))
static __inline__ unsigned long cmpxchg8(volatile void *ptr, unsigned long old,
					 unsigned long New)
{
	unsigned long prev;
	asm volatile(LOCK_PREFIX "cmpxchgq %1,%2"
		     : "=a"(prev)
		     : "r"(New), "m"(*__xg(ptr)), "0"(old)
		     : "memory");
	return prev;
}


/**
 * atomic64_add_unless - add unless the number is a given value
 * @v: pointer of type atomic64_t
 * @a: the amount to add to v...
 * @u: ...unless v is equal to u.
 *
 * Atomically adds @a to @v, so unsigned long as it was not @u.
 * Returns non-zero if @v was not @u, and zero otherwise.
 */
static inline unsigned long atomic64_add_unless(atomic64_t *v, unsigned long a, unsigned long u)
{
	unsigned long c, old;
	c = atomic64_read(v);
	for (;;) {
		if (unlikely(c == (u)))
			break;
		old = atomic64_cmpxchg((v), c, c + (a));
		if (likely(old == c))
			break;
		c = old;
	}
	return c != (u);
}
#endif /* FF_ARCH_X86_64_ATOMIC_H */
