/* Massimo: 
 *  This is a slightly modified version of the linux kernel file 
 *   /<source-dir>/include/asm-powerpc/atomic.h
 *
 */
#ifndef FF_ASM_POWERPC_ATOMIC_H
#define FF_ASM_POWERPC_ATOMIC_H


/* some defines taken from: */
/* stringify.h */
#define __stringify_1(x)	#x
#define __stringify(x)		__stringify_1(x)

/* asm-compat.h  */
#define __stringify_in_c(...)	#__VA_ARGS__
#define stringify_in_c(...)	__stringify_in_c(__VA_ARGS__) " "

#define PPC405_ERR77(ra,rb)	stringify_in_c(dcbt	ra, rb;)
#define	PPC405_ERR77_SYNC	stringify_in_c(sync;)

/* synch.h */
#ifdef __powerpc64__
#define __SUBARCH_HAS_LWSYNC
#endif

#ifdef __SUBARCH_HAS_LWSYNC
#    define LWSYNC	lwsync
#else
#    define LWSYNC	sync
#endif
#define ISYNC_ON_SMP	"\n\tisync\n"
#define LWSYNC_ON_SMP	__stringify(LWSYNC) "\n"


/*
 * PowerPC atomic operations
 */

typedef struct { unsigned int counter; } atomic_t;


#define ATOMIC_INIT(i)		{ (i) }

static __inline__ int atomic_read(const atomic_t *v)
{
	int t;

	__asm__ __volatile__("lwz%U1%X1 %0,%1" : "=r"(t) : "m"(v->counter));

	return t;
}

static __inline__ void atomic_set(atomic_t *v, int i)
{
	__asm__ __volatile__("stw%U0%X0 %1,%0" : "=m"(v->counter) : "r"(i));
}

static __inline__ void atomic_add(int a, atomic_t *v)
{
	int t;

	__asm__ __volatile__(
"1:	lwarx	%0,0,%3	\n\
	add	%0,%2,%0\n"
	PPC405_ERR77(0,%3)
"	stwcx.	%0,0,%3 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (a), "r" (&v->counter)
	: "cc");
}

static __inline__ int atomic_add_return(int a, atomic_t *v)
{
	int t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%2	\n\
	add	%0,%1,%0\n"
	PPC405_ERR77(0,%2)
"	stwcx.	%0,0,%2 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (a), "r" (&v->counter)
	: "cc", "memory");

	return t;
}

#define atomic_add_negative(a, v)	(atomic_add_return((a), (v)) < 0)

static __inline__ void atomic_sub(int a, atomic_t *v)
{
	int t;

	__asm__ __volatile__(
"1:	lwarx	%0,0,%3	\n\
	subf	%0,%2,%0\n"
	PPC405_ERR77(0,%3)
"	stwcx.	%0,0,%3 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (a), "r" (&v->counter)
	: "cc");
}

static __inline__ int atomic_sub_return(int a, atomic_t *v)
{
	int t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%2	\n\
	subf	%0,%1,%0\n"
	PPC405_ERR77(0,%2)
"	stwcx.	%0,0,%2 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (a), "r" (&v->counter)
	: "cc", "memory");

	return t;
}

static __inline__ void atomic_inc(atomic_t *v)
{
	int t;

	__asm__ __volatile__(
"1:	lwarx	%0,0,%2 \n\
	addic	%0,%0,1\n"
	PPC405_ERR77(0,%2)
"	stwcx.	%0,0,%2 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (&v->counter)
	: "cc");
}

static __inline__ int atomic_inc_return(atomic_t *v)
{
	int t;
__asm__ __volatile__(
		LWSYNC_ON_SMP
"\n\
1:      lwarx   %0,0,%2\n\
        addic   %0,%0,1\n\
        stwcx.  %0,0,%2\n\
        bne-    1b"
		ISYNC_ON_SMP
        : "=&r" (t), "=m" (v->counter)
        : "r" (v), "m" (v->counter)
        : "cc");

	/*
	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%1	\n\
	addic	%0,%0,1\n"
	PPC405_ERR77(0,%1)
"	stwcx.	%0,0,%1 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (&v->counter)
	: "cc", "memory");
	*/
	return t;
}

/*
 * atomic_inc_and_test - increment and test
 * @v: pointer of type atomic_t
 *
 * Atomically increments @v by 1
 * and returns true if the result is zero, or false for all
 * other cases.
 */
#define atomic_inc_and_test(v) (atomic_inc_return(v) == 0)

static __inline__ void atomic_dec(atomic_t *v)
{
	int t;

	__asm__ __volatile__(
"1:	lwarx	%0,0,%2	\n\
	addic	%0,%0,-1\n"
	PPC405_ERR77(0,%2)\
"	stwcx.	%0,0,%2\n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (&v->counter)
	: "cc");
}

static __inline__ int atomic_dec_return(atomic_t *v)
{
	int t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%1	\n\
	addic	%0,%0,-1\n"
	PPC405_ERR77(0,%1)
"	stwcx.	%0,0,%1\n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (&v->counter)
	: "cc", "memory");

	return t;
}

/**
 * atomic_add_unless - add unless the number is a given value
 * @v: pointer of type atomic_t
 * @a: the amount to add to v...
 * @u: ...unless v is equal to u.
 *
 * Atomically adds @a to @v, so long as it was not @u.
 * Returns non-zero if @v was not @u, and zero otherwise.
 */
static __inline__ int atomic_add_unless(atomic_t *v, int a, int u)
{
	int t;

	__asm__ __volatile__ (
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%1	\n\
	cmpw	0,%0,%3 \n\
	beq-	2f \n\
	add	%0,%2,%0 \n"
	PPC405_ERR77(0,%2)
"	stwcx.	%0,0,%1 \n\
	bne-	1b \n"
	ISYNC_ON_SMP
"	subf	%0,%2,%0 \n\
2:"
	: "=&r" (t)
	: "r" (&v->counter), "r" (a), "r" (u)
	: "cc", "memory");

	return t != u;
}

#define atomic_inc_not_zero(v) atomic_add_unless((v), 1, 0)

#define atomic_sub_and_test(a, v)	(atomic_sub_return((a), (v)) == 0)
#define atomic_dec_and_test(v)		(atomic_dec_return((v)) == 0)

/*
 * Atomically test *v and decrement if it is greater than 0.
 * The function returns the old value of *v minus 1, even if
 * the atomic variable, v, was not decremented.
 */
static __inline__ int atomic_dec_if_positive(atomic_t *v)
{
	int t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%1	\n\
	cmpwi	%0,1\n\
	addi	%0,%0,-1\n\
	blt-	2f\n"
	PPC405_ERR77(0,%1)
"	stwcx.	%0,0,%1\n\
	bne-	1b"
	ISYNC_ON_SMP
	"\n\
2:"	: "=&b" (t)
	: "r" (&v->counter)
	: "cc", "memory");

	return t;
}

static __inline__ unsigned long 
__cmpxchg_u32(volatile unsigned int *p, unsigned long old, unsigned long _new)
{
	unsigned int prev;
	
	__asm__ __volatile__ (
	LWSYNC_ON_SMP
"1:	lwarx	%0,0,%2		# __cmpxchg_u32\n\
	cmpw	0,%0,%3\n\
	bne-	2f\n"
	PPC405_ERR77(0,%2)
"	stwcx.	%4,0,%2\n\
	bne-	1b"
	ISYNC_ON_SMP
	"\n\
2:"
	: "=&r" (prev), "+m" (*p)
	: "r" (p), "r" (old), "r" (_new)
	: "cc", "memory");

	return prev;
}

#define atomic_cmpxchg(v, o, n) (__cmpxchg_u32(&((v)->counter), (o), (n)))
#define atomic_xchg(v, n) (xchg(&((v)->counter), n))


#define smp_mb__before_atomic_dec()     smp_mb()
#define smp_mb__after_atomic_dec()      smp_mb()
#define smp_mb__before_atomic_inc()     smp_mb()
#define smp_mb__after_atomic_inc()      smp_mb()

#ifdef __powerpc64__

typedef struct { unsigned long counter; } atomic64_t;

#define ATOMIC64_INIT(i)	{ (i) }

static __inline__ long atomic64_read(const atomic64_t *v)
{
	long t;

	__asm__ __volatile__("ld%U1%X1 %0,%1" : "=r"(t) : "m"(v->counter));

	return t;
}

static __inline__ void atomic64_set(atomic64_t *v, long i)
{
	__asm__ __volatile__("std%U0%X0 %1,%0" : "=m"(v->counter) : "r"(i));
}

static __inline__ void atomic64_add(long a, atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
"1:	ldarx	%0,0,%3	\n\
	add	%0,%2,%0\n\
	stdcx.	%0,0,%3 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (a), "r" (&v->counter)
	: "cc");
}

static __inline__ long atomic64_add_return(long a, atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%2	\n\
	add	%0,%1,%0\n\
	stdcx.	%0,0,%2 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (a), "r" (&v->counter)
	: "cc", "memory");

	return t;
}

#define atomic64_add_negative(a, v)	(atomic64_add_return((a), (v)) < 0)

static __inline__ void atomic64_sub(long a, atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
"1:	ldarx	%0,0,%3	\n\
	subf	%0,%2,%0\n\
	stdcx.	%0,0,%3 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (a), "r" (&v->counter)
	: "cc");
}

static __inline__ long atomic64_sub_return(long a, atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%2	\n\
	subf	%0,%1,%0\n\
	stdcx.	%0,0,%2 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (a), "r" (&v->counter)
	: "cc", "memory");

	return t;
}

static __inline__ void atomic64_inc(atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
"1:	ldarx	%0,0,%2	\n\
	addic	%0,%0,1\n\
	stdcx.	%0,0,%2 \n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (&v->counter)
	: "cc");
}

static __inline__ long atomic64_inc_return(atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%1	\n\
	addic	%0,%0,1\n\
	stdcx.	%0,0,%1 \n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (&v->counter)
	: "cc", "memory");

	return t;
}

/*
 * atomic64_inc_and_test - increment and test
 * @v: pointer of type atomic64_t
 *
 * Atomically increments @v by 1
 * and returns true if the result is zero, or false for all
 * other cases.
 */
#define atomic64_inc_and_test(v) (atomic64_inc_return(v) == 0)

static __inline__ void atomic64_dec(atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
"1:	ldarx	%0,0,%2	\n\
	addic	%0,%0,-1\n\
	stdcx.	%0,0,%2\n\
	bne-	1b"
	: "=&r" (t), "+m" (v->counter)
	: "r" (&v->counter)
	: "cc");
}

static __inline__ long atomic64_dec_return(atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%1	\n\
	addic	%0,%0,-1\n\
	stdcx.	%0,0,%1\n\
	bne-	1b"
	ISYNC_ON_SMP
	: "=&r" (t)
	: "r" (&v->counter)
	: "cc", "memory");

	return t;
}

#define atomic64_sub_and_test(a, v)	(atomic64_sub_return((a), (v)) == 0)
#define atomic64_dec_and_test(v)	(atomic64_dec_return((v)) == 0)

/*
 * Atomically test *v and decrement if it is greater than 0.
 * The function returns the old value of *v minus 1.
 */
static __inline__ long atomic64_dec_if_positive(atomic64_t *v)
{
	long t;

	__asm__ __volatile__(
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%1	\n\
	addic.	%0,%0,-1\n\
	blt-	2f\n\
	stdcx.	%0,0,%1\n\
	bne-	1b"
	ISYNC_ON_SMP
	"\n\
2:"	: "=&r" (t)
	: "r" (&v->counter)
	: "cc", "memory");

	return t;
}

/**
 * atomic64_add_unless - add unless the number is a given value
 * @v: pointer of type atomic64_t
 * @a: the amount to add to v...
 * @u: ...unless v is equal to u.
 *
 * Atomically adds @a to @v, so long as it was not @u.
 * Returns non-zero if @v was not @u, and zero otherwise.
 */
static __inline__ int atomic64_add_unless(atomic64_t *v, long a, long u)
{
	long t;

	__asm__ __volatile__ (
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%1	\n\
	cmpd	0,%0,%3 \n\
	beq-	2f \n\
	add	%0,%2,%0 \n"
"	stdcx.	%0,0,%1 \n\
	bne-	1b \n"
	ISYNC_ON_SMP
"	subf	%0,%2,%0 \n\
2:"
	: "=&r" (t)
	: "r" (&v->counter), "r" (a), "r" (u)
	: "cc", "memory");

	return t != u;
}

#define atomic64_inc_not_zero(v) atomic64_add_unless((v), 1, 0)


static __inline__ unsigned long
__cmpxchg_u64(volatile unsigned long *p, unsigned long old, unsigned long _new)
{
	unsigned long prev;

	__asm__ __volatile__ (
	LWSYNC_ON_SMP
"1:	ldarx	%0,0,%2		# __cmpxchg_u64\n\
	cmpd	0,%0,%3\n\
	bne-	2f\n\
	stdcx.	%4,0,%2\n\
	bne-	1b"
	ISYNC_ON_SMP
	"\n\
2:"
	: "=&r" (prev), "+m" (*p)
	: "r" (p), "r" (old), "r" (_new)
	: "cc", "memory");

	return prev;
}

#define atomic64_cmpxchg(v, o, n) (__cmpxchg_u64(&((v)->counter), (o), (n)))
#define atomic64_xchg(v, n) (xchg(&((v)->counter), n))

#endif /* __powerpc64__ */


#endif /* FF_ASM_POWERPC_ATOMIC_H */
