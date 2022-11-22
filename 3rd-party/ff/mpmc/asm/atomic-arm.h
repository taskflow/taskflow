/* Mauro Mulatero:
 *  This is a slightly modified version of the linux kernel file
 *   /<source-dir>/include/asm-arm/atomic.h
 *
 */
#ifndef FF_ASM_ARM_ATOMIC_H
#define FF_ASM_ARM_ATOMIC_H

typedef struct { volatile int counter; } atomic_t;

#define ATOMIC_INIT(i)  { (i) }

#define atomic_read(v)  ((v)->counter)

/*
 * ARMv6 UP and SMP safe atomic ops.  We use load exclusive and
 * store exclusive to ensure that these are atomic.  We may loop
 * to ensure that the update happens.  Writing to 'v->counter'
 * without using the following operations WILL break the atomic
 * nature of these ops.
 */
static inline void atomic_set(atomic_t *v, int i)
{
  unsigned long tmp;

        __asm__ __volatile__("@ atomic_set\n"
"1:     ldrex   %0, [%1]\n"               // load ex: tmp <- v->counter
"       strex   %0, %2, [%1]\n"           // store ex: tmp(status), i -> v->counter ; tmp=0 se tutto ok
"       teq     %0, #0\n"                 // controlla se tmp contiene 0 e loop se !=
"       bne     1b"
        : "=&r" (tmp)                     // output: %0
        : "r" (&v->counter), "r" (i)      // input:  %1, %2
        : "cc", "memory");
}

static inline int atomic_add_return(int i, atomic_t *v)
{
  unsigned long tmp;
  int result;

        __asm__ __volatile__("@ atomic_add_return\n"
"1:     ldrex   %0, [%2]\n"       // load ex: result <- v->counter
"       add     %0, %0, %3\n"     // result = result + i
"       strex   %1, %0, [%2]\n"   // store ex: tmp(status), result -> v->counter ; tmp=0 se tutto ok
"       teq     %1, #0\n"         // controlla se tmp contiene 0 e loop se !=
"       bne     1b"
        : "=&r" (result), "=&r" (tmp)   // output: %0, %1
        : "r" (&v->counter), "Ir" (i)   // input:  %2, %3
        : "cc", "memory");

  return result;
}

static inline int atomic_sub_return(int i, atomic_t *v)
{
  unsigned long tmp;
  int result;

        __asm__ __volatile__("@ atomic_sub_return\n"
"1:     ldrex   %0, [%2]\n"
"       sub     %0, %0, %3\n"
"       strex   %1, %0, [%2]\n"
"       teq     %1, #0\n"
"       bne     1b"
        : "=&r" (result), "=&r" (tmp)
        : "r" (&v->counter), "Ir" (i)
        : "cc", "memory");

  return result;
}

/*****
  oldval = ptr->counter;
  if(oldval == old)
    ptr->counter = New;
 ****/
static inline int atomic_cmpxchg(atomic_t *ptr, int old, int New)
{
  unsigned long oldval, res;

  do {
    __asm__ __volatile__("@ atomic_cmpxchg\n"
    "ldrex  %1, [%2]\n"
    "mov    %0, #0\n"
    "teq    %1, %3\n"
    "strexeq %0, %4, [%2]\n"
        : "=&r" (res), "=&r" (oldval)
        : "r" (&ptr->counter), "Ir" (old), "r" (New)
        : "cc", "memory");
  } while (res);

  return oldval;
}

#define atomic_xchg(v, New) (xchg(&((v)->counter), New))

static inline int atomic_add_unless(atomic_t *v, int a, int u)
{
  int c, old;

  c = atomic_read(v);
  while (c != u && (old = atomic_cmpxchg((v), c, c + a)) != c)
    c = old;
  return c != u;
}

#define atomic_inc_not_zero(v) atomic_add_unless((v), 1, 0)

#define atomic_add(i, v)        (void) atomic_add_return(i, v)
#define atomic_inc(v)           (void) atomic_add_return(1, v)
#define atomic_sub(i, v)        (void) atomic_sub_return(i, v)
#define atomic_dec(v)           (void) atomic_sub_return(1, v)

#define atomic_inc_and_test(v)  (atomic_add_return(1, v) == 0)
#define atomic_dec_and_test(v)  (atomic_sub_return(1, v) == 0)
#define atomic_inc_return(v)    (atomic_add_return(1, v))
#define atomic_dec_return(v)    (atomic_sub_return(1, v))
#define atomic_sub_and_test(i, v) (atomic_sub_return(i, v) == 0)

#define atomic_add_negative(i,v) (atomic_add_return(i, v) < 0)

/* Atomic operations are already serializing on ARM */
#define smp_mb__before_atomic_dec()     barrier()
#define smp_mb__after_atomic_dec()      barrier()
#define smp_mb__before_atomic_inc()     barrier()
#define smp_mb__after_atomic_inc()      barrier()

#endif /* FF_ASM_ARM_ATOMIC_H */
