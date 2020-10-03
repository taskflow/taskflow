/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "harness_defs.h"

//Concurrency scheduler is not supported by Windows* new UI apps
//TODO: check whether we can test anything here
#include "tbb/tbb_config.h"
#if !__TBB_WIN8UI_SUPPORT
#ifndef TBBTEST_USE_TBB
    #define TBBTEST_USE_TBB 1
#endif
#else
    #define TBBTEST_USE_TBB 0
    #undef __TBB_TASK_GROUP_CONTEXT
    #define __TBB_TASK_GROUP_CONTEXT 0
#endif

#if !TBBTEST_USE_TBB
    #if defined(_MSC_VER) && _MSC_VER < 1600
        #ifdef TBBTEST_USE_TBB
            #undef TBBTEST_USE_TBB
        #endif
        #define TBBTEST_USE_TBB 1
    #endif
#endif

#if TBBTEST_USE_TBB

    #include "tbb/compat/ppl.h"
    #include "tbb/task_scheduler_init.h"

    #if _MSC_VER
        typedef tbb::internal::uint32_t uint_t;
    #else
        typedef uint32_t uint_t;
    #endif

#else /* !TBBTEST_USE_TBB */

    #if defined(_MSC_VER)
    #pragma warning(disable: 4100 4180)
    #endif

    #include <ppl.h>

    typedef unsigned int uint_t;

    // Bug in this ConcRT version results in task_group::wait() rethrowing
    // internal cancellation exception propagated by the scheduler from the nesting
    // task group.
    #define __TBB_SILENT_CANCELLATION_BROKEN  (_MSC_VER == 1600)

#endif /* !TBBTEST_USE_TBB */

#if __TBB_TASK_GROUP_CONTEXT

#include "tbb/atomic.h"
#include "tbb/aligned_space.h"
#include "harness.h"
#include "harness_concurrency_tracker.h"

unsigned g_MaxConcurrency = 0;

typedef tbb::atomic<uint_t> atomic_t;
typedef Concurrency::task_handle<void(*)()> handle_type;

//------------------------------------------------------------------------
// Tests for the thread safety of the task_group manipulations
//------------------------------------------------------------------------

#include "harness_barrier.h"

enum SharingMode {
    VagabondGroup = 1,
    ParallelWait = 2
};

template<typename task_group_type>
class SharedGroupBodyImpl : NoCopy, Harness::NoAfterlife {
    static const uint_t c_numTasks0 = 4096,
                        c_numTasks1 = 1024;

    const uint_t m_numThreads;
    const uint_t m_sharingMode;

    task_group_type *m_taskGroup;
    atomic_t m_tasksSpawned,
             m_threadsReady;
    Harness::SpinBarrier m_barrier;

    static atomic_t s_tasksExecuted;

    struct TaskFunctor {
        SharedGroupBodyImpl *m_pOwner;
        void operator () () const {
            if ( m_pOwner->m_sharingMode & ParallelWait ) {
                while ( Harness::ConcurrencyTracker::PeakParallelism() < m_pOwner->m_numThreads )
                    __TBB_Yield();
            }
            ++s_tasksExecuted;
        }
    };

    TaskFunctor m_taskFunctor;

    void Spawn ( uint_t numTasks ) {
        for ( uint_t i = 0; i < numTasks; ++i ) {
            ++m_tasksSpawned;
            Harness::ConcurrencyTracker ct;
            m_taskGroup->run( m_taskFunctor );
        }
        ++m_threadsReady;
    }

    void DeleteTaskGroup () {
        delete m_taskGroup;
        m_taskGroup = NULL;
    }

    void Wait () {
        while ( m_threadsReady != m_numThreads )
            __TBB_Yield();
        const uint_t numSpawned = c_numTasks0 + c_numTasks1 * (m_numThreads - 1);
        ASSERT ( m_tasksSpawned == numSpawned, "Wrong number of spawned tasks. The test is broken" );
        REMARK("Max spawning parallelism is %u out of %u\n", Harness::ConcurrencyTracker::PeakParallelism(), g_MaxConcurrency);
        if ( m_sharingMode & ParallelWait ) {
            m_barrier.wait( &Harness::ConcurrencyTracker::Reset );
            {
                Harness::ConcurrencyTracker ct;
                m_taskGroup->wait();
            }
            if ( Harness::ConcurrencyTracker::PeakParallelism() == 1 )
                REPORT ( "Warning: No parallel waiting detected in TestParallelWait\n" );
            m_barrier.wait();
        }
        else
            m_taskGroup->wait();
        ASSERT ( m_tasksSpawned == numSpawned, "No tasks should be spawned after wait starts. The test is broken" );
        ASSERT ( s_tasksExecuted == numSpawned, "Not all spawned tasks were executed" );
    }

public:
    SharedGroupBodyImpl ( uint_t numThreads, uint_t sharingMode = 0 )
        : m_numThreads(numThreads)
        , m_sharingMode(sharingMode)
        , m_taskGroup(NULL)
        , m_barrier(numThreads)
    {
        ASSERT ( m_numThreads > 1, "SharedGroupBody tests require concurrency" );
        ASSERT ( !(m_sharingMode & VagabondGroup) || m_numThreads == 2, "In vagabond mode SharedGroupBody must be used with 2 threads only" );
        Harness::ConcurrencyTracker::Reset();
        s_tasksExecuted = 0;
        m_tasksSpawned = 0;
        m_threadsReady = 0;
        m_taskFunctor.m_pOwner = this;
    }

    void Run ( uint_t idx ) {
#if TBBTEST_USE_TBB
        tbb::task_scheduler_init init(g_MaxConcurrency);
#endif
        AssertLive();
        if ( idx == 0 ) {
            ASSERT ( !m_taskGroup && !m_tasksSpawned, "SharedGroupBody must be reset before reuse");
            m_taskGroup = new task_group_type;
            Spawn( c_numTasks0 );
            Wait();
            if ( m_sharingMode & VagabondGroup )
                m_barrier.wait();
            else
                DeleteTaskGroup();
        }
        else {
            while ( m_tasksSpawned == 0 )
                __TBB_Yield();
            ASSERT ( m_taskGroup, "Task group is not initialized");
            Spawn (c_numTasks1);
            if ( m_sharingMode & ParallelWait )
                Wait();
            if ( m_sharingMode & VagabondGroup ) {
                ASSERT ( idx == 1, "In vagabond mode SharedGroupBody must be used with 2 threads only" );
                m_barrier.wait();
                DeleteTaskGroup();
            }
        }
        AssertLive();
    }
};

template<typename task_group_type>
atomic_t SharedGroupBodyImpl<task_group_type>::s_tasksExecuted;

template<typename task_group_type>
class  SharedGroupBody : NoAssign, Harness::NoAfterlife {
    bool m_bOwner;
    SharedGroupBodyImpl<task_group_type> *m_pImpl;
public:
    SharedGroupBody ( uint_t numThreads, uint_t sharingMode = 0 )
        : NoAssign()
        , Harness::NoAfterlife()
        , m_bOwner(true)
        , m_pImpl( new SharedGroupBodyImpl<task_group_type>(numThreads, sharingMode) )
    {}
    SharedGroupBody ( const SharedGroupBody& src )
        : NoAssign()
        , Harness::NoAfterlife()
        , m_bOwner(false)
        , m_pImpl(src.m_pImpl)
    {}
    ~SharedGroupBody () {
        if ( m_bOwner )
            delete m_pImpl;
    }
    void operator() ( uint_t idx ) const { m_pImpl->Run(idx); }
};

template<typename task_group_type>
class RunAndWaitSyncronizationTestBody : NoAssign {
    Harness::SpinBarrier& m_barrier;
    tbb::atomic<bool>& m_completed;
    task_group_type& m_tg;
public:
    RunAndWaitSyncronizationTestBody(Harness::SpinBarrier& barrier, tbb::atomic<bool>& completed, task_group_type& tg)
        : m_barrier(barrier), m_completed(completed), m_tg(tg) {}

    void operator()() const {
        m_barrier.wait();
        for (volatile int i = 0; i < 100000; ++i) {}
        m_completed = true;
    }

    void operator()(int id) const {
        if (id == 0) {
            m_tg.run_and_wait(*this);
        } else {
            m_barrier.wait();
            m_tg.wait();
            ASSERT(m_completed, "A concurrent waiter has left the wait method earlier than work has finished");
        }
    }
};



template<typename task_group_type>
void TestParallelSpawn () {
    NativeParallelFor( g_MaxConcurrency, SharedGroupBody<task_group_type>(g_MaxConcurrency) );
}

template<typename task_group_type>
void TestParallelWait () {
    NativeParallelFor( g_MaxConcurrency, SharedGroupBody<task_group_type>(g_MaxConcurrency, ParallelWait) );

    Harness::SpinBarrier barrier(g_MaxConcurrency);
    tbb::atomic<bool> completed;
    completed = false;
    task_group_type tg;
    RunAndWaitSyncronizationTestBody<task_group_type> b(barrier, completed, tg);
    NativeParallelFor( g_MaxConcurrency, b );
}

// Tests non-stack-bound task group (the group that is allocated by one thread and destroyed by the other)
template<typename task_group_type>
void TestVagabondGroup () {
    NativeParallelFor( 2, SharedGroupBody<task_group_type>(2, VagabondGroup) );
}



template<typename task_group_type>
void TestThreadSafety() {
    TestParallelSpawn<task_group_type>();
    TestParallelWait<task_group_type>();
    TestVagabondGroup<task_group_type>();
}

//------------------------------------------------------------------------
// Common requisites of the Fibonacci tests
//------------------------------------------------------------------------

const uint_t N = 20;
const uint_t F = 6765;

atomic_t g_Sum;

#define FIB_TEST_PROLOGUE() \
    const unsigned numRepeats = g_MaxConcurrency * (TBB_USE_DEBUG ? 4 : 16);    \
    Harness::ConcurrencyTracker::Reset()

#define FIB_TEST_EPILOGUE(sum) \
    ASSERT( sum == numRepeats * F, NULL ); \
    REMARK("Realized parallelism in Fib test is %u out of %u\n", Harness::ConcurrencyTracker::PeakParallelism(), g_MaxConcurrency)

// Fibonacci tasks specified as functors
template<class task_group_type>
class FibTaskBase : NoAssign, Harness::NoAfterlife {
protected:
    uint_t* m_pRes;
    mutable uint_t m_Num;
    virtual void impl() const = 0;
public:
    FibTaskBase( uint_t* y, uint_t n ) : m_pRes(y), m_Num(n) {}
    void operator()() const {
        Harness::ConcurrencyTracker ct;
        AssertLive();
        if( m_Num < 2 ) {
            *m_pRes = m_Num;
        } else {
            impl();
        }
    }
    virtual ~FibTaskBase() {}
};

template<class task_group_type>
class FibTaskAsymmetricTreeWithTaskHandle : public FibTaskBase<task_group_type> {
public:
    FibTaskAsymmetricTreeWithTaskHandle( uint_t* y, uint_t n ) : FibTaskBase<task_group_type>(y, n) {}
    virtual void impl() const __TBB_override {
        uint_t x = ~0u;
        task_group_type tg;
        Concurrency::task_handle<FibTaskAsymmetricTreeWithTaskHandle>
            h = FibTaskAsymmetricTreeWithTaskHandle(&x, this->m_Num-1);
        tg.run( h );
        this->m_Num -= 2; (*this)();
        tg.wait();
        *(this->m_pRes) += x;
    }
};

template<class task_group_type>
class FibTaskSymmetricTreeWithTaskHandle : public FibTaskBase<task_group_type> {
public:
    FibTaskSymmetricTreeWithTaskHandle( uint_t* y, uint_t n ) : FibTaskBase<task_group_type>(y, n) {}
    virtual void impl() const __TBB_override {
        uint_t x = ~0u,
               y = ~0u;
        task_group_type tg;
        Concurrency::task_handle<FibTaskSymmetricTreeWithTaskHandle>
            h1 = FibTaskSymmetricTreeWithTaskHandle(&x, this->m_Num-1),
            h2 = FibTaskSymmetricTreeWithTaskHandle(&y, this->m_Num-2);
        tg.run( h1 );
        tg.run( h2 );
        tg.wait();
        *(this->m_pRes) = x + y;
    }
};

template<class task_group_type>
class FibTaskAsymmetricTreeWithFunctor : public FibTaskBase<task_group_type> {
public:
    FibTaskAsymmetricTreeWithFunctor( uint_t* y, uint_t n ) : FibTaskBase<task_group_type>(y, n) {}
    virtual void impl() const __TBB_override {
        uint_t x = ~0u;
        task_group_type tg;
        tg.run( FibTaskAsymmetricTreeWithFunctor(&x, this->m_Num-1) );
        this->m_Num -= 2; tg.run_and_wait( *this );
        *(this->m_pRes) += x;
    }
};

template<class task_group_type>
class FibTaskSymmetricTreeWithFunctor : public FibTaskBase<task_group_type> {
public:
    FibTaskSymmetricTreeWithFunctor( uint_t* y, uint_t n ) : FibTaskBase<task_group_type>(y, n) {}
    virtual void impl() const __TBB_override {
        uint_t x = ~0u,
               y = ~0u;
        task_group_type tg;
        tg.run( FibTaskSymmetricTreeWithFunctor(&x, this->m_Num-1) );
        tg.run( FibTaskSymmetricTreeWithFunctor(&y, this->m_Num-2) );
        tg.wait();
        *(this->m_pRes) = x + y;
    }
};



// Helper functions
template<class fib_task>
uint_t RunFibTask(uint_t n) {
    uint_t res = ~0u;
    fib_task(&res, n)();
    return res;
}

template<typename fib_task>
void RunFibTest() {
    FIB_TEST_PROLOGUE();
    uint_t sum = 0;
    for( unsigned i = 0; i < numRepeats; ++i )
        sum += RunFibTask<fib_task>(N);
    FIB_TEST_EPILOGUE(sum);
}

template<typename fib_task>
void FibFunctionNoArgs() {
    g_Sum += RunFibTask<fib_task>(N);
}



template<typename task_group_type>
void TestFibWithTaskHandle() {
    RunFibTest<FibTaskAsymmetricTreeWithTaskHandle<task_group_type> >();
    RunFibTest< FibTaskSymmetricTreeWithTaskHandle<task_group_type> >();
}

#if __TBB_CPP11_LAMBDAS_PRESENT
template<typename task_group_type>
void TestFibWithMakeTask() {
    REMARK ("make_task test\n");
    atomic_t sum;
    sum = 0;
    task_group_type tg;
    auto h1 = Concurrency::make_task( [&](){sum += RunFibTask<FibTaskSymmetricTreeWithTaskHandle<task_group_type> >(N);} );
    auto h2 = Concurrency::make_task( [&](){sum += RunFibTask<FibTaskSymmetricTreeWithTaskHandle<task_group_type> >(N);} );
    tg.run( h1 );
    tg.run_and_wait( h2 );
    ASSERT( sum == 2 * F, NULL );
}

template<typename task_group_type>
void TestFibWithLambdas() {
    REMARK ("Lambdas test");
    FIB_TEST_PROLOGUE();
    atomic_t sum;
    sum = 0;
    task_group_type tg;
    for( unsigned i = 0; i < numRepeats; ++i )
        tg.run( [&](){sum += RunFibTask<FibTaskSymmetricTreeWithFunctor<task_group_type> >(N);} );
    tg.wait();
    FIB_TEST_EPILOGUE(sum);
}
#endif //__TBB_CPP11_LAMBDAS_PRESENT

template<typename task_group_type>
void TestFibWithFunctor() {
    RunFibTest<FibTaskAsymmetricTreeWithFunctor<task_group_type> >();
    RunFibTest< FibTaskSymmetricTreeWithFunctor<task_group_type> >();
}

template<typename task_group_type>
void TestFibWithFunctionPtr() {
    FIB_TEST_PROLOGUE();
    g_Sum = 0;
    task_group_type tg;
    for( unsigned i = 0; i < numRepeats; ++i )
        tg.run( &FibFunctionNoArgs<FibTaskSymmetricTreeWithFunctor<task_group_type> > );
    tg.wait();
    FIB_TEST_EPILOGUE(g_Sum);
}

template<typename task_group_type>
void TestFibInvalidMultipleScheduling() {
    FIB_TEST_PROLOGUE();
    g_Sum = 0;
    task_group_type tg;
    typedef tbb::aligned_space<handle_type> handle_space_t;
    handle_space_t *handles = new handle_space_t[numRepeats];
    handle_type *h = NULL;
#if __TBB_ipf && __TBB_GCC_VERSION==40601
    volatile // Workaround for unexpected exit from the loop below after the exception was caught
#endif
    unsigned i = 0;
    for( ;; ++i ) {
        h = handles[i].begin();
#if __TBB_FUNC_PTR_AS_TEMPL_PARAM_BROKEN
        new ( h ) handle_type((void(*)())FibFunctionNoArgs<FibTaskSymmetricTreeWithTaskHandle<task_group_type> >);
#else
        new ( h ) handle_type(FibFunctionNoArgs<FibTaskSymmetricTreeWithTaskHandle<task_group_type> >);
#endif
        if ( i == numRepeats - 1 )
            break;
        tg.run( *h );
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
        bool caught = false;
        try {
            if( i&1 ) tg.run( *h );
            else tg.run_and_wait( *h );
        }
        catch ( Concurrency::invalid_multiple_scheduling& e ) {
            ASSERT( e.what(), "Error message is absent" );
            caught = true;
        }
        catch ( ... ) {
            ASSERT ( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unrecognized exception" );
        }
        ASSERT ( caught, "Expected invalid_multiple_scheduling exception is missing" );
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
    }
    ASSERT( i == numRepeats - 1, "unexpected exit from the loop" );
    tg.run_and_wait( *h );

    for( i = 0; i < numRepeats; ++i )
#if __TBB_UNQUALIFIED_CALL_OF_DTOR_BROKEN
        handles[i].begin()->Concurrency::task_handle<void(*)()>::~task_handle();
#else
        handles[i].begin()->~handle_type();
#endif
    delete []handles;
    FIB_TEST_EPILOGUE(g_Sum);
}



template<typename task_group_type>
void RunFibonacciTests() {
    TestFibWithTaskHandle<task_group_type>();
#if __TBB_CPP11_LAMBDAS_PRESENT
    TestFibWithMakeTask<task_group_type>();
    TestFibWithLambdas<task_group_type>();
#endif
    TestFibWithFunctor<task_group_type>();
    TestFibWithFunctionPtr<task_group_type>();
    TestFibInvalidMultipleScheduling<task_group_type>();
}

// tbb::structured_task_group accepts tasks only as task_handle object
template<> void RunFibonacciTests<Concurrency::structured_task_group>() {
    TestFibWithTaskHandle<Concurrency::structured_task_group>();
#if __TBB_CPP11_LAMBDAS_PRESENT
    TestFibWithMakeTask<Concurrency::structured_task_group>();
#endif
    TestFibInvalidMultipleScheduling<Concurrency::structured_task_group>();
}



class test_exception : public std::exception
{
    const char* m_strDescription;
public:
    test_exception ( const char* descr ) : m_strDescription(descr) {}

    const char* what() const throw() __TBB_override { return m_strDescription; }
};

#if TBB_USE_CAPTURED_EXCEPTION
    #include "tbb/tbb_exception.h"
    typedef tbb::captured_exception TestException;
#else
    typedef test_exception TestException;
#endif

#include <string.h>

#define NUM_CHORES      512
#define NUM_GROUPS      64
#define SKIP_CHORES     (NUM_CHORES/4)
#define SKIP_GROUPS     (NUM_GROUPS/4)
#define EXCEPTION_DESCR1 "Test exception 1"
#define EXCEPTION_DESCR2 "Test exception 2"

atomic_t g_ExceptionCount;
atomic_t g_TaskCount;
unsigned g_ExecutedAtCancellation;
bool g_Rethrow;
bool g_Throw;
#if __TBB_SILENT_CANCELLATION_BROKEN
    volatile bool g_CancellationPropagationInProgress;
    #define CATCH_ANY()                                     \
        __TBB_CATCH( ... ) {                                \
            if ( g_CancellationPropagationInProgress ) {    \
                if ( g_Throw ) {                            \
                    exceptionCaught = true;                 \
                    ++g_ExceptionCount;                     \
                }                                           \
            } else                                          \
                ASSERT( false, "Unknown exception" );       \
        }
#else
    #define CATCH_ANY()  __TBB_CATCH( ... ) { ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unknown exception" ); }
#endif

class ThrowingTask : NoAssign, Harness::NoAfterlife {
    atomic_t &m_TaskCount;
public:
    ThrowingTask( atomic_t& counter ) : m_TaskCount(counter) {}
    void operator() () const {
        Harness::ConcurrencyTracker ct;
        AssertLive();
        if ( g_Throw ) {
            if ( ++m_TaskCount == SKIP_CHORES )
                __TBB_THROW( test_exception(EXCEPTION_DESCR1) );
            __TBB_Yield();
        }
        else {
            ++g_TaskCount;
            while( !Concurrency::is_current_task_group_canceling() )
                __TBB_Yield();
        }
    }
};

inline void ResetGlobals ( bool bThrow, bool bRethrow ) {
    g_Throw = bThrow;
    g_Rethrow = bRethrow;
#if __TBB_SILENT_CANCELLATION_BROKEN
    g_CancellationPropagationInProgress = false;
#endif
    g_ExceptionCount = 0;
    g_TaskCount = 0;
    Harness::ConcurrencyTracker::Reset();
}

template<typename task_group_type>
void LaunchChildrenWithFunctor () {
    atomic_t count;
    count = 0;
    task_group_type g;
    bool exceptionCaught = false;
    for( unsigned i = 0; i < NUM_CHORES; ++i )
        g.run( ThrowingTask(count) );
    Concurrency::task_group_status status = Concurrency::not_complete;
    __TBB_TRY {
        status = g.wait();
    } __TBB_CATCH ( TestException& e ) {
#if TBB_USE_EXCEPTIONS
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR1) == 0, "Unknown exception" );
#endif /* TBB_USE_EXCEPTIONS */
        exceptionCaught = true;
        ++g_ExceptionCount;
    } CATCH_ANY();
    ASSERT( !g_Throw || exceptionCaught || status == Concurrency::canceled, "No exception in the child task group" );
    if ( g_Rethrow && g_ExceptionCount > SKIP_GROUPS ) {
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        __TBB_THROW( test_exception(EXCEPTION_DESCR2) );
    }
}

template<typename task_group_type>
void LaunchChildrenWithTaskHandle () {
    atomic_t count;
    count = 0;
    task_group_type g;
    bool exceptionCaught = false;
    typedef Concurrency::task_handle<ThrowingTask> throwing_handle_type;
    tbb::aligned_space<throwing_handle_type,NUM_CHORES> handles;
    for( unsigned i = 0; i < NUM_CHORES; ++i ) {
        throwing_handle_type *h = handles.begin()+i;
        new ( h ) throwing_handle_type( ThrowingTask(count) );
        g.run( *h );
    }
    __TBB_TRY {
        g.wait();
    } __TBB_CATCH( TestException& e ) {
#if TBB_USE_EXCEPTIONS
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR1) == 0, "Unknown exception" );
#endif /* TBB_USE_EXCEPTIONS */
#if __TBB_SILENT_CANCELLATION_BROKEN
        ASSERT ( !g.is_canceling() || g_CancellationPropagationInProgress, "wait() has not reset cancellation state" );
#else
        ASSERT ( !g.is_canceling(), "wait() has not reset cancellation state" );
#endif
        exceptionCaught = true;
        ++g_ExceptionCount;
    } CATCH_ANY();
    ASSERT( !g_Throw || exceptionCaught, "No exception in the child task group" );
    for( unsigned i = 0; i < NUM_CHORES; ++i )
        (handles.begin()+i)->~throwing_handle_type();
    if ( g_Rethrow && g_ExceptionCount > SKIP_GROUPS ) {
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        __TBB_THROW( test_exception(EXCEPTION_DESCR2) );
    }
}

template<typename task_group_type>
class LaunchChildrenWithTaskHandleDriver {
    tbb::aligned_space<handle_type,NUM_CHORES> m_handles;

public:
    void Launch ( task_group_type& tg ) {
        ResetGlobals( false, false );
        for( unsigned i = 0; i < NUM_GROUPS; ++i ) {
            handle_type *h = m_handles.begin()+i;
            new ( h ) handle_type( LaunchChildrenWithTaskHandle<task_group_type> );
            tg.run( *h );
        }
        ASSERT ( !Concurrency::is_current_task_group_canceling(), "Unexpected cancellation" );
        ASSERT ( !tg.is_canceling(), "Unexpected cancellation" );
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        while ( g_MaxConcurrency > 1 && g_TaskCount == 0 )
            __TBB_Yield();
    }

    void Finish () {
        for( unsigned i = 0; i < NUM_GROUPS; ++i )
            (m_handles.begin()+i)->~handle_type();
        ASSERT( g_TaskCount <= NUM_GROUPS * NUM_CHORES, "Too many tasks reported. The test is broken" );
        ASSERT( g_TaskCount < NUM_GROUPS * NUM_CHORES, "No tasks were cancelled. Cancellation model changed?" );
        ASSERT( g_TaskCount <= g_ExecutedAtCancellation + g_MaxConcurrency, "Too many tasks survived cancellation" );
    }
}; // LaunchChildrenWithTaskHandleDriver



// Tests for cancellation and exception handling behavior
template<typename task_group_type>
void TestManualCancellationWithFunctor () {
    ResetGlobals( false, false );
    task_group_type tg;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        tg.run( &LaunchChildrenWithFunctor<task_group_type> );
    ASSERT ( !Concurrency::is_current_task_group_canceling(), "Unexpected cancellation" );
    ASSERT ( !tg.is_canceling(), "Unexpected cancellation" );
#if __TBB_SILENT_CANCELLATION_BROKEN
    g_CancellationPropagationInProgress = true;
#endif
    while ( g_MaxConcurrency > 1 && g_TaskCount == 0 )
        __TBB_Yield();
    tg.cancel();
    g_ExecutedAtCancellation = g_TaskCount;
    ASSERT ( tg.is_canceling(), "No cancellation reported" );
    tg.wait();
    ASSERT( g_TaskCount <= NUM_GROUPS * NUM_CHORES, "Too many tasks reported. The test is broken" );
    ASSERT( g_TaskCount < NUM_GROUPS * NUM_CHORES, "No tasks were cancelled. Cancellation model changed?" );
    ASSERT( g_TaskCount <= g_ExecutedAtCancellation + Harness::ConcurrencyTracker::PeakParallelism(), "Too many tasks survived cancellation" );
}

template<typename task_group_type>
void TestManualCancellationWithTaskHandle () {
    LaunchChildrenWithTaskHandleDriver<task_group_type> driver;
    task_group_type tg;
    driver.Launch( tg );
    tg.cancel();
    g_ExecutedAtCancellation = g_TaskCount;
    ASSERT ( tg.is_canceling(), "No cancellation reported" );
    tg.wait();
    driver.Finish();
}

#if TBB_USE_EXCEPTIONS
template<typename task_group_type>
void TestExceptionHandling1 () {
    ResetGlobals( true, false );
    task_group_type tg;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        tg.run( &LaunchChildrenWithFunctor<task_group_type> );
    try {
        tg.wait();
    } catch ( ... ) {
        ASSERT( false, "Unexpected exception" );
    }
    ASSERT( g_ExceptionCount <= NUM_GROUPS, "Too many exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount == NUM_GROUPS, "Not all child groups threw the exception" );
}

template<typename task_group_type>
void TestExceptionHandling2 () {
    ResetGlobals( true, true );
    task_group_type tg;
    bool exceptionCaught = false;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        tg.run( &LaunchChildrenWithFunctor<task_group_type> );
    try {
        tg.wait();
    } catch ( TestException& e ) {
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR2) == 0, "Unknown exception" );
        ASSERT ( !tg.is_canceling(), "wait() has not reset cancellation state" );
        exceptionCaught = true;
    } CATCH_ANY();
    ASSERT( exceptionCaught, "No exception thrown from the root task group" );
    ASSERT( g_ExceptionCount >= SKIP_GROUPS, "Too few exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount <= NUM_GROUPS - SKIP_GROUPS, "Too many exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount < NUM_GROUPS - SKIP_GROUPS, "None of the child groups was cancelled" );
}

#if defined(_MSC_VER)
    #pragma warning (disable: 4127)
#endif

template<typename task_group_type, bool Throw>
void TestMissingWait () {
    bool exception_occurred = false,
         unexpected_exception = false;
    LaunchChildrenWithTaskHandleDriver<task_group_type> driver;
    try {
        task_group_type tg;
        driver.Launch( tg );
        if ( Throw )
            throw int(); // Initiate stack unwinding
    }
    catch ( const Concurrency::missing_wait& e ) {
        ASSERT( e.what(), "Error message is absent" );
        exception_occurred = true;
        unexpected_exception = Throw;
    }
    catch ( int ) {
        exception_occurred = true;
        unexpected_exception = !Throw;
    }
    catch ( ... ) {
        exception_occurred = unexpected_exception = true;
    }
    ASSERT( exception_occurred, NULL );
    ASSERT( !unexpected_exception, NULL );
    driver.Finish();
}
#endif /* TBB_USE_EXCEPTIONS */

template<typename task_group_type>
void RunCancellationAndExceptionHandlingTests() {
    TestManualCancellationWithFunctor              <Concurrency::task_group>();
    TestManualCancellationWithTaskHandle<Concurrency::task_group>();
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    TestExceptionHandling1<Concurrency::task_group>();
    TestExceptionHandling2<Concurrency::task_group>();

    TestMissingWait<Concurrency::task_group, true>();
#if !(__TBB_THROW_FROM_DTOR_BROKEN || __TBB_STD_UNCAUGHT_EXCEPTION_BROKEN)
    TestMissingWait<Concurrency::task_group, false>();
#else
    REPORT("Known issue: TestMissingWait<task_group_type, false>() is skipped.\n");
#endif
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
}

template<> void RunCancellationAndExceptionHandlingTests<Concurrency::structured_task_group>() {
    TestManualCancellationWithTaskHandle<Concurrency::structured_task_group>();
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    TestMissingWait<Concurrency::structured_task_group, true>();
#if !(__TBB_THROW_FROM_DTOR_BROKEN || __TBB_STD_UNCAUGHT_EXCEPTION_BROKEN)
    TestMissingWait<Concurrency::structured_task_group, false>();
#else
    REPORT("Known issue: TestMissingWait<task_group_type, false>() is skipped.\n");
#endif
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
}



void EmptyFunction () {}

void TestStructuredWait () {
    Concurrency::structured_task_group sg;
    handle_type h(EmptyFunction);
    sg.run(h);
    sg.wait();
    handle_type h2(EmptyFunction);
    sg.run(h2);
    sg.wait();
}

struct TestFunctor {
    void operator()() { ASSERT( false, "Non-const operator called" ); }
    void operator()() const { /* library requires this overload only */ }
};

template<typename task_group_type>
void TestConstantFunctorRequirement() {
    task_group_type g;
    TestFunctor tf;
    g.run( tf ); g.wait();
    g.run_and_wait( tf );
}



//------------------------------------------------------------------------
#if __TBB_CPP11_RVALUE_REF_PRESENT
namespace TestMoveSemanticsNS {
    struct TestFunctor {
        void operator()() const {};
    };

    struct MoveOnlyFunctor : MoveOnly, TestFunctor {
        MoveOnlyFunctor() : MoveOnly() {};
        MoveOnlyFunctor(MoveOnlyFunctor&& other) : MoveOnly(std::move(other)) {};
    };

    struct MovePreferableFunctor : Movable, TestFunctor {
        MovePreferableFunctor() : Movable() {};
        MovePreferableFunctor(MovePreferableFunctor&& other) : Movable(std::move(other)) {};
        MovePreferableFunctor(const MovePreferableFunctor& other) : Movable(other) {};
    };

    struct NoMoveNoCopyFunctor : NoCopy, TestFunctor {
        NoMoveNoCopyFunctor() : NoCopy() {};
        // mv ctor is not allowed as cp ctor from parent NoCopy
    private:
        NoMoveNoCopyFunctor(NoMoveNoCopyFunctor&&);
    };

    template<typename task_group_type>
    void TestFunctorsWithinTaskHandles() {
        // working with task_handle rvalues is not supported in task_group

        task_group_type tg;
        MovePreferableFunctor mpf;
        typedef tbb::task_handle<MoveOnlyFunctor> th_mv_only_type;
        typedef tbb::task_handle<MovePreferableFunctor> th_mv_pref_type;

        th_mv_only_type th_mv_only = th_mv_only_type(MoveOnlyFunctor());
        tg.run_and_wait(th_mv_only);

        th_mv_only_type th_mv_only1 = th_mv_only_type(MoveOnlyFunctor());
        tg.run(th_mv_only1);
        tg.wait();

        th_mv_pref_type th_mv_pref = th_mv_pref_type(mpf);
        tg.run_and_wait(th_mv_pref);
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref1 = th_mv_pref_type(std::move(mpf));
        tg.run_and_wait(th_mv_pref1);
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref2 = th_mv_pref_type(mpf);
        tg.run(th_mv_pref2);
        tg.wait();
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref3 = th_mv_pref_type(std::move(mpf));
        tg.run(th_mv_pref3);
        tg.wait();
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }

    template<typename task_group_type>
    void TestBareFunctors() {
        task_group_type tg;
        MovePreferableFunctor mpf;
        // run_and_wait() doesn't have any copies or moves of arguments inside the impl
        tg.run_and_wait( NoMoveNoCopyFunctor() );

        tg.run( MoveOnlyFunctor() );
        tg.wait();

        tg.run( mpf );
        tg.wait();
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        tg.run( std::move(mpf) );
        tg.wait();
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }

    void TestMakeTask() {
        MovePreferableFunctor mpf;

        tbb::make_task( MoveOnly() );

        tbb::make_task( mpf );
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        tbb::make_task( std::move(mpf) );
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

template<typename task_group_type>
void TestMoveSemantics() {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveSemanticsNS::TestBareFunctors<task_group_type>();
    TestMoveSemanticsNS::TestFunctorsWithinTaskHandles<task_group_type>();
    TestMoveSemanticsNS::TestMakeTask();
#else
    REPORT("Known issue: move support tests are skipped.\n");
#endif
}
//------------------------------------------------------------------------


#if TBBTEST_USE_TBB && TBB_PREVIEW_ISOLATED_TASK_GROUP
namespace TestIsolationNS {
    class DummyFunctor {
    public:
        DummyFunctor() {}
        void operator()() const {
            for ( volatile int j = 0; j < 10; ++j ) {}
        }
    };

    template<typename task_group_type>
    class ParForBody {
        task_group_type& m_tg;
        tbb::atomic<bool>& m_preserved;
        tbb::enumerable_thread_specific<int>& m_ets;
    public:
        ParForBody(
            task_group_type& tg,
            tbb::atomic<bool>& preserved,
            tbb::enumerable_thread_specific<int>& ets
        ) : m_tg(tg), m_preserved(preserved), m_ets(ets) {}

        void operator()(int) const {
            if (++m_ets.local() > 1) m_preserved = false;

            for (int i = 0; i < 1000; ++i)
                m_tg.run(DummyFunctor());
            m_tg.wait();
            m_tg.run_and_wait(DummyFunctor());

            --m_ets.local();
        }
    };

    template<typename task_group_type>
    void CheckIsolation(bool isolation_is_expected) {
        task_group_type tg;
        tbb::atomic<bool> isolation_is_preserved;
        isolation_is_preserved = true;
        tbb::enumerable_thread_specific<int> ets(0);

        tbb::parallel_for(0, 100, ParForBody<task_group_type>(tg, isolation_is_preserved, ets));

        ASSERT(
            isolation_is_expected == isolation_is_preserved,
            "Actual and expected isolation-related behaviours are different"
        );
    }

    // Should be called only when > 1 thread is used, because otherwise isolation is guaranteed to take place
    void TestIsolation() {
        CheckIsolation<tbb::task_group>(false);
        CheckIsolation<tbb::isolated_task_group>(true);
    }
}
#endif


int TestMain () {
    REMARK ("Testing %s task_group functionality\n", TBBTEST_USE_TBB ? "TBB" : "PPL");
    for( int p=MinThread; p<=MaxThread; ++p ) {
        g_MaxConcurrency = p;
#if TBBTEST_USE_TBB
        tbb::task_scheduler_init init(p);
#else
        Concurrency::SchedulerPolicy sp( 4,
                                Concurrency::SchedulerKind, Concurrency::ThreadScheduler,
                                Concurrency::MinConcurrency, 1,
                                Concurrency::MaxConcurrency, p,
                                Concurrency::TargetOversubscriptionFactor, 1);
        Concurrency::Scheduler  *s = Concurrency::Scheduler::Create( sp );
#endif /* !TBBTEST_USE_TBB */
        if ( p > 1 )
            TestThreadSafety<Concurrency::task_group>();

        RunFibonacciTests<Concurrency::task_group>();
        RunFibonacciTests<Concurrency::structured_task_group>();

        RunCancellationAndExceptionHandlingTests<Concurrency::task_group>();
        RunCancellationAndExceptionHandlingTests<Concurrency::structured_task_group>();

#if TBBTEST_USE_TBB && TBB_PREVIEW_ISOLATED_TASK_GROUP
        if ( p > 1 ) {
            TestThreadSafety<tbb::isolated_task_group>();
            TestIsolationNS::TestIsolation();
        }
        RunFibonacciTests<tbb::isolated_task_group>();
        RunCancellationAndExceptionHandlingTests<tbb::isolated_task_group>();
        TestConstantFunctorRequirement<tbb::isolated_task_group>();
        TestMoveSemantics<tbb::isolated_task_group>();
#else
        REPORT ("Known issue: tests for tbb::isolated_task_group are skipped.\n");
#endif

#if !TBBTEST_USE_TBB
        s->Release();
#endif
    }
    TestStructuredWait();
    TestConstantFunctorRequirement<Concurrency::task_group>();
    TestMoveSemantics<Concurrency::task_group>();
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception handling tests are skipped.\n");
#endif
    return Harness::Done;
}

#else /* !__TBB_TASK_GROUP_CONTEXT */

#include "harness.h"

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_TASK_GROUP_CONTEXT */
