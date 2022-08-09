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

#include "harness_task.h"
#include "tbb/atomic.h"
#include "tbb/tbb_thread.h"
#include "tbb/task_scheduler_init.h"
#include <cstdlib>

//------------------------------------------------------------------------
// Test for task::spawn_children and task_list
//------------------------------------------------------------------------

class UnboundedlyRecursiveOnUnboundedStealingTask : public tbb::task {
    typedef UnboundedlyRecursiveOnUnboundedStealingTask this_type;

    this_type *m_Parent;
    const int m_Depth;
    volatile bool m_GoAhead;

    // Well, virtually unboundedly, for any practical purpose
    static const int max_depth = 1000000;

public:
    UnboundedlyRecursiveOnUnboundedStealingTask( this_type *parent_ = NULL, int depth_ = max_depth )
        : m_Parent(parent_)
        , m_Depth(depth_)
        , m_GoAhead(true)
    {}

    tbb::task* execute() __TBB_override {
        // Using large padding array speeds up reaching stealing limit
        const int paddingSize = 16 * 1024;
        volatile char padding[paddingSize];
        if( !m_Parent || (m_Depth > 0 &&  m_Parent->m_GoAhead) ) {
            if ( m_Parent ) {
                // We are stolen, let our parent start waiting for us
                m_Parent->m_GoAhead = false;
            }
            tbb::task &t = *new( allocate_child() ) this_type(this, m_Depth - 1);
            set_ref_count( 2 );
            spawn( t );
            // Give a willing thief a chance to steal
            for( int i = 0; i < 1000000 && m_GoAhead; ++i ) {
                ++padding[i % paddingSize];
                __TBB_Yield();
            }
            // If our child has not been stolen yet, then prohibit it siring ones
            // of its own (when this thread executes it inside the next wait_for_all)
            m_GoAhead = false;
            wait_for_all();
        }
        return NULL;
    }
}; // UnboundedlyRecursiveOnUnboundedStealingTask

tbb::atomic<int> Count;

class RecursiveTask: public tbb::task {
    const int m_ChildCount;
    const int m_Depth;
    //! Spawn tasks in list.  Exact method depends upon m_Depth&bit_mask.
    void SpawnList( tbb::task_list& list, int bit_mask ) {
        if( m_Depth&bit_mask ) {
            // Take address to check that signature of spawn(task_list&) is static.
            void (*s)(tbb::task_list&) = &tbb::task::spawn;
            (*s)(list);
            ASSERT( list.empty(), NULL );
            wait_for_all();
        } else {
            spawn_and_wait_for_all(list);
            ASSERT( list.empty(), NULL );
        }
    }
public:
    RecursiveTask( int child_count, int depth_ ) : m_ChildCount(child_count), m_Depth(depth_) {}
    tbb::task* execute() __TBB_override {
        ++Count;
        if( m_Depth>0 ) {
            tbb::task_list list;
            ASSERT( list.empty(), NULL );
            for( int k=0; k<m_ChildCount; ++k ) {
                list.push_back( *new( allocate_child() ) RecursiveTask(m_ChildCount/2,m_Depth-1 ) );
                ASSERT( !list.empty(), NULL );
            }
            set_ref_count( m_ChildCount+1 );
            SpawnList( list, 1 );
            // Now try reusing this as the parent.
            set_ref_count(2);
            list.push_back( *new ( allocate_child() ) tbb::empty_task() );
            SpawnList( list, 2 );
        }
        return NULL;
    }
};

//! Compute what Count should be after RecursiveTask(child_count,depth) runs.
static int Expected( int child_count, int depth ) {
    return depth<=0 ? 1 : 1+child_count*Expected(child_count/2,depth-1);
}

void TestStealLimit( int nthread ) {
#if __TBB_DEFINE_MIC
    REMARK( "skipping steal limiting heuristics for %d threads\n", nthread );
#else// !_TBB_DEFINE_MIC
    REMARK( "testing steal limiting heuristics for %d threads\n", nthread );
    tbb::task_scheduler_init init(nthread);
    tbb::task &t = *new( tbb::task::allocate_root() ) UnboundedlyRecursiveOnUnboundedStealingTask();
    tbb::task::spawn_root_and_wait(t);
#endif// _TBB_DEFINE_MIC
}

//! Test task::spawn( task_list& )
void TestSpawnChildren( int nthread ) {
    REMARK("testing task::spawn(task_list&) for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=0; j<50; ++j ) {
        Count = 0;
        RecursiveTask& p = *new( tbb::task::allocate_root() ) RecursiveTask(j,4);
        tbb::task::spawn_root_and_wait(p);
        int expected = Expected(j,4);
        ASSERT( Count==expected, NULL );
    }
}

//! Test task::spawn_root_and_wait( task_list& )
void TestSpawnRootList( int nthread ) {
    REMARK("testing task::spawn_root_and_wait(task_list&) for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=0; j<5; ++j )
        for( int k=0; k<10; ++k ) {
            Count = 0;
            tbb::task_list list;
            for( int i=0; i<k; ++i )
                list.push_back( *new( tbb::task::allocate_root() ) RecursiveTask(j,4) );
            tbb::task::spawn_root_and_wait(list);
            int expected = k*Expected(j,4);
            ASSERT( Count==expected, NULL );
        }
}

//------------------------------------------------------------------------
// Test for task::recycle_as_safe_continuation
//------------------------------------------------------------------------

void TestSafeContinuation( int nthread ) {
    REMARK("testing task::recycle_as_safe_continuation for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=8; j<33; ++j ) {
        TaskGenerator& p = *new( tbb::task::allocate_root() ) TaskGenerator(j,5);
        tbb::task::spawn_root_and_wait(p);
    }
}

//------------------------------------------------------------------------
// Test affinity interface
//------------------------------------------------------------------------
tbb::atomic<int> TotalCount;

struct AffinityTask: public tbb::task {
    const affinity_id expected_affinity_id;
    bool noted;
    /** Computing affinities is NOT supported by TBB, and may disappear in the future.
        It is done here for sake of unit testing. */
    AffinityTask( int expected_affinity_id_ ) :
        expected_affinity_id(affinity_id(expected_affinity_id_)),
        noted(false)
    {
        set_affinity(expected_affinity_id);
        ASSERT( 0u-expected_affinity_id>0u, "affinity_id not an unsigned integral type?" );
        ASSERT( affinity()==expected_affinity_id, NULL );
    }
    tbb::task* execute() __TBB_override {
        ++TotalCount;
        return NULL;
    }
    void note_affinity( affinity_id id ) __TBB_override {
        // There is no guarantee in TBB that a task runs on its affinity thread.
        // However, the current implementation does accidentally guarantee it
        // under certain conditions, such as the conditions here.
        // We exploit those conditions for sake of unit testing.
        ASSERT( id!=expected_affinity_id, NULL );
        ASSERT( !noted, "note_affinity_id called twice!" );
        ASSERT ( &self() == (tbb::task*)this, "Wrong innermost running task" );
        noted = true;
    }
};

/** Note: This test assumes a lot about the internal implementation of affinity.
    Do NOT use this as an example of good programming practice with TBB */
void TestAffinity( int nthread ) {
    TotalCount = 0;
    int n = tbb::task_scheduler_init::default_num_threads();
    if( n>nthread )
        n = nthread;
    tbb::task_scheduler_init init(n);
    tbb::empty_task* t = new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task::affinity_id affinity_id = t->affinity();
    ASSERT( affinity_id==0, NULL );
    // Set ref_count for n-1 children, plus 1 for the wait.
    t->set_ref_count(n);
    // Spawn n-1 affinitized children.
    for( int i=1; i<n; ++i )
        tbb::task::spawn( *new(t->allocate_child()) AffinityTask(i) );
    if( n>1 ) {
        // Keep master from stealing
        while( TotalCount!=n-1 )
            __TBB_Yield();
    }
    // Wait for the children
    t->wait_for_all();
    int k = 0;
    GetTaskPtr(k)->destroy(*t);
    ASSERT(k==1,NULL);
}

struct NoteAffinityTask: public tbb::task {
    bool noted;
    NoteAffinityTask( int id ) : noted(false)
    {
        set_affinity(affinity_id(id));
    }
    ~NoteAffinityTask () {
        ASSERT (noted, "note_affinity has not been called");
    }
    tbb::task* execute() __TBB_override {
        return NULL;
    }
    void note_affinity( affinity_id /*id*/ ) __TBB_override {
        noted = true;
        ASSERT ( &self() == (tbb::task*)this, "Wrong innermost running task" );
    }
};

// This test checks one of the paths inside the scheduler by affinitizing the child task
// to non-existent thread so that it is proxied in the local task pool but not retrieved
// by another thread.
// If no workers requested, the extra slot #2 is allocated for a worker thread to serve
// "enqueued" tasks. In this test, it is used only for the affinity purpose.
void TestNoteAffinityContext() {
    tbb::task_scheduler_init init(1);
    tbb::empty_task* t = new( tbb::task::allocate_root() ) tbb::empty_task;
    t->set_ref_count(2);
    // This master in the absence of workers will have an affinity id of 1.
    // So use another number to make the task get proxied.
    tbb::task::spawn( *new(t->allocate_child()) NoteAffinityTask(2) );
    t->wait_for_all();
    tbb::task::destroy(*t);
}

//------------------------------------------------------------------------
// Test that recovery actions work correctly for task::allocate_* methods
// when a task's constructor throws an exception.
//------------------------------------------------------------------------

#if TBB_USE_EXCEPTIONS
static int TestUnconstructibleTaskCount;

struct ConstructionFailure {
};

#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    // Suppress pointless "unreachable code" warning.
    #pragma warning (push)
    #pragma warning (disable: 4702)
#endif

//! Task that cannot be constructed.
template<size_t N>
struct UnconstructibleTask: public tbb::empty_task {
    char space[N];
    UnconstructibleTask() {
        throw ConstructionFailure();
    }
};

#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    #pragma warning (pop)
#endif

#define TRY_BAD_CONSTRUCTION(x)                  \
    {                                            \
        try {                                    \
            new(x) UnconstructibleTask<N>;       \
        } catch( const ConstructionFailure& ) {                                                    \
            ASSERT( parent()==original_parent, NULL ); \
            ASSERT( ref_count()==original_ref_count, "incorrectly changed ref_count" );\
            ++TestUnconstructibleTaskCount;      \
        }                                        \
    }

template<size_t N>
struct RootTaskForTestUnconstructibleTask: public tbb::task {
    tbb::task* execute() __TBB_override {
        tbb::task* original_parent = parent();
        ASSERT( original_parent!=NULL, NULL );
        int original_ref_count = ref_count();
        TRY_BAD_CONSTRUCTION( allocate_root() );
        TRY_BAD_CONSTRUCTION( allocate_child() );
        TRY_BAD_CONSTRUCTION( allocate_continuation() );
        TRY_BAD_CONSTRUCTION( allocate_additional_child_of(*this) );
        return NULL;
    }
};

template<size_t N>
void TestUnconstructibleTask() {
    TestUnconstructibleTaskCount = 0;
    tbb::task_scheduler_init init;
    tbb::task* t = new( tbb::task::allocate_root() ) RootTaskForTestUnconstructibleTask<N>;
    tbb::task::spawn_root_and_wait(*t);
    ASSERT( TestUnconstructibleTaskCount==4, NULL );
}
#endif /* TBB_USE_EXCEPTIONS */

//------------------------------------------------------------------------
// Test for alignment problems with task objects.
//------------------------------------------------------------------------

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for pointless warning "structure was padded due to __declspec(align())
    #pragma warning (push)
    #pragma warning (disable: 4324)
#endif

//! Task with members of type T.
/** The task recursively creates tasks. */
template<typename T>
class TaskWithMember: public tbb::task {
    T x;
    T y;
    unsigned char count;
    tbb::task* execute() __TBB_override {
        x = y;
        if( count>0 ) {
            set_ref_count(2);
            tbb::task* t = new( allocate_child() ) TaskWithMember<T>(count-1);
            spawn_and_wait_for_all(*t);
        }
        return NULL;
    }
public:
    TaskWithMember( unsigned char n ) : count(n) {}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif

template<typename T>
void TestAlignmentOfOneClass() {
    typedef TaskWithMember<T> task_type;
    tbb::task* t = new( tbb::task::allocate_root() ) task_type(10);
    tbb::task::spawn_root_and_wait(*t);
}

#include "harness_m128.h"

void TestAlignment() {
    REMARK("testing alignment\n");
    tbb::task_scheduler_init init;
    // Try types that have variety of alignments
    TestAlignmentOfOneClass<char>();
    TestAlignmentOfOneClass<short>();
    TestAlignmentOfOneClass<int>();
    TestAlignmentOfOneClass<long>();
    TestAlignmentOfOneClass<void*>();
    TestAlignmentOfOneClass<float>();
    TestAlignmentOfOneClass<double>();
#if HAVE_m128
    TestAlignmentOfOneClass<__m128>();
#endif
#if HAVE_m256
    if (have_AVX()) TestAlignmentOfOneClass<__m256>();
#endif
}

//------------------------------------------------------------------------
// Test for recursing on left while spawning on right
//------------------------------------------------------------------------

int Fib( int n );

struct RightFibTask: public tbb::task {
    int* y;
    const int n;
    RightFibTask( int* y_, int n_ ) : y(y_), n(n_) {}
    task* execute() __TBB_override {
        *y = Fib(n-1);
        return 0;
    }
};

int Fib( int n ) {
    if( n<2 ) {
        return n;
    } else {
        // y actually does not need to be initialized.  It is initialized solely to suppress
        // a gratuitous warning "potentially uninitialized local variable".
        int y=-1;
        tbb::task* root_task = new( tbb::task::allocate_root() ) tbb::empty_task;
        root_task->set_ref_count(2);
        tbb::task::spawn( *new( root_task->allocate_child() ) RightFibTask(&y,n) );
        int x = Fib(n-2);
        root_task->wait_for_all();
        tbb::task::destroy(*root_task);
        return y+x;
    }
}

void TestLeftRecursion( int p ) {
    REMARK("testing non-spawned roots for %d threads\n",p);
    tbb::task_scheduler_init init(p);
    int sum = 0;
    for( int i=0; i<100; ++i )
        sum +=Fib(10);
    ASSERT( sum==5500, NULL );
}

//------------------------------------------------------------------------
// Test for computing with DAG of tasks.
//------------------------------------------------------------------------

class DagTask: public tbb::task {
    typedef unsigned long long number_t;
    const int i, j;
    number_t sum_from_left, sum_from_above;
    void check_sum( number_t sum ) {
        number_t expected_sum = 1;
        for( int k=i+1; k<=i+j; ++k )
            expected_sum *= k;
        for( int k=1; k<=j; ++k )
            expected_sum /= k;
        ASSERT(sum==expected_sum, NULL);
    }
public:
    DagTask *successor_to_below, *successor_to_right;
    DagTask( int i_, int j_ ) : i(i_), j(j_), sum_from_left(0), sum_from_above(0) {}
    task* execute() __TBB_override {
        ASSERT( ref_count()==0, NULL );
        number_t sum = i==0 && j==0 ? 1 : sum_from_left+sum_from_above;
        check_sum(sum);
        ++execution_count;
        if( DagTask* t = successor_to_right ) {
            t->sum_from_left = sum;
            if( t->decrement_ref_count()==0 )
                // Test using spawn to evaluate DAG
                spawn( *t );
        }
        if( DagTask* t = successor_to_below ) {
            t->sum_from_above = sum;
            if( t->add_ref_count(-1)==0 )
                // Test using bypass to evaluate DAG
                return t;
        }
        return NULL;
    }
    ~DagTask() {++destruction_count;}
    static tbb::atomic<int> execution_count;
    static tbb::atomic<int> destruction_count;
};

tbb::atomic<int> DagTask::execution_count;
tbb::atomic<int> DagTask::destruction_count;

void TestDag( int p ) {
    REMARK("testing evaluation of DAG for %d threads\n",p);
    tbb::task_scheduler_init init(p);
    DagTask::execution_count=0;
    DagTask::destruction_count=0;
    const int n = 10;
    DagTask* a[n][n];
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j )
            a[i][j] = new( tbb::task::allocate_root() ) DagTask(i,j);
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j ) {
            a[i][j]->successor_to_below = i+1<n ? a[i+1][j] : NULL;
            a[i][j]->successor_to_right = j+1<n ? a[i][j+1] : NULL;
            a[i][j]->set_ref_count((i>0)+(j>0));
        }
    a[n-1][n-1]->increment_ref_count();
    a[n-1][n-1]->spawn_and_wait_for_all(*a[0][0]);
    ASSERT( DagTask::execution_count == n*n - 1, NULL );
    tbb::task::destroy(*a[n-1][n-1]);
    ASSERT( DagTask::destruction_count > n*n - p, NULL );
    while ( DagTask::destruction_count != n*n )
        __TBB_Yield();
}

#include "harness_barrier.h"

class RelaxedOwnershipTask: public tbb::task {
    tbb::task &m_taskToSpawn,
              &m_taskToDestroy,
              &m_taskToExecute;
    static Harness::SpinBarrier m_barrier;

    tbb::task* execute () __TBB_override {
        tbb::task &p = *parent();
        tbb::task &r = *new( allocate_root() ) tbb::empty_task;
        r.set_ref_count( 1 );
        m_barrier.wait();
        p.spawn( *new(p.allocate_child()) tbb::empty_task );
        p.spawn( *new(task::allocate_additional_child_of(p)) tbb::empty_task );
        p.spawn( m_taskToSpawn );
        p.destroy( m_taskToDestroy );
        r.spawn_and_wait_for_all( m_taskToExecute );
        p.destroy( r );
        return NULL;
    }
public:
    RelaxedOwnershipTask ( tbb::task& toSpawn, tbb::task& toDestroy, tbb::task& toExecute )
        : m_taskToSpawn(toSpawn)
        , m_taskToDestroy(toDestroy)
        , m_taskToExecute(toExecute)
    {}
    static void SetBarrier ( int numThreads ) { m_barrier.initialize( numThreads ); }
};

Harness::SpinBarrier RelaxedOwnershipTask::m_barrier;

void TestRelaxedOwnership( int p ) {
    if ( p < 2 )
        return;

    if( unsigned(p)>tbb::tbb_thread::hardware_concurrency() )
        return;

    REMARK("testing tasks exercising relaxed ownership freedom for %d threads\n", p);
    tbb::task_scheduler_init init(p);
    RelaxedOwnershipTask::SetBarrier(p);
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task_list tl;
    for ( int i = 0; i < p; ++i ) {
        tbb::task &tS = *new( r.allocate_child() ) tbb::empty_task,
                  &tD = *new( r.allocate_child() ) tbb::empty_task,
                  &tE = *new( r.allocate_child() ) tbb::empty_task;
        tl.push_back( *new( r.allocate_child() ) RelaxedOwnershipTask(tS, tD, tE) );
    }
    r.set_ref_count( 5 * p + 1 );
    int k=0;
    GetTaskPtr(k)->spawn( tl );
    ASSERT(k==1,NULL);
    r.wait_for_all();
    r.destroy( r );
}

//------------------------------------------------------------------------
// Test for running TBB scheduler on user-created thread.
//------------------------------------------------------------------------

void RunSchedulerInstanceOnUserThread( int n_child ) {
    tbb::task* e = new( tbb::task::allocate_root() ) tbb::empty_task;
    e->set_ref_count(1+n_child);
    for( int i=0; i<n_child; ++i )
        tbb::task::spawn( *new(e->allocate_child()) tbb::empty_task );
    e->wait_for_all();
    e->destroy(*e);
}

void TestUserThread( int p ) {
    tbb::task_scheduler_init init(p);
    // Try with both 0 and 1 children.  Only the latter scenario permits stealing.
    for( int n_child=0; n_child<2; ++n_child ) {
        tbb::tbb_thread t( RunSchedulerInstanceOnUserThread, n_child );
        t.join();
    }
}

class TaskWithChildToSteal : public tbb::task {
    const int m_Depth;
    volatile bool m_GoAhead;

public:
    TaskWithChildToSteal( int depth_ )
        : m_Depth(depth_)
        , m_GoAhead(false)
    {}

    tbb::task* execute() __TBB_override {
        m_GoAhead = true;
        if ( m_Depth > 0 ) {
            TaskWithChildToSteal &t = *new( allocate_child() ) TaskWithChildToSteal(m_Depth - 1);
            t.SpawnAndWaitOnParent();
        }
        else
            Harness::Sleep(50); // The last task in chain sleeps for 50 ms
        return NULL;
    }

    void SpawnAndWaitOnParent() {
        parent()->set_ref_count( 2 );
        parent()->spawn( *this );
        while (!this->m_GoAhead )
            __TBB_Yield();
        parent()->wait_for_all();
    }
}; // TaskWithChildToSteal

// Success criterion of this test is not hanging
void TestDispatchLoopResponsiveness() {
    REMARK("testing that dispatch loops do not go into eternal sleep when all remaining children are stolen\n");
    // Recursion depth values test the following sorts of dispatch loops
    // 0 - master's outermost
    // 1 - worker's nested
    // 2 - master's nested
    tbb::task_scheduler_init init(2);
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    for ( int depth = 0; depth < 3; ++depth ) {
        TaskWithChildToSteal &t = *new( r.allocate_child() ) TaskWithChildToSteal(depth);
        t.SpawnAndWaitOnParent();
    }
    r.destroy(r);
}

void TestWaitDiscriminativenessWithoutStealing() {
    REMARK( "testing that task::wait_for_all is specific to the root it is called on (no workers)\n" );
    // The test relies on the strict LIFO scheduling order in the absence of workers
    tbb::task_scheduler_init init(1);
    tbb::task &r1 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task &r2 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    const int NumChildren = 10;
    r1.set_ref_count( NumChildren + 1 );
    r2.set_ref_count( NumChildren + 1 );
    for( int i=0; i < NumChildren; ++i ) {
        tbb::empty_task &t1 = *new( r1.allocate_child() ) tbb::empty_task;
        tbb::empty_task &t2 = *new( r2.allocate_child() ) tbb::empty_task;
        tbb::task::spawn(t1);
        tbb::task::spawn(t2);
    }
    r2.wait_for_all();
    ASSERT( r2.ref_count() <= 1, "Not all children of r2 executed" );
    ASSERT( r1.ref_count() > 1, "All children of r1 prematurely executed" );
    r1.wait_for_all();
    ASSERT( r1.ref_count() <= 1, "Not all children of r1 executed" );
    r1.destroy(r1);
    r2.destroy(r2);
}


using tbb::internal::spin_wait_until_eq;

//! Deterministic emulation of a long running task
class LongRunningTask : public tbb::task {
    volatile bool& m_CanProceed;

    tbb::task* execute() __TBB_override {
        spin_wait_until_eq( m_CanProceed, true );
        return NULL;
    }
public:
    LongRunningTask ( volatile bool& canProceed ) : m_CanProceed(canProceed) {}
};

void TestWaitDiscriminativenessWithStealing() {
    if( tbb::tbb_thread::hardware_concurrency() < 2 )
        return;
    REMARK( "testing that task::wait_for_all is specific to the root it is called on (one worker)\n" );
    volatile bool canProceed = false;
    tbb::task_scheduler_init init(2);
    tbb::task &r1 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task &r2 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r1.set_ref_count( 2 );
    r2.set_ref_count( 2 );
    tbb::task& t1 = *new( r1.allocate_child() ) tbb::empty_task;
    tbb::task& t2 = *new( r2.allocate_child() ) LongRunningTask(canProceed);
    tbb::task::spawn(t2);
    tbb::task::spawn(t1);
    r1.wait_for_all();
    ASSERT( r1.ref_count() <= 1, "Not all children of r1 executed" );
    ASSERT( r2.ref_count() == 2, "All children of r2 prematurely executed" );
    canProceed = true;
    r2.wait_for_all();
    ASSERT( r2.ref_count() <= 1, "Not all children of r2 executed" );
    r1.destroy(r1);
    r2.destroy(r2);
}

struct MasterBody : NoAssign, Harness::NoAfterlife {
    static Harness::SpinBarrier my_barrier;

    class BarrenButLongTask : public tbb::task {
        volatile bool& m_Started;
        volatile bool& m_CanProceed;

        tbb::task* execute() __TBB_override {
            m_Started = true;
            spin_wait_until_eq( m_CanProceed, true );
            volatile int k = 0;
            for ( int i = 0; i < 1000000; ++i ) ++k;
            return NULL;
        }
    public:
        BarrenButLongTask ( volatile bool& started, volatile bool& can_proceed )
            : m_Started(started), m_CanProceed(can_proceed)
        {}
    };

    class BinaryRecursiveTask : public tbb::task {
        int m_Depth;

        tbb::task* execute() __TBB_override {
            if( !m_Depth )
                return NULL;
            set_ref_count(3);
            spawn( *new( allocate_child() ) BinaryRecursiveTask(m_Depth - 1) );
            spawn( *new( allocate_child() ) BinaryRecursiveTask(m_Depth - 1) );
            wait_for_all();
            return NULL;
        }

        void note_affinity( affinity_id ) __TBB_override {
            ASSERT( false, "These tasks cannot be stolen" );
        }
    public:
        BinaryRecursiveTask ( int depth_ ) : m_Depth(depth_) {}
    };

    void operator() ( int id ) const {
        if ( id ) {
            tbb::task_scheduler_init init(2);
            volatile bool child_started = false,
                          can_proceed = false;
            tbb::task& r = *new( tbb::task::allocate_root() ) tbb::empty_task;
            r.set_ref_count(2);
            r.spawn( *new(r.allocate_child()) BarrenButLongTask(child_started, can_proceed) );
            spin_wait_until_eq( child_started, true );
            my_barrier.wait();
            can_proceed = true;
            r.wait_for_all();
            r.destroy(r);
        }
        else {
            my_barrier.wait();
            tbb::task_scheduler_init init(1);
            Count = 0;
            int depth = 16;
            BinaryRecursiveTask& r = *new( tbb::task::allocate_root() ) BinaryRecursiveTask(depth);
            tbb::task::spawn_root_and_wait(r);
        }
    }
public:
    MasterBody ( int num_masters ) { my_barrier.initialize(num_masters); }
};

Harness::SpinBarrier MasterBody::my_barrier;

/** Ensures that tasks spawned by a master thread or one of the workers servicing
    it cannot be stolen by another master thread. **/
void TestMastersIsolation ( int p ) {
    // The test requires at least 3-way parallelism to work correctly
    if ( p > 2 && tbb::task_scheduler_init::default_num_threads() >= p ) {
        tbb::task_scheduler_init init(p);
        NativeParallelFor( p, MasterBody(p) );
    }
}

struct waitable_task : tbb::task {
    tbb::task* execute() __TBB_override {
        recycle_as_safe_continuation(); // do not destroy the task after execution
        set_parent(this);               // decrement its own ref_count after completion
        __TBB_Yield();
        return NULL;
    }
};
void TestWaitableTask() {
    waitable_task &wt = *new( tbb::task::allocate_root() ) waitable_task;
    for( int i = 0; i < 100000; i++ ) {
        wt.set_ref_count(2);            // prepare for waiting on it
        wt.spawn(wt);
        if( i&1 ) __TBB_Yield();
        wt.wait_for_all();
    }
    wt.set_parent(NULL);                // prevents assertions and atomics in task::destroy
    tbb::task::destroy(wt);
}

#if __TBB_PREVIEW_CRITICAL_TASKS && __TBB_TASK_PRIORITY
#include <stdexcept>
#include <vector>
#include <map>
#include "tbb/parallel_for.h"

namespace CriticalTaskSupport {

using tbb::task;
task* g_root_task = NULL;

// markers to capture execution profile (declaration order is important)
enum task_marker_t {
    no_task, regular_task, isolated_regular_task,
    outer_critical_task, nested_critical_task, critical_from_isolated_task, bypassed_critical_task
};
enum bypassed_critical_task_stage_t { not_bypassed, bypassed, executed };

typedef std::vector< std::vector<task_marker_t> > task_map_t;
task_map_t g_execution_profile;

const int g_per_thread_regular_tasks_num = 5;
const int g_isolated_regular_task_num = 3;
tbb::atomic<bool> g_is_critical_task_submitted;
size_t g_bypassed_critical_task_index = size_t(-1);
task* g_bypassed_task_pointer = NULL;
int g_bypassed_task_creator = -1;
tbb::atomic<bypassed_critical_task_stage_t> g_bypassed_critical_task_stage;
tbb::task_arena g_arena;
Harness::SpinBarrier g_spin_barrier;

struct parallel_for_body {
    parallel_for_body(task_marker_t task_marker, bool submit_critical = false)
        : my_task_marker(task_marker), my_submit_critical(submit_critical) {}
    void operator()( int i ) const;
private:
    task_marker_t my_task_marker;
    bool my_submit_critical;
};

struct IsolatedFunctor {
    void operator()() const {
        parallel_for_body body(isolated_regular_task, /*submit_critical=*/ true);
        tbb::parallel_for( 0, g_isolated_regular_task_num, body, tbb::simple_partitioner() );
    }
};

struct CriticalTaskBody : public task {
    CriticalTaskBody(task_marker_t task_marker) : my_task_mark(task_marker) {}
    task* execute() __TBB_override {
        task* ret_task = NULL;
        task* nested_task = NULL;
        int thread_idx = tbb::this_task_arena::current_thread_index();
        g_execution_profile[thread_idx].push_back(my_task_mark);
        switch( my_task_mark ) {
        case outer_critical_task:
            g_spin_barrier.wait(); // allow each thread to take its own critical task
            // prefill queue with critical tasks
            nested_task = new( task::allocate_additional_child_of(*g_root_task) )
                CriticalTaskBody(nested_critical_task);
            enqueue( *nested_task, tbb::priority_t(tbb::internal::priority_critical) );
            if( not_bypassed ==
                g_bypassed_critical_task_stage.compare_and_swap(bypassed, not_bypassed) ) {

                // first, should process all the work from isolated region
                tbb::this_task_arena::isolate( IsolatedFunctor() );

                CriticalTaskBody* bypassed_task =
                    new( task::allocate_additional_child_of(*g_root_task) )
                    CriticalTaskBody(bypassed_critical_task);
                g_bypassed_task_pointer = bypassed_task;
                g_bypassed_critical_task_index = g_execution_profile[thread_idx].size() + 1;
                g_bypassed_task_creator = thread_idx;
                tbb::internal::make_critical(*bypassed_task);
                ret_task = bypassed_task;
            }
            g_spin_barrier.wait(); // allow thread to execute isolated region
            break;
        case nested_critical_task:
            // wait until bypassed critical task has been executed
            g_spin_barrier.wait();
            break;
        case bypassed_critical_task:
            ASSERT( bypassed == g_bypassed_critical_task_stage, "Unexpected bypassed critical task" );
            g_bypassed_critical_task_stage = executed;
            ASSERT( thread_idx == g_bypassed_task_creator,
                    "Bypassed critical task is not being executed by the thread that bypassed it." );
            ASSERT( g_bypassed_task_pointer == this, "This is not bypassed task." );
            ASSERT( g_bypassed_critical_task_index == g_execution_profile[thread_idx].size(),
                    "Bypassed critical task was not selected as the next task." );
            break;
        case critical_from_isolated_task:
            break;
        default:
            ASSERT( false, "Incorrect critical task id." );
        }
        return ret_task;
    }
private:
    task_marker_t my_task_mark;
};

void parallel_for_body::operator()( int i ) const {
    int thread_idx = tbb::this_task_arena::current_thread_index();
    g_execution_profile[thread_idx].push_back(my_task_marker);
    if( my_submit_critical && i == 0 ) {
        task* isolated_task = new( task::allocate_additional_child_of(*g_root_task) )
            CriticalTaskBody(critical_from_isolated_task);
        task::enqueue( *isolated_task, tbb::priority_t(tbb::internal::priority_critical) );
    }
}

struct TaskBody: public task {
    TaskBody() {}
    TaskBody(task_marker_t /*mark*/) {}
    task* execute() __TBB_override {
        int thread_idx = tbb::this_task_arena::current_thread_index();
        g_execution_profile[thread_idx].push_back(regular_task);
        if( !g_is_critical_task_submitted ) {
            g_spin_barrier.wait(); // allow each thread to take its own task.
            // prefill task pools with regular tasks
            int half = g_per_thread_regular_tasks_num / 2;
            for( int i = 0; i < half; ++i ) {
                task& t = *new( task::allocate_additional_child_of(*g_root_task) )
                    TaskBody;
                spawn(t);
            }
            {
                // prefill with critical tasks
                task& t = *new( task::allocate_additional_child_of(*g_root_task) )
                    CriticalTaskBody(outer_critical_task);
                tbb::internal::make_critical(t);
                tbb::task::spawn(t);
            }
            // prefill task pools with regular tasks
            for( int i = half; i < g_per_thread_regular_tasks_num; ++i ) {
                task& t = *new( task::allocate_additional_child_of(*g_root_task) )
                    TaskBody;
                spawn(t);
            }
            g_is_critical_task_submitted.store<tbb::relaxed>(true);
            g_spin_barrier.wait();
        }
        return NULL;
    }
};

template<typename TaskType, void(*submit_task)(task&)>
struct WorkCreator {
    WorkCreator(task*& root_task, size_t num_tasks, size_t num_critical_tasks = 0,
                tbb::task_group_context* ctx = NULL)
        : my_root_task(root_task), my_num_tasks(num_tasks), my_num_critical_tasks(num_critical_tasks),
          my_context(ctx) {}
    void operator()() const {
        ASSERT( my_root_task == NULL, "Incorrect test set up." );
        task* root_task = NULL;
        if( my_context )
            root_task = new( task::allocate_root(*my_context) ) TaskType(regular_task);
        else
            root_task = new( task::allocate_root() ) TaskType(regular_task);
        root_task->increment_ref_count();
        for( size_t i = 0; i < my_num_tasks; ++i ) {
            task& t = *new( task::allocate_additional_child_of(*root_task) ) TaskType(regular_task);
            submit_task(t);
        }
        for( size_t i = 0; i < my_num_critical_tasks; ++i ) {
            task& t = *new( task::allocate_additional_child_of(*root_task) )
                TaskType( outer_critical_task );
            tbb::task::enqueue( t, tbb::priority_t(tbb::internal::priority_critical) );
        }
        my_root_task = root_task;
    }
private:
    task*& my_root_task;
    size_t my_num_tasks;
    size_t my_num_critical_tasks;
    tbb::task_group_context* my_context;
};

struct WorkAwaiter {
    WorkAwaiter(task*& root_task) : my_root_task(root_task) {}
    void operator()() const {
        while( !my_root_task ) __TBB_Yield(); // waiting on a tree construction
        my_root_task->wait_for_all();
        task::destroy(*my_root_task);
        my_root_task = NULL;
    }
private:
    task*& my_root_task;
};

void TestSchedulerTaskSelectionWhenSpawn() {
    REMARK( "\tPreferring critical tasks among spawned\n" );
    typedef std::multimap<task_marker_t, task_marker_t> state_machine_t;
    typedef state_machine_t::iterator states_it;
    task_marker_t from_to_pairs[] = {
        // from regular
        regular_task, regular_task,
        regular_task, outer_critical_task,
        // from outermost critical
        outer_critical_task, isolated_regular_task,
        outer_critical_task, critical_from_isolated_task,
        outer_critical_task, nested_critical_task,
        // from isolated regular
        isolated_regular_task, isolated_regular_task,
        isolated_regular_task, critical_from_isolated_task,
        isolated_regular_task, bypassed_critical_task,
        // from critical that was enqueued from isolated region
        critical_from_isolated_task, isolated_regular_task,
        critical_from_isolated_task, nested_critical_task,
        critical_from_isolated_task, regular_task,
        critical_from_isolated_task, bypassed_critical_task,
        // from bypassed critical
        bypassed_critical_task, nested_critical_task,
        bypassed_critical_task, critical_from_isolated_task,
        // from nested critical
        nested_critical_task, critical_from_isolated_task,
        nested_critical_task, regular_task
    };

    state_machine_t allowed_transitions;
    for( size_t i = 0; i < sizeof(from_to_pairs) / sizeof(from_to_pairs[0]); i += 2 )
        allowed_transitions.insert( std::make_pair( from_to_pairs[i], from_to_pairs[i+1] ) );

    for( int num_threads = MinThread; num_threads <= MaxThread; ++num_threads ) {
        for( int repeat = 0; repeat < 10; ++repeat ) {
            // test initialization
            g_bypassed_critical_task_stage = not_bypassed;
            g_is_critical_task_submitted = false;
            g_bypassed_critical_task_index = size_t(-1);
            g_bypassed_task_creator = -1;
            g_bypassed_task_pointer = NULL;
            g_execution_profile.resize(num_threads);
            g_spin_barrier.initialize(num_threads);
            g_arena.initialize(num_threads);

            // test execution
            g_arena.execute(
                WorkCreator<TaskBody, task::spawn>(g_root_task, /*num_tasks=*/size_t(num_threads)) );
            g_arena.execute( WorkAwaiter(g_root_task) );

            // checking how execution went
            int critical_task_count = 0;
            for( int thread = 0; thread < num_threads; ++thread ) {
                bool started_critical_region = false;
                bool pass_through_critical_region = false;
                size_t thread_task_num = g_execution_profile[thread].size();
                for( size_t task_index = 0; task_index < thread_task_num; ++task_index ) {
                    const task_marker_t& executed_task = g_execution_profile[thread][task_index];

                    if( pass_through_critical_region ) {
                        ASSERT( executed_task < outer_critical_task,
                                "Thread did not process all the critical work at once." );
                    } else if( isolated_regular_task <= executed_task &&
                               executed_task <= bypassed_critical_task) {
                        started_critical_region = true;
                        if( isolated_regular_task < executed_task )
                            ++critical_task_count;
                        if( bypassed_critical_task == executed_task ) {
                            size_t expected_bypass_task_min_index =
                                /* number of regular task before critical region */1 +
                                /* number of outermost critical tasks before isolated region */ 1 +
                                g_isolated_regular_task_num;
                            size_t expected_bypass_task_max_index = expected_bypass_task_min_index +
                                /* number of critical tasks inside isolated region */ 1;
                            ASSERT( expected_bypass_task_min_index <= task_index &&
                                    task_index <= expected_bypass_task_max_index,
                                    "Bypassed critical task has been executed in wrong order" );
                        }
                    } else if( started_critical_region ) {
                        pass_through_critical_region = true;
                        started_critical_region = false;
                    }

                    if( thread_task_num - 1 == task_index )
                        continue;   // no transition check for the last executed task
                    const task_marker_t& next_task = g_execution_profile[thread][task_index + 1];
                    std::pair<states_it, states_it> range =
                        allowed_transitions.equal_range( executed_task );
                    bool is_choosen_task_allowed = false;
                    for (states_it it = range.first; it != range.second; ++it) {
                        is_choosen_task_allowed |= next_task == it->second;
                    }
                    ASSERT( is_choosen_task_allowed, "Thread chose incorrect task for execution." );
                }
            }
            ASSERT( critical_task_count == 2 * num_threads + 2, "Wrong number of critical tasks" );
            ASSERT( g_bypassed_critical_task_stage == executed, "Was bypassed critical task executed?" );

            // test deinitialization
            g_execution_profile.clear();
            g_arena.terminate();
        }
    }
}

struct TaskTypeExecutionMarker : public task {
    TaskTypeExecutionMarker( task_marker_t mark ) : my_mark( mark ) {}
    task* execute() __TBB_override {
        g_execution_profile[tbb::this_task_arena::current_thread_index()].push_back( my_mark );
        return NULL;
    }
private:
    task_marker_t my_mark;
};

struct RegularTaskMarkChecker {
    bool operator()(const task_marker_t& m) { return regular_task == m; }
};

void TestSchedulerTaskSelectionWhenEnqueue() {
    REMARK( "\tPreferring critical tasks among enqueued\n" );
    g_execution_profile.clear();
    // creating two profiles because of enforced concurrency
    g_execution_profile.resize(2);
    g_root_task = NULL;
    unsigned task_num = 99;
    unsigned num_critical_tasks = 1;
    g_arena.initialize( /*num_threads=*/1, /*reserved_for_masters=*/0 );
    g_arena.enqueue(
        WorkCreator<TaskTypeExecutionMarker, task::enqueue>(
            g_root_task, task_num, num_critical_tasks)
    );
    WorkAwaiter awaiter(g_root_task); awaiter(); // waiting outside arena
    g_arena.terminate();

    unsigned idx = !g_execution_profile[1].empty();
    ASSERT( g_execution_profile[!idx].empty(), "" );

    ASSERT( g_execution_profile[idx].size() == task_num + num_critical_tasks,
            "Incorrect number of tasks executed" );
    ASSERT( *(g_execution_profile[idx].end() - 1) == outer_critical_task,
            "Critical task was executed in wrong order. It should be the last one." );
    bool all_regular = true;
    for( std::vector<task_marker_t>::const_iterator it = g_execution_profile[idx].begin();
         it != g_execution_profile[idx].end() - 1; ++it )
        all_regular &= regular_task == *it;
    ASSERT( all_regular, "Critical task was executed in wrong order. It should be the last one." );
}

enum ways_to_cancel_t {
    by_explicit_call = 0,
    by_exception,
    no_cancellation
};

tbb::atomic<size_t> g_num_executed_from_cancelled_context;
tbb::atomic<size_t> g_num_executed_from_working_context;
int g_cancelling_task_id = -1;

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (push)
#pragma warning (disable: 4127)  /* suppress conditional expression is constant */
#endif

template<bool cancelled_group>
struct ATask : public task {
    ATask( task_marker_t /*mark*/ ) : my_cancellation_method( no_cancellation ) {}
    ATask( ways_to_cancel_t cancellation_method ) : my_cancellation_method( cancellation_method ) {}
    task* execute() __TBB_override {
        while( ! g_is_critical_task_submitted ) __TBB_Yield();
        // scheduler should take critical task as the next task for execution.
        bypassed_critical_task_stage_t previous_critical_task_stage =
            g_bypassed_critical_task_stage.compare_and_swap(bypassed, not_bypassed);
        while(
            cancelled_group                             // Only tasks from cancelled group wait
            && !this->is_cancelled()                    // for their group to be cancelled
            && !tbb::internal::is_critical(*this)       // allowing thread that took critical task
            && bypassed == previous_critical_task_stage // to proceed and cancel the whole group.
        ) __TBB_Yield();
        if( cancelled_group )
            ++g_num_executed_from_cancelled_context;
        else
            ++g_num_executed_from_working_context;
        switch( my_cancellation_method ) {
        case by_explicit_call:
            g_cancelling_task_id = int(g_num_executed_from_cancelled_context);
            self().cancel_group_execution();
            break;
        case by_exception:
            g_cancelling_task_id = int(g_num_executed_from_cancelled_context);
            throw std::runtime_error("Exception data");
            break;
        case no_cancellation: break;
        default:
            ASSERT( false, "Should not be here!" );
            break;
        }
        return NULL;
    }
private:
    ways_to_cancel_t my_cancellation_method;
};

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (pop)
#endif

template<void(*submit_task)(task&)>
struct SubmitTaskFunctor {
    SubmitTaskFunctor( task& t ) : my_task( t ) {}
    void operator()() const {
        submit_task(my_task);
    }
private:
    task& my_task;
};

void TestCancellation(bool cancel_by_exception) {
    g_is_critical_task_submitted = false;
    g_bypassed_critical_task_stage = not_bypassed;
    tbb::task_group_context context_to_leave_working;
    tbb::task_group_context context_to_cancel;
    task* root_task_of_to_be_cancelled_context = NULL;
    task* root_task_of_working_to_completion_context = NULL;
    size_t task_num = 64;
    size_t task_num_for_cancelled_context = 2 * MaxThread;
    g_num_executed_from_cancelled_context = g_num_executed_from_working_context = 0;
    g_cancelling_task_id = -1;
    g_arena.initialize( MaxThread ); // leaving one slot to be occupied by master to submit the work
    g_arena.execute(
        WorkCreator<ATask</*cancelled_group=*/true>, task::spawn>
        (root_task_of_to_be_cancelled_context, task_num_for_cancelled_context,
         /*num_critical_tasks=*/0, &context_to_cancel)
    );
    g_arena.execute(
        WorkCreator<ATask</*cancelled_group=*/false>, task::spawn>
        (root_task_of_working_to_completion_context, task_num, /*num_critical_tasks=*/1,
         &context_to_leave_working)
    );
    ways_to_cancel_t cancellation_method = ways_to_cancel_t( cancel_by_exception );
    task& terminating_task = *new( task::allocate_additional_child_of(*root_task_of_to_be_cancelled_context) )
        ATask</*cancelled_group=*/true>( cancellation_method );
    tbb::internal::make_critical( terminating_task ); // stop the work as soon as possible!
    g_arena.enqueue( SubmitTaskFunctor<task::enqueue>(terminating_task),
                     tbb::priority_t(tbb::internal::priority_critical) );
    g_is_critical_task_submitted = true;
    try {
        g_arena.execute( WorkAwaiter(root_task_of_to_be_cancelled_context) );
    } catch( const std::runtime_error& e ) {
        ASSERT( cancel_by_exception, "Exception was not expected!" );
        ASSERT( std::string(e.what()) == "Exception data", "Unexpected exception data!" );
    } catch( const tbb::captured_exception& e ) {
        ASSERT( cancel_by_exception, "Exception was not expected!" );
        ASSERT( std::string(e.what()) == "Exception data", "Unexpected exception data!" );
    } catch( ... ) {
        ASSERT( false, "Failed to catch specific exception" );
    }
    g_arena.execute( WorkAwaiter(root_task_of_working_to_completion_context) );
    g_arena.terminate();

    if( !cancel_by_exception ) {
        ASSERT( context_to_cancel.is_group_execution_cancelled(), "Execution must be cancelled" );
    }
    ASSERT( !context_to_leave_working.is_group_execution_cancelled(),
            "Execution must NOT be cancelled" );

    ASSERT( g_num_executed_from_working_context == task_num + /*one critical*/1,
            "Incorrect number of tasks executed!" );
    ASSERT( g_num_executed_from_cancelled_context < task_num_for_cancelled_context,
            "Number of executed tasks from the cancelled context should be less than submitted!" );
    ASSERT( 0 < g_cancelling_task_id && g_cancelling_task_id < MaxThread + 1,
            "Critical task was executed in wrong order." );
}

void TestCancellationSupport(bool cancel_by_exception) {
    const char* test_type[] = { "by explicit call to cancel", "by throwing an exception" };
    REMARK( "\tCancellation support %s\n", test_type[!!cancel_by_exception] );
    TestCancellation( cancel_by_exception );
}

namespace NestedArenaCase {

static const size_t g_num_critical_tasks = 10;
static const size_t g_num_critical_nested = 5;

struct CriticalTask : public task {
    CriticalTask(task_marker_t /*mark*/) {}
    task* execute() __TBB_override {
        ++g_num_executed_from_working_context;
        task* nested_root = NULL;
        if( !g_is_critical_task_submitted ) {
            g_is_critical_task_submitted = true;
            g_arena.execute(
                WorkCreator<CriticalTask, task::spawn>(nested_root, /*num_tasks=*/size_t(0),
                                                       g_num_critical_nested) );
            g_arena.execute( WorkAwaiter(nested_root) );
        }
        return NULL;
    }
};

void TestInNestedArena(tbb::task_arena& outer_arena) {
    g_root_task = NULL;
    g_is_critical_task_submitted = false;
    g_num_executed_from_working_context = 0;
    g_arena.initialize( 1 );
    outer_arena.execute(
        WorkCreator<CriticalTask, task::spawn>(
            g_root_task, /*num_tasks=*/size_t(0), g_num_critical_tasks) );
    outer_arena.execute( WorkAwaiter(g_root_task) );
    ASSERT( g_num_executed_from_working_context == g_num_critical_tasks + g_num_critical_nested,
            "Mismatch in number of critical tasks executed in nested and outer arenas." );
    g_arena.terminate();
}

void test() {
    REMARK( "\tWork in nested arenas\n" );
    TestInNestedArena( g_arena );

    tbb::task_arena a( 1 );
    TestInNestedArena( a );
}
} // namespace NestedArenaCase

void test() {
    REMARK("Testing support for critical tasks\n");
    TestSchedulerTaskSelectionWhenSpawn();
    TestSchedulerTaskSelectionWhenEnqueue();
    TestCancellationSupport(/*cancel_by_exception=*/false);
    TestCancellationSupport(/*cancel_by_exception=*/true);
    NestedArenaCase::test();
}
} // namespace CriticalTaskSupport
#endif /* __TBB_PREVIEW_CRITICAL_TASKS && __TBB_TASK_PRIORITY */

int TestMain () {
#if TBB_USE_EXCEPTIONS
    TestUnconstructibleTask<1>();
    TestUnconstructibleTask<10000>();
#endif
    TestAlignment();
    TestNoteAffinityContext();
    TestDispatchLoopResponsiveness();
    TestWaitDiscriminativenessWithoutStealing();
    TestWaitDiscriminativenessWithStealing();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        TestSpawnChildren( p );
        TestSpawnRootList( p );
        TestSafeContinuation( p );
        TestLeftRecursion( p );
        TestDag( p );
        TestAffinity( p );
        TestUserThread( p );
        TestStealLimit( p );
        TestRelaxedOwnership( p );
        TestMastersIsolation( p );
    }
    TestWaitableTask();
#if __TBB_PREVIEW_CRITICAL_TASKS && __TBB_TASK_PRIORITY
    CriticalTaskSupport::test();
#endif
    return Harness::Done;
}
