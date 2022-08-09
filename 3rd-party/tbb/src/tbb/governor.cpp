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

#include <stdio.h>
#include <stdlib.h>
#include "governor.h"
#include "tbb_main.h"
#include "scheduler.h"
#include "market.h"
#include "arena.h"

#include "tbb/task_scheduler_init.h"

#include "dynamic_link.h"

namespace tbb {
namespace internal {

//------------------------------------------------------------------------
// governor
//------------------------------------------------------------------------

#if __TBB_SURVIVE_THREAD_SWITCH
// Support for interoperability with Intel(R) Cilk(TM) Plus.

#if _WIN32
#define CILKLIB_NAME "cilkrts20.dll"
#else
#define CILKLIB_NAME "libcilkrts.so"
#endif

//! Handler for interoperation with cilkrts library.
static __cilk_tbb_retcode (*watch_stack_handler)(struct __cilk_tbb_unwatch_thunk* u,
                                                 struct __cilk_tbb_stack_op_thunk o);

//! Table describing how to link the handlers.
static const dynamic_link_descriptor CilkLinkTable[] = {
    DLD_NOWEAK(__cilkrts_watch_stack, watch_stack_handler)
};

static atomic<do_once_state> cilkrts_load_state;

bool initialize_cilk_interop() {
    // Pinning can fail. This is a normal situation, and means that the current
    // thread does not use cilkrts and consequently does not need interop.
    return dynamic_link( CILKLIB_NAME, CilkLinkTable, 1,  /*handle=*/0, DYNAMIC_LINK_GLOBAL );
}
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

namespace rml {
    tbb_server* make_private_server( tbb_client& client );
}

void governor::acquire_resources () {
#if USE_PTHREAD
    int status = theTLS.create(auto_terminate);
#else
    int status = theTLS.create();
#endif
    if( status )
        handle_perror(status, "TBB failed to initialize task scheduler TLS\n");
    is_speculation_enabled = cpu_has_speculation();
    is_rethrow_broken = gcc_rethrow_exception_broken();
}

void governor::release_resources () {
    theRMLServerFactory.close();
    destroy_process_mask();
#if TBB_USE_ASSERT
    if( __TBB_InitOnce::initialization_done() && theTLS.get() ) 
        runtime_warning( "TBB is unloaded while tbb::task_scheduler_init object is alive?" );
#endif
    int status = theTLS.destroy();
    if( status )
        runtime_warning("failed to destroy task scheduler TLS: %s", strerror(status));
    dynamic_unlink_all();
}

rml::tbb_server* governor::create_rml_server ( rml::tbb_client& client ) {
    rml::tbb_server* server = NULL;
    if( !UsePrivateRML ) {
        ::rml::factory::status_type status = theRMLServerFactory.make_server( server, client );
        if( status != ::rml::factory::st_success ) {
            UsePrivateRML = true;
            runtime_warning( "rml::tbb_factory::make_server failed with status %x, falling back on private rml", status );
        }
    }
    if ( !server ) {
        __TBB_ASSERT( UsePrivateRML, NULL );
        server = rml::make_private_server( client );
    }
    __TBB_ASSERT( server, "Failed to create RML server" );
    return server;
}


uintptr_t governor::tls_value_of( generic_scheduler* s ) {
    __TBB_ASSERT( (uintptr_t(s)&1) == 0, "Bad pointer to the scheduler" );
    // LSB marks the scheduler initialized with arena
    return uintptr_t(s) | uintptr_t((s && (s->my_arena || s->is_worker()))? 1 : 0);
}

void governor::assume_scheduler( generic_scheduler* s ) {
    theTLS.set( tls_value_of(s) );
}

bool governor::is_set( generic_scheduler* s ) {
    return theTLS.get() == tls_value_of(s);
}

void governor::sign_on(generic_scheduler* s) {
    __TBB_ASSERT( is_set(NULL) && s, NULL );
    assume_scheduler( s );
#if __TBB_SURVIVE_THREAD_SWITCH
    if( watch_stack_handler ) {
        __cilk_tbb_stack_op_thunk o;
        o.routine = &stack_op_handler;
        o.data = s;
        if( (*watch_stack_handler)(&s->my_cilk_unwatch_thunk, o) ) {
            // Failed to register with cilkrts, make sure we are clean
            s->my_cilk_unwatch_thunk.routine = NULL;
        }
#if TBB_USE_ASSERT
        else
            s->my_cilk_state = generic_scheduler::cs_running;
#endif /* TBB_USE_ASSERT */
    }
#endif /* __TBB_SURVIVE_THREAD_SWITCH */
    __TBB_ASSERT( is_set(s), NULL );
}

void governor::sign_off(generic_scheduler* s) {
    suppress_unused_warning(s);
    __TBB_ASSERT( is_set(s), "attempt to unregister a wrong scheduler instance" );
    assume_scheduler(NULL);
#if __TBB_SURVIVE_THREAD_SWITCH
    __cilk_tbb_unwatch_thunk &ut = s->my_cilk_unwatch_thunk;
    if ( ut.routine )
       (*ut.routine)(ut.data);
#endif /* __TBB_SURVIVE_THREAD_SWITCH */
}

void governor::one_time_init() {
    if( !__TBB_InitOnce::initialization_done() )
        DoOneTimeInitializations();
#if __TBB_SURVIVE_THREAD_SWITCH
    atomic_do_once( &initialize_cilk_interop, cilkrts_load_state );
#endif /* __TBB_SURVIVE_THREAD_SWITCH */
}

generic_scheduler* governor::init_scheduler_weak() {
    one_time_init();
    __TBB_ASSERT( is_set(NULL), "TLS contains a scheduler?" );
    generic_scheduler* s = generic_scheduler::create_master( NULL ); // without arena
    s->my_auto_initialized = true;
    return s;
}

generic_scheduler* governor::init_scheduler( int num_threads, stack_size_type stack_size, bool auto_init ) {
    one_time_init();
    if ( uintptr_t v = theTLS.get() ) {
        generic_scheduler* s = tls_scheduler_of( v );
        if ( (v&1) == 0 ) { // TLS holds scheduler instance without arena
            __TBB_ASSERT( s->my_ref_count == 1, "weakly initialized scheduler must have refcount equal to 1" );
            __TBB_ASSERT( !s->my_arena, "weakly initialized scheduler  must have no arena" );
            __TBB_ASSERT( s->my_auto_initialized, "weakly initialized scheduler is supposed to be auto-initialized" );
            s->attach_arena( market::create_arena( default_num_threads(), 1, 0 ), 0, /*is_master*/true );
            __TBB_ASSERT( s->my_arena_index == 0, "Master thread must occupy the first slot in its arena" );
            s->my_arena_slot->my_scheduler = s;
#if __TBB_TASK_GROUP_CONTEXT
            s->my_arena->my_default_ctx = s->default_context(); // it also transfers implied ownership
#endif
            // Mark the scheduler as fully initialized
            assume_scheduler( s );
        }
        // Increment refcount only for explicit instances of task_scheduler_init.
        if ( !auto_init ) s->my_ref_count += 1;
        __TBB_ASSERT( s->my_arena, "scheduler is not initialized fully" );
        return s;
    }
    // Create new scheduler instance with arena
    if( num_threads == task_scheduler_init::automatic )
        num_threads = default_num_threads();
    arena *a = market::create_arena( num_threads, 1, stack_size );
    generic_scheduler* s = generic_scheduler::create_master( a );
    __TBB_ASSERT(s, "Somehow a local scheduler creation for a master thread failed");
    __TBB_ASSERT( is_set(s), NULL );
    s->my_auto_initialized = auto_init;
    return s;
}

bool governor::terminate_scheduler( generic_scheduler* s, bool blocking ) {
    bool ok = false;
    __TBB_ASSERT( is_set(s), "Attempt to terminate non-local scheduler instance" );
    if (0 == --(s->my_ref_count)) {
        ok = s->cleanup_master( blocking );
        __TBB_ASSERT( is_set(NULL), "cleanup_master has not cleared its TLS slot" );
    }
    return ok;
}

void governor::auto_terminate(void* arg){
    generic_scheduler* s = tls_scheduler_of( uintptr_t(arg) ); // arg is equivalent to theTLS.get()
    if( s && s->my_auto_initialized ) {
        if( !--(s->my_ref_count) ) {
            // If the TLS slot is already cleared by OS or underlying concurrency
            // runtime, restore its value.
            if( !is_set(s) )
                assume_scheduler(s);
            s->cleanup_master( /*blocking_terminate=*/false );
            __TBB_ASSERT( is_set(NULL), "cleanup_master has not cleared its TLS slot" );
        }
    }
}

void governor::print_version_info () {
    if ( UsePrivateRML )
        PrintExtraVersionInfo( "RML", "private" );
    else {
        PrintExtraVersionInfo( "RML", "shared" );
        theRMLServerFactory.call_with_server_info( PrintRMLVersionInfo, (void*)"" );
    }
#if __TBB_SURVIVE_THREAD_SWITCH
    if( watch_stack_handler )
        PrintExtraVersionInfo( "CILK", CILKLIB_NAME );
#endif /* __TBB_SURVIVE_THREAD_SWITCH */
}

void governor::initialize_rml_factory () {
    ::rml::factory::status_type res = theRMLServerFactory.open();
    UsePrivateRML = res != ::rml::factory::st_success;
}

#if __TBB_SURVIVE_THREAD_SWITCH
__cilk_tbb_retcode governor::stack_op_handler( __cilk_tbb_stack_op op, void* data ) {
    __TBB_ASSERT(data,NULL);
    generic_scheduler* s = static_cast<generic_scheduler*>(data);
#if TBB_USE_ASSERT
    void* current = local_scheduler_if_initialized();
#if _WIN32||_WIN64
    uintptr_t thread_id = GetCurrentThreadId();
#else
    uintptr_t thread_id = uintptr_t(pthread_self());
#endif
#endif /* TBB_USE_ASSERT */
    switch( op ) {
        case CILK_TBB_STACK_ADOPT: {
            __TBB_ASSERT( !current && s->my_cilk_state==generic_scheduler::cs_limbo ||
                          current==s && s->my_cilk_state==generic_scheduler::cs_running, "invalid adoption" );
#if TBB_USE_ASSERT
            if( current==s )
                runtime_warning( "redundant adoption of %p by thread %p\n", s, (void*)thread_id );
            s->my_cilk_state = generic_scheduler::cs_running;
#endif /* TBB_USE_ASSERT */
            assume_scheduler( s );
            break;
        }
        case CILK_TBB_STACK_ORPHAN: {
            __TBB_ASSERT( current==s && s->my_cilk_state==generic_scheduler::cs_running, "invalid orphaning" );
#if TBB_USE_ASSERT
            s->my_cilk_state = generic_scheduler::cs_limbo;
#endif /* TBB_USE_ASSERT */
            assume_scheduler(NULL);
            break;
        }
        case CILK_TBB_STACK_RELEASE: {
            __TBB_ASSERT( !current && s->my_cilk_state==generic_scheduler::cs_limbo ||
                          current==s && s->my_cilk_state==generic_scheduler::cs_running, "invalid release" );
#if TBB_USE_ASSERT
            s->my_cilk_state = generic_scheduler::cs_freed;
#endif /* TBB_USE_ASSERT */
            s->my_cilk_unwatch_thunk.routine = NULL;
            auto_terminate( s );
            break;
        }
        default:
            __TBB_ASSERT(0, "invalid op");
    }
    return 0;
}
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

#if __TBB_NUMA_SUPPORT

#if __TBB_WEAK_SYMBOLS_PRESENT
#pragma weak initialize_numa_topology
#pragma weak allocate_binding_handler
#pragma weak deallocate_binding_handler
#pragma weak bind_to_node
#pragma weak restore_affinity

extern "C" {
void initialize_numa_topology(
    size_t groups_num, int& nodes_count, int*& indexes_list, int*& concurrency_list );

binding_handler* allocate_binding_handler( int slot_num );
void deallocate_binding_handler( binding_handler* handler_ptr );

void bind_to_node( binding_handler* handler_ptr, int slot_num, int numa_id );
void restore_affinity( binding_handler* handler_ptr, int slot_num );
}
#endif /* __TBB_WEAK_SYMBOLS_PRESENT */

// Handlers for communication with TBBbind
#if _WIN32 || _WIN64 || __linux__
static void (*initialize_numa_topology_ptr)(
    size_t groups_num, int& nodes_count, int*& indexes_list, int*& concurrency_list ) = NULL;
#endif /* _WIN32 || _WIN64 || __linux__ */

static binding_handler* (*allocate_binding_handler_ptr)( int slot_num ) = NULL;
static void (*deallocate_binding_handler_ptr)( binding_handler* handler_ptr ) = NULL;

static void (*bind_to_node_ptr)( binding_handler* handler_ptr, int slot_num, int numa_id ) = NULL;
static void (*restore_affinity_ptr)( binding_handler* handler_ptr, int slot_num ) = NULL;

#if _WIN32 || _WIN64 || __linux__
// Table describing how to link the handlers.
static const dynamic_link_descriptor TbbBindLinkTable[] = {
    DLD(initialize_numa_topology, initialize_numa_topology_ptr),
    DLD(allocate_binding_handler, allocate_binding_handler_ptr),
    DLD(deallocate_binding_handler, deallocate_binding_handler_ptr),
    DLD(bind_to_node, bind_to_node_ptr),
    DLD(restore_affinity, restore_affinity_ptr)
};

static const unsigned LinkTableSize = 5;

#if TBB_USE_DEBUG
#define DEBUG_SUFFIX "_debug"
#else
#define DEBUG_SUFFIX
#endif /* TBB_USE_DEBUG */

#if _WIN32 || _WIN64
#define TBBBIND_NAME "tbbbind" DEBUG_SUFFIX ".dll"
#elif __linux__
#define TBBBIND_NAME "libtbbbind" DEBUG_SUFFIX  __TBB_STRING(.so.TBB_COMPATIBLE_INTERFACE_VERSION)
#endif /* __linux__ */
#endif /* _WIN32 || _WIN64 || __linux__ */

// Stubs that will be used if TBBbind library is unavailable.
static binding_handler* dummy_allocate_binding_handler ( int ) { return NULL; }
static void dummy_deallocate_binding_handler ( binding_handler* ) { }
static void dummy_bind_to_node ( binding_handler*, int, int ) { }
static void dummy_restore_affinity ( binding_handler*, int ) { }

// Representation of NUMA topology information on the TBB side.
// NUMA topology may be initialized by third-party component (e.g. hwloc)
// or just filled by default stubs (1 NUMA node with 0 index and
// default_num_threads value as default_concurrency).
namespace numa_topology {
namespace {
int  numa_nodes_count = 0;
int* numa_indexes = NULL;
int* default_concurrency_list = NULL;
static tbb::atomic<do_once_state> numa_topology_init_state;
} // internal namespace

// Tries to load TBBbind library API, if success, gets NUMA topology information from it,
// in another case, fills NUMA topology by stubs.
// TODO: Add TBBbind loading status if TBB_VERSION is set.
void initialization_impl() {
    governor::one_time_init();

#if _WIN32 || _WIN64 || __linux__
    bool load_tbbbind = true;
#if _WIN32 && !_WIN64
    // For 32-bit Windows applications, process affinity masks can only support up to 32 logical CPUs.
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    load_tbbbind = si.dwNumberOfProcessors <= 32;
#endif /* _WIN32 && !_WIN64 */

    if (load_tbbbind && dynamic_link(TBBBIND_NAME, TbbBindLinkTable, LinkTableSize)) {
        int number_of_groups = 1;
#if _WIN32 || _WIN64
        number_of_groups = NumberOfProcessorGroups();
#endif /* _WIN32 || _WIN64 */
        initialize_numa_topology_ptr(
            number_of_groups, numa_nodes_count, numa_indexes, default_concurrency_list);

        if (numa_nodes_count==1 && numa_indexes[0] >= 0) {
            __TBB_ASSERT(default_concurrency_list[numa_indexes[0]] == (int)governor::default_num_threads(),
                "default_concurrency() should be equal to governor::default_num_threads() on single"
                "NUMA node systems.");
        }
        return;
    }
#endif /* _WIN32 || _WIN64 || __linux__ */

    static int dummy_index = -1;
    static int dummy_concurrency = governor::default_num_threads();

    numa_nodes_count = 1;
    numa_indexes = &dummy_index;
    default_concurrency_list = &dummy_concurrency;

    allocate_binding_handler_ptr = dummy_allocate_binding_handler;
    deallocate_binding_handler_ptr = dummy_deallocate_binding_handler;

    bind_to_node_ptr = dummy_bind_to_node;
    restore_affinity_ptr = dummy_restore_affinity;
}

void initialize() {
    atomic_do_once(initialization_impl, numa_topology_init_state);
}

unsigned nodes_count() {
    initialize();
    return numa_nodes_count;
}

void fill( int* indexes_array ) {
    initialize();
    for ( int i = 0; i < numa_nodes_count; i++ ) {
        indexes_array[i] = numa_indexes[i];
    }
}

int default_concurrency( int node_id ) {
    if (node_id >= 0) {
        initialize();
        return default_concurrency_list[node_id];
    }
    return governor::default_num_threads();
}

} // namespace numa_topology

binding_handler* construct_binding_handler(int slot_num) {
    __TBB_ASSERT(allocate_binding_handler_ptr, "tbbbind loading was not perfromed");
    return allocate_binding_handler_ptr(slot_num);
}

void destroy_binding_handler(binding_handler* handler_ptr) {
    __TBB_ASSERT(deallocate_binding_handler_ptr, "tbbbind loading was not perfromed");
    deallocate_binding_handler_ptr(handler_ptr);
}

void bind_thread_to_node(binding_handler* handler_ptr, int slot_num , int numa_id) {
    __TBB_ASSERT(slot_num >= 0, "Negative thread index");
    __TBB_ASSERT(bind_to_node_ptr, "tbbbind loading was not perfromed");
    bind_to_node_ptr(handler_ptr, slot_num, numa_id);
}

void restore_affinity_mask(binding_handler* handler_ptr, int slot_num) {
    __TBB_ASSERT(slot_num >= 0, "Negative thread index");
    __TBB_ASSERT(restore_affinity_ptr, "tbbbind loading was not perfromed");
    restore_affinity_ptr(handler_ptr, slot_num);
}

#endif /* __TBB_NUMA_SUPPORT */

} // namespace internal

//------------------------------------------------------------------------
// task_scheduler_init
//------------------------------------------------------------------------

using namespace internal;

/** Left out-of-line for the sake of the backward binary compatibility **/
void task_scheduler_init::initialize( int number_of_threads ) {
    initialize( number_of_threads, 0 );
}

void task_scheduler_init::initialize( int number_of_threads, stack_size_type thread_stack_size ) {
#if __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS
    uintptr_t new_mode = thread_stack_size & propagation_mode_mask;
#endif
    thread_stack_size &= ~(stack_size_type)propagation_mode_mask;
    if( number_of_threads!=deferred ) {
        __TBB_ASSERT_RELEASE( !my_scheduler, "task_scheduler_init already initialized" );
        __TBB_ASSERT_RELEASE( number_of_threads==automatic || number_of_threads > 0,
                    "number_of_threads for task_scheduler_init must be automatic or positive" );
        internal::generic_scheduler *s = governor::init_scheduler( number_of_threads, thread_stack_size, /*auto_init=*/false );
#if __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS
        if ( s->master_outermost_level() ) {
            uintptr_t &vt = s->default_context()->my_version_and_traits;
            uintptr_t prev_mode = vt & task_group_context::exact_exception ? propagation_mode_exact : 0;
            vt = new_mode & propagation_mode_exact ? vt | task_group_context::exact_exception
                    : new_mode & propagation_mode_captured ? vt & ~task_group_context::exact_exception : vt;
            // Use least significant bit of the scheduler pointer to store previous mode.
            // This is necessary when components compiled with different compilers and/or
            // TBB versions initialize the
            my_scheduler = static_cast<scheduler*>((generic_scheduler*)((uintptr_t)s | prev_mode));
        }
        else
#endif /* __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS */
            my_scheduler = s;
    } else {
        __TBB_ASSERT_RELEASE( !thread_stack_size, "deferred initialization ignores stack size setting" );
    }
}

bool task_scheduler_init::internal_terminate( bool blocking ) {
#if __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS
    uintptr_t prev_mode = (uintptr_t)my_scheduler & propagation_mode_exact;
    my_scheduler = (scheduler*)((uintptr_t)my_scheduler & ~(uintptr_t)propagation_mode_exact);
#endif /* __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS */
    generic_scheduler* s = static_cast<generic_scheduler*>(my_scheduler);
    my_scheduler = NULL;
    __TBB_ASSERT_RELEASE( s, "task_scheduler_init::terminate without corresponding task_scheduler_init::initialize()");
#if __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS
    if ( s->master_outermost_level() ) {
        uintptr_t &vt = s->default_context()->my_version_and_traits;
        vt = prev_mode & propagation_mode_exact ? vt | task_group_context::exact_exception
                                        : vt & ~task_group_context::exact_exception;
    }
#endif /* __TBB_TASK_GROUP_CONTEXT && TBB_USE_EXCEPTIONS */
    return governor::terminate_scheduler(s, blocking);
}

void task_scheduler_init::terminate() {
    internal_terminate(/*blocking_terminate=*/false);
}

#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
bool task_scheduler_init::internal_blocking_terminate( bool throwing ) {
    bool ok = internal_terminate( /*blocking_terminate=*/true );
#if TBB_USE_EXCEPTIONS
    if( throwing && !ok )
        throw_exception( eid_blocking_thread_join_impossible );
#else
    suppress_unused_warning( throwing );
#endif
    return ok;
}
#endif // __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE

int task_scheduler_init::default_num_threads() {
    return governor::default_num_threads();
}

} // namespace tbb
