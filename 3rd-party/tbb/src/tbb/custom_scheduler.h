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

#ifndef _TBB_custom_scheduler_H
#define _TBB_custom_scheduler_H

#include "scheduler.h"
#include "observer_proxy.h"
#include "itt_notify.h"

namespace tbb {
namespace internal {

//------------------------------------------------------------------------
//! Traits classes for scheduler
//------------------------------------------------------------------------

struct DefaultSchedulerTraits {
    static const bool itt_possible = true;
    static const bool has_slow_atomic = false;
};

struct IntelSchedulerTraits {
    static const bool itt_possible = false;
#if __TBB_x86_32||__TBB_x86_64
    static const bool has_slow_atomic = true;
#else
    static const bool has_slow_atomic = false;
#endif /* __TBB_x86_32||__TBB_x86_64 */
};

//------------------------------------------------------------------------
// custom_scheduler
//------------------------------------------------------------------------

//! A scheduler with a customized evaluation loop.
/** The customization can use SchedulerTraits to make decisions without needing a run-time check. */
template<typename SchedulerTraits>
class custom_scheduler: private generic_scheduler {
    typedef custom_scheduler<SchedulerTraits> scheduler_type;

    custom_scheduler( market& m, bool genuine ) : generic_scheduler(m, genuine) {}

    //! Scheduler loop that dispatches tasks.
    /** If child is non-NULL, it is dispatched first.
        Then, until "parent" has a reference count of 1, other task are dispatched or stolen. */
    void local_wait_for_all( task& parent, task* child ) __TBB_override;

    //! Entry point from client code to the scheduler loop that dispatches tasks.
    /** The method is virtual, but the *this object is used only for sake of dispatching on the correct vtable,
        not necessarily the correct *this object.  The correct *this object is looked up in TLS. */
    void wait_for_all( task& parent, task* child ) __TBB_override {
        static_cast<custom_scheduler*>(governor::local_scheduler())->scheduler_type::local_wait_for_all( parent, child );
    }

    //! Decrements ref_count of a predecessor.
    /** If it achieves 0, the predecessor is scheduled for execution.
        When changing, remember that this is a hot path function. */
    void tally_completion_of_predecessor( task& s, __TBB_ISOLATION_ARG( task*& bypass_slot, isolation_tag isolation ) ) {
        task_prefix& p = s.prefix();
        __TBB_ASSERT(p.ref_count > 0, NULL);
        if( SchedulerTraits::itt_possible )
            ITT_NOTIFY(sync_releasing, &p.ref_count);
        if( SchedulerTraits::has_slow_atomic && p.ref_count==1 )
            p.ref_count=0;
        else {
            reference_count old_ref_count = __TBB_FetchAndDecrementWrelease(&p.ref_count);
#if __TBB_PREVIEW_RESUMABLE_TASKS
            if (old_ref_count == internal::abandon_flag + 2) {
                // Remove the abandon flag.
                p.ref_count = 1;
                // The wait has been completed. Spawn a resume task.
                tbb::task::resume(p.abandoned_scheduler);
                return;
            }
#endif
            if (old_ref_count > 1) {
                // more references exist
                // '__TBB_cl_evict(&p)' degraded performance of parallel_preorder example
                return;
            }
        }

        // Ordering on p.ref_count (superfluous if SchedulerTraits::has_slow_atomic)
        __TBB_control_consistency_helper();
        __TBB_ASSERT(p.ref_count==0, "completion of task caused predecessor's reference count to underflow");
        if( SchedulerTraits::itt_possible )
            ITT_NOTIFY(sync_acquired, &p.ref_count);
#if TBB_USE_ASSERT
        p.extra_state &= ~es_ref_count_active;
#endif /* TBB_USE_ASSERT */
#if __TBB_TASK_ISOLATION
        if ( isolation != no_isolation ) {
            // The parent is allowed not to have isolation (even if a child has isolation) because it has never spawned.
            __TBB_ASSERT(p.isolation == no_isolation || p.isolation == isolation, NULL);
            p.isolation = isolation;
        }
#endif /* __TBB_TASK_ISOLATION */

#if __TBB_RECYCLE_TO_ENQUEUE
        if (p.state==task::to_enqueue) {
            // related to __TBB_TASK_ARENA TODO: try keep priority of the task
            // e.g. rework task_prefix to remember priority of received task and use here
            my_arena->enqueue_task(s, 0, my_random );
        } else
#endif /*__TBB_RECYCLE_TO_ENQUEUE*/
        if( bypass_slot==NULL )
            bypass_slot = &s;
#if __TBB_PREVIEW_CRITICAL_TASKS
        else if( internal::is_critical( s ) ) {
            local_spawn( bypass_slot, bypass_slot->prefix().next );
            bypass_slot = &s;
        }
#endif /* __TBB_PREVIEW_CRITICAL_TASKS */
        else
            local_spawn( &s, s.prefix().next );
    }

    //! Implements the bypass loop of the dispatch loop (local_wait_for_all).
    bool process_bypass_loop( context_guard_helper</*report_tasks=*/SchedulerTraits::itt_possible>& context_guard,
        __TBB_ISOLATION_ARG(task* t, isolation_tag isolation) );

public:
    static generic_scheduler* allocate_scheduler( market& m, bool genuine ) {
        void* p = NFS_Allocate(1, sizeof(scheduler_type), NULL);
        std::memset(p, 0, sizeof(scheduler_type));
        scheduler_type* s = new( p ) scheduler_type( m, genuine );
        s->assert_task_pool_valid();
        ITT_SYNC_CREATE(s, SyncType_Scheduler, SyncObj_TaskPoolSpinning);
        return s;
    }

    //! Try getting a task from the mailbox or stealing from another scheduler.
    /** Returns the stolen task or NULL if all attempts fail. */
    task* receive_or_steal_task( __TBB_ISOLATION_ARG( __TBB_atomic reference_count& completion_ref_count, isolation_tag isolation ) ) __TBB_override;

}; // class custom_scheduler<>

//------------------------------------------------------------------------
// custom_scheduler methods
//------------------------------------------------------------------------
template<typename SchedulerTraits>
task* custom_scheduler<SchedulerTraits>::receive_or_steal_task( __TBB_ISOLATION_ARG(__TBB_atomic reference_count& completion_ref_count, isolation_tag isolation) ) {
    task* t = NULL;
    bool outermost_worker_level = worker_outermost_level();
    bool outermost_dispatch_level = outermost_worker_level || master_outermost_level();
    bool can_steal_here = can_steal();
    bool outermost_current_worker_level = outermost_worker_level;
#if __TBB_PREVIEW_RESUMABLE_TASKS
    outermost_current_worker_level &= my_properties.genuine;
#endif
    my_inbox.set_is_idle( true );
#if __TBB_HOARD_NONLOCAL_TASKS
    __TBB_ASSERT(!my_nonlocal_free_list, NULL);
#endif
#if __TBB_TASK_PRIORITY
    if ( outermost_dispatch_level ) {
        if ( intptr_t skipped_priority = my_arena->my_skipped_fifo_priority ) {
            // This thread can dequeue FIFO tasks, and some priority levels of
            // FIFO tasks have been bypassed (to prevent deadlock caused by
            // dynamic priority changes in nested task group hierarchy).
            if ( my_arena->my_skipped_fifo_priority.compare_and_swap(0, skipped_priority) == skipped_priority
                 && skipped_priority > my_arena->my_top_priority )
            {
                my_market->update_arena_priority( *my_arena, skipped_priority );
            }
        }
    }
#endif /* !__TBB_TASK_PRIORITY */
    // TODO: Try to find a place to reset my_limit (under market's lock)
    // The number of slots potentially used in the arena. Updated once in a while, as my_limit changes rarely.
    size_t n = my_arena->my_limit-1;
    int yield_count = 0;
    // The state "failure_count==-1" is used only when itt_possible is true,
    // and denotes that a sync_prepare has not yet been issued.
    for( int failure_count = -static_cast<int>(SchedulerTraits::itt_possible);; ++failure_count) {
        __TBB_ASSERT( my_arena->my_limit > 0, NULL );
        __TBB_ASSERT( my_arena_index <= n, NULL );
        if( completion_ref_count == 1 ) {
            if( SchedulerTraits::itt_possible ) {
                if( failure_count!=-1 ) {
                    ITT_NOTIFY(sync_prepare, &completion_ref_count);
                    // Notify Intel(R) Thread Profiler that thread has stopped spinning.
                    ITT_NOTIFY(sync_acquired, this);
                }
                ITT_NOTIFY(sync_acquired, &completion_ref_count);
            }
            __TBB_ASSERT( !t, NULL );
            // A worker thread in its outermost dispatch loop (i.e. its execution stack is empty) should
            // exit it either when there is no more work in the current arena, or when revoked by the market.
            __TBB_ASSERT( !outermost_worker_level, NULL );
            __TBB_control_consistency_helper(); // on ref_count
            break; // exit stealing loop and return;
        }
        // Check if the resource manager requires our arena to relinquish some threads
        if ( outermost_current_worker_level ) {
            if ( ( my_arena->my_num_workers_allotted < my_arena->num_workers_active() ) ) {
                if ( SchedulerTraits::itt_possible && failure_count != -1 )
                    ITT_NOTIFY(sync_cancel, this);
                return NULL;
            }
        }
#if __TBB_PREVIEW_RESUMABLE_TASKS
        else if ( *my_arena_slot->my_scheduler_is_recalled ) {
            // Original scheduler was requested, return from stealing loop and recall.
            if ( my_inbox.is_idle_state(true) )
                my_inbox.set_is_idle(false);
            return NULL;
        }
#endif
#if __TBB_TASK_PRIORITY
        const int p = int(my_arena->my_top_priority);
#else /* !__TBB_TASK_PRIORITY */
        static const int p = 0;
#endif
        // Check if there are tasks mailed to this thread via task-to-thread affinity mechanism.
        __TBB_ASSERT(my_affinity_id, NULL);
        if ( n && !my_inbox.empty() ) {
            t = get_mailbox_task( __TBB_ISOLATION_EXPR( isolation ) );
#if __TBB_TASK_ISOLATION
            // There is a race with a thread adding a new task (possibly with suitable isolation)
            // to our mailbox, so the below conditions might result in a false positive.
            // Then set_is_idle(false) allows that task to be stolen; it's OK.
            if ( isolation != no_isolation && !t && !my_inbox.empty()
                     && my_inbox.is_idle_state( true ) ) {
                // We have proxy tasks in our mailbox but the isolation blocks their execution.
                // So publish the proxy tasks in mailbox to be available for stealing from owner's task pool.
                my_inbox.set_is_idle( false );
            }
#endif /* __TBB_TASK_ISOLATION */
        }
        if ( t ) {
            GATHER_STATISTIC( ++my_counters.mails_received );
        }
        // Check if there are tasks in starvation-resistant stream.
        // Only allowed at the outermost dispatch level without isolation.
        else if (__TBB_ISOLATION_EXPR(isolation == no_isolation &&) outermost_dispatch_level &&
                 !my_arena->my_task_stream.empty(p) && (
#if __TBB_PREVIEW_CRITICAL_TASKS && __TBB_CPF_BUILD
                     t = my_arena->my_task_stream.pop( p, subsequent_lane_selector(my_arena_slot->hint_for_pop) )
#else
                     t = my_arena->my_task_stream.pop( p, my_arena_slot->hint_for_pop )
#endif
                 ) ) {
            ITT_NOTIFY(sync_acquired, &my_arena->my_task_stream);
            // just proceed with the obtained task
        }
#if __TBB_TASK_PRIORITY
        // Check if any earlier offloaded non-top priority tasks become returned to the top level
        else if ( my_offloaded_tasks && (t = reload_tasks( __TBB_ISOLATION_EXPR( isolation ) )) ) {
            __TBB_ASSERT( !is_proxy(*t), "The proxy task cannot be offloaded" );
            // just proceed with the obtained task
        }
#endif /* __TBB_TASK_PRIORITY */
        else if ( can_steal_here && n && (t = steal_task( __TBB_ISOLATION_EXPR(isolation) )) ) {
            // just proceed with the obtained task
        }
#if __TBB_PREVIEW_CRITICAL_TASKS
        else if( (t = get_critical_task( __TBB_ISOLATION_EXPR(isolation) )) ) {
            __TBB_ASSERT( internal::is_critical(*t), "Received task must be critical one" );
            ITT_NOTIFY(sync_acquired, &my_arena->my_critical_task_stream);
            // just proceed with the obtained task
        }
#endif // __TBB_PREVIEW_CRITICAL_TASKS
        else
            goto fail;
        // A task was successfully obtained somewhere
        __TBB_ASSERT(t,NULL);
#if __TBB_ARENA_OBSERVER
        my_arena->my_observers.notify_entry_observers( my_last_local_observer, is_worker() );
#endif
#if __TBB_SCHEDULER_OBSERVER
        the_global_observer_list.notify_entry_observers( my_last_global_observer, is_worker() );
#endif /* __TBB_SCHEDULER_OBSERVER */
        if ( SchedulerTraits::itt_possible && failure_count != -1 ) {
            // FIXME - might be victim, or might be selected from a mailbox
            // Notify Intel(R) Thread Profiler that thread has stopped spinning.
            ITT_NOTIFY(sync_acquired, this);
        }
        break; // exit stealing loop and return
fail:
        GATHER_STATISTIC( ++my_counters.steals_failed );
        if( SchedulerTraits::itt_possible && failure_count==-1 ) {
            // The first attempt to steal work failed, so notify Intel(R) Thread Profiler that
            // the thread has started spinning.  Ideally, we would do this notification
            // *before* the first failed attempt to steal, but at that point we do not
            // know that the steal will fail.
            ITT_NOTIFY(sync_prepare, this);
            failure_count = 0;
        }
        // Pause, even if we are going to yield, because the yield might return immediately.
        prolonged_pause();
        const int failure_threshold = 2*int(n+1);
        if( failure_count>=failure_threshold ) {
#if __TBB_YIELD2P
            failure_count = 0;
#else
            failure_count = failure_threshold;
#endif
            __TBB_Yield();
#if __TBB_TASK_PRIORITY
            // Check if there are tasks abandoned by other workers
            if ( my_arena->my_orphaned_tasks ) {
                // Epoch must be advanced before seizing the list pointer
                ++my_arena->my_abandonment_epoch;
                task* orphans = (task*)__TBB_FetchAndStoreW( &my_arena->my_orphaned_tasks, 0 );
                if ( orphans ) {
                    task** link = NULL;
                    // Get local counter out of the way (we've just brought in external tasks)
                    my_local_reload_epoch--;
                    t = reload_tasks( orphans, link, __TBB_ISOLATION_ARG( effective_reference_priority(), isolation ) );
                    if ( orphans ) {
                        *link = my_offloaded_tasks;
                        if ( !my_offloaded_tasks )
                            my_offloaded_task_list_tail_link = link;
                        my_offloaded_tasks = orphans;
                    }
                    __TBB_ASSERT( !my_offloaded_tasks == !my_offloaded_task_list_tail_link, NULL );
                    if ( t ) {
                        if( SchedulerTraits::itt_possible )
                            ITT_NOTIFY(sync_cancel, this);
                        __TBB_ASSERT( !is_proxy(*t), "The proxy task cannot be offloaded" );
                        break; // exit stealing loop and return
                    }
                }
            }
#endif /* __TBB_TASK_PRIORITY */
#if __APPLE__
            // threshold value tuned separately for macOS due to high cost of sched_yield there
            const int yield_threshold = 10;
#else
            const int yield_threshold = 100;
#endif
            if( yield_count++ >= yield_threshold ) {
                // When a worker thread has nothing to do, return it to RML.
                // For purposes of affinity support, the thread is considered idle while in RML.
#if __TBB_TASK_PRIORITY
                if( outermost_current_worker_level || my_arena->my_top_priority > my_arena->my_bottom_priority ) {
                    if ( my_arena->is_out_of_work() && outermost_current_worker_level ) {
#else /* !__TBB_TASK_PRIORITY */
                    if ( outermost_current_worker_level && my_arena->is_out_of_work() ) {
#endif /* !__TBB_TASK_PRIORITY */
                        if( SchedulerTraits::itt_possible )
                            ITT_NOTIFY(sync_cancel, this);
                        return NULL;
                    }
#if __TBB_TASK_PRIORITY
                }
                if ( my_offloaded_tasks ) {
                    // Safeguard against any sloppiness in managing reload epoch
                    // counter (e.g. on the hot path because of performance reasons).
                    my_local_reload_epoch--;
                    // Break the deadlock caused by a higher priority dispatch loop
                    // stealing and offloading a lower priority task. Priority check
                    // at the stealing moment cannot completely preclude such cases
                    // because priorities can changes dynamically.
                    if ( !outermost_worker_level && *my_ref_top_priority > my_arena->my_top_priority ) {
                        GATHER_STATISTIC( ++my_counters.prio_ref_fixups );
                        my_ref_top_priority = &my_arena->my_top_priority;
                        // it's expected that only outermost workers can use global reload epoch
                        __TBB_ASSERT(my_ref_reload_epoch == &my_arena->my_reload_epoch, NULL);
                    }
                }
#endif /* __TBB_TASK_PRIORITY */
            } // end of arena snapshot branch
            // If several attempts did not find work, re-read the arena limit.
            n = my_arena->my_limit-1;
        } // end of yielding branch
    } // end of nonlocal task retrieval loop
    if ( my_inbox.is_idle_state( true ) )
        my_inbox.set_is_idle( false );
    return t;
}

template<typename SchedulerTraits>
bool custom_scheduler<SchedulerTraits>::process_bypass_loop(
            context_guard_helper</*report_tasks=*/SchedulerTraits::itt_possible>& context_guard,
            __TBB_ISOLATION_ARG(task* t, isolation_tag isolation) )
{
    while ( t ) {
        __TBB_ASSERT( my_inbox.is_idle_state(false), NULL );
        __TBB_ASSERT(!is_proxy(*t),"unexpected proxy");
        __TBB_ASSERT( t->prefix().owner, NULL );
#if __TBB_TASK_ISOLATION
        __TBB_ASSERT_EX( isolation == no_isolation || isolation == t->prefix().isolation,
            "A task from another isolated region is going to be executed" );
#endif /* __TBB_TASK_ISOLATION */
        assert_task_valid(t);
#if __TBB_TASK_GROUP_CONTEXT && TBB_USE_ASSERT
        assert_context_valid(t->prefix().context);
        if ( !t->prefix().context->my_cancellation_requested )
#endif
        // TODO: make the assert stronger by prohibiting allocated state.
        __TBB_ASSERT( 1L<<t->state() & (1L<<task::allocated|1L<<task::ready|1L<<task::reexecute), NULL );
        assert_task_pool_valid();
#if __TBB_PREVIEW_CRITICAL_TASKS
        // TODO: check performance and optimize if needed for added conditions on the
        // hot-path.
        if( !internal::is_critical(*t) && !t->is_enqueued_task() ) {
            if( task* critical_task = get_critical_task( __TBB_ISOLATION_EXPR(isolation) ) ) {
                __TBB_ASSERT( internal::is_critical(*critical_task),
                              "Received task must be critical one" );
                ITT_NOTIFY(sync_acquired, &my_arena->my_critical_task_stream);
                t->prefix().state = task::allocated;
                my_innermost_running_task = t; // required during spawn to propagate isolation
                local_spawn(t, t->prefix().next);
                t = critical_task;
            } else {
#endif /* __TBB_PREVIEW_CRITICAL_TASKS */
#if __TBB_TASK_PRIORITY
                intptr_t p = priority(*t);
                if ( p != *my_ref_top_priority
                     && !t->is_enqueued_task() ) {
                    assert_priority_valid(p);
                    if ( p != my_arena->my_top_priority ) {
                        my_market->update_arena_priority( *my_arena, p );
                    }
                    if ( p < effective_reference_priority() ) {
                        if ( !my_offloaded_tasks ) {
                            my_offloaded_task_list_tail_link = &t->prefix().next_offloaded;
                            // Erase possible reference to the owner scheduler
                            // (next_offloaded is a union member)
                            *my_offloaded_task_list_tail_link = NULL;
                        }
                        offload_task( *t, p );
                        t = NULL;
                        if ( is_task_pool_published() ) {
                            t = winnow_task_pool( __TBB_ISOLATION_EXPR( isolation ) );
                            if ( t )
                                continue;
                        } else {
                            // Mark arena as full to unlock arena priority level adjustment
                            // by arena::is_out_of_work(), and ensure worker's presence.
                            my_arena->advertise_new_work<arena::wakeup>();
                        }
                        break; /* exit bypass loop */
                    }
                }
#endif /* __TBB_TASK_PRIORITY */
#if __TBB_PREVIEW_CRITICAL_TASKS
            }
        } // if is not critical
#endif
        task* t_next = NULL;
        my_innermost_running_task = t;
        t->prefix().owner = this;
        t->prefix().state = task::executing;
#if __TBB_TASK_GROUP_CONTEXT
        context_guard.set_ctx( t->prefix().context );
        if ( !t->prefix().context->my_cancellation_requested )
#endif
        {
            GATHER_STATISTIC( ++my_counters.tasks_executed );
            GATHER_STATISTIC( my_counters.avg_arena_concurrency += my_arena->num_workers_active() );
            GATHER_STATISTIC( my_counters.avg_assigned_workers += my_arena->my_num_workers_allotted );
#if __TBB_TASK_PRIORITY
            GATHER_STATISTIC( my_counters.avg_arena_prio += p );
            GATHER_STATISTIC( my_counters.avg_market_prio += my_market->my_global_top_priority );
#endif /* __TBB_TASK_PRIORITY */
            ITT_STACK(SchedulerTraits::itt_possible, callee_enter, t->prefix().context->itt_caller);
            t_next = t->execute();
            ITT_STACK(SchedulerTraits::itt_possible, callee_leave, t->prefix().context->itt_caller);
            if (t_next) {
                assert_task_valid(t_next);
                __TBB_ASSERT( t_next->state()==task::allocated,
                              "if task::execute() returns task, it must be marked as allocated" );
                reset_extra_state(t_next);
                __TBB_ISOLATION_EXPR( t_next->prefix().isolation = t->prefix().isolation );
#if TBB_USE_ASSERT
                affinity_id next_affinity=t_next->prefix().affinity;
                if (next_affinity != 0 && next_affinity != my_affinity_id)
                    GATHER_STATISTIC( ++my_counters.affinity_ignored );
#endif
            } // if there is bypassed task
        }
        assert_task_pool_valid();
        switch( t->state() ) {
            case task::executing: {
                task* s = t->parent();
                __TBB_ASSERT( my_innermost_running_task==t, NULL );
                __TBB_ASSERT( t->prefix().ref_count==0, "Task still has children after it has been executed" );
                t->~task();
                if( s )
                    tally_completion_of_predecessor( *s, __TBB_ISOLATION_ARG( t_next, t->prefix().isolation ) );
                free_task<no_hint>( *t );
                poison_pointer( my_innermost_running_task );
                assert_task_pool_valid();
                break;
            }

            case task::recycle: // set by recycle_as_safe_continuation()
                t->prefix().state = task::allocated;
#if __TBB_RECYCLE_TO_ENQUEUE
                __TBB_fallthrough;
            case task::to_enqueue: // set by recycle_to_enqueue()
#endif
                __TBB_ASSERT( t_next != t, "a task returned from method execute() can not be recycled in another way" );
                reset_extra_state(t);
                // for safe continuation, need atomically decrement ref_count;
                tally_completion_of_predecessor(*t, __TBB_ISOLATION_ARG( t_next, t->prefix().isolation ) );
                assert_task_pool_valid();
                break;

            case task::reexecute: // set by recycle_to_reexecute()
                __TBB_ASSERT( t_next, "reexecution requires that method execute() return another task" );
                __TBB_ASSERT( t_next != t, "a task returned from method execute() can not be recycled in another way" );
                t->prefix().state = task::allocated;
                reset_extra_state(t);
                local_spawn( t, t->prefix().next );
                assert_task_pool_valid();
                break;
            case task::allocated:
                reset_extra_state(t);
                break;
#if __TBB_PREVIEW_RESUMABLE_TASKS
            case task::to_resume:
                __TBB_ASSERT(my_innermost_running_task == t, NULL);
                __TBB_ASSERT(t->prefix().ref_count == 0, "Task still has children after it has been executed");
                t->~task();
                free_task<no_hint>(*t);
                __TBB_ASSERT(!my_properties.genuine && my_properties.outermost,
                    "Only a coroutine on outermost level can be left.");
                // Leave the outermost coroutine
                return false;
#endif
#if TBB_USE_ASSERT
            case task::ready:
                __TBB_ASSERT( false, "task is in READY state upon return from method execute()" );
                break;
#endif
            default:
                __TBB_ASSERT( false, "illegal state" );
                break;
        }
        GATHER_STATISTIC( t_next ? ++my_counters.spawns_bypassed : 0 );
        t = t_next;
    } // end of scheduler bypass loop
    return true;
}

// TODO: Rename args 'parent' into 'controlling_task' and 'child' into 't' or consider introducing
// a wait object (a la task_handle) to replace the 'parent' logic.
template<typename SchedulerTraits>
void custom_scheduler<SchedulerTraits>::local_wait_for_all( task& parent, task* child ) {
    __TBB_ASSERT( governor::is_set(this), NULL );
    __TBB_ASSERT( parent.ref_count() >= (child && child->parent() == &parent ? 2 : 1), "ref_count is too small" );
    __TBB_ASSERT( my_innermost_running_task, NULL );
#if __TBB_TASK_GROUP_CONTEXT
    __TBB_ASSERT( parent.prefix().context, "parent task does not have context" );
#endif /* __TBB_TASK_GROUP_CONTEXT */
    assert_task_pool_valid();
    // Using parent's refcount in sync_prepare (in the stealing loop below) is
    // a workaround for TP. We need to name it here to display correctly in Ampl.
    if( SchedulerTraits::itt_possible )
        ITT_SYNC_CREATE(&parent.prefix().ref_count, SyncType_Scheduler, SyncObj_TaskStealingLoop);

    // TODO: consider extending the "context" guard to a "dispatch loop" guard to additionally
    // guard old_innermost_running_task and old_properties states.
    context_guard_helper</*report_tasks=*/SchedulerTraits::itt_possible> context_guard;
    task* old_innermost_running_task = my_innermost_running_task;
    scheduler_properties old_properties = my_properties;

    task* t = child;
    bool cleanup = !is_worker() && &parent==my_dummy_task;
    // Remove outermost property to indicate nested level.
    __TBB_ASSERT(my_properties.outermost || my_innermost_running_task!=my_dummy_task, "The outermost property should be set out of a dispatch loop");
    my_properties.outermost &= my_innermost_running_task==my_dummy_task;
#if __TBB_PREVIEW_CRITICAL_TASKS
    my_properties.has_taken_critical_task |= is_critical(*my_innermost_running_task);
#endif
#if __TBB_TASK_PRIORITY
    __TBB_ASSERT( (uintptr_t)*my_ref_top_priority < (uintptr_t)num_priority_levels, NULL );
    volatile intptr_t *old_ref_top_priority = my_ref_top_priority;
    // When entering nested parallelism level market level counter
    // must be replaced with the one local to this arena.
    volatile uintptr_t *old_ref_reload_epoch = my_ref_reload_epoch;
    if ( !outermost_level() ) {
        // We are in a nested dispatch loop.
        // Market or arena priority must not prevent child tasks from being
        // executed so that dynamic priority changes did not cause deadlock.
        my_ref_top_priority = &parent.prefix().context->my_priority;
        my_ref_reload_epoch = &my_arena->my_reload_epoch;
        if (my_ref_reload_epoch != old_ref_reload_epoch)
            my_local_reload_epoch = *my_ref_reload_epoch - 1;
    }
#endif /* __TBB_TASK_PRIORITY */
#if __TBB_TASK_ISOLATION
    isolation_tag isolation = my_innermost_running_task->prefix().isolation;
    if (t && isolation != no_isolation) {
        __TBB_ASSERT(t->prefix().isolation == no_isolation, NULL);
        // Propagate the isolation to the task executed without spawn.
        t->prefix().isolation = isolation;
    }
#endif /* __TBB_TASK_ISOLATION */
#if __TBB_PREVIEW_RESUMABLE_TASKS
    // The recall flag for the original owner of this scheduler.
    // It is used only on outermost level of currently attached arena slot.
    tbb::atomic<bool> recall_flag;
    recall_flag = false;
    if (outermost_level() && my_wait_task == NULL && my_properties.genuine) {
        __TBB_ASSERT(my_arena_slot->my_scheduler == this, NULL);
        __TBB_ASSERT(my_arena_slot->my_scheduler_is_recalled == NULL, NULL);
        my_arena_slot->my_scheduler_is_recalled = &recall_flag;
        my_current_is_recalled = &recall_flag;
    }
    __TBB_ASSERT(my_arena_slot->my_scheduler_is_recalled != NULL, NULL);
    task* old_wait_task = my_wait_task;
    my_wait_task = &parent;
#endif
#if TBB_USE_EXCEPTIONS
    // Infinite safeguard EH loop
    for (;;) {
    try {
#endif /* TBB_USE_EXCEPTIONS */
    // Outer loop receives tasks from global environment (via mailbox, FIFO queue(s),
    // and by  stealing from other threads' task pools).
    // All exit points from the dispatch loop are located in its immediate scope.
    for(;;) {
        // Middle loop retrieves tasks from the local task pool.
        for(;;) {
            // Inner loop evaluates tasks coming from nesting loops and those returned
            // by just executed tasks (bypassing spawn or enqueue calls).
            if ( !process_bypass_loop( context_guard, __TBB_ISOLATION_ARG(t, isolation) ) ) {
#if __TBB_PREVIEW_RESUMABLE_TASKS
                // Restore the old properties for the coroutine reusage (leave in a valid state)
                my_innermost_running_task = old_innermost_running_task;
                my_properties = old_properties;
                my_wait_task = old_wait_task;
#endif
                return;
            }

            // Check "normal" exit condition when parent's work is done.
            if ( parent.prefix().ref_count == 1 ) {
                __TBB_ASSERT( !cleanup, NULL );
                __TBB_control_consistency_helper(); // on ref_count
                ITT_NOTIFY( sync_acquired, &parent.prefix().ref_count );
                goto done;
            }
#if __TBB_PREVIEW_RESUMABLE_TASKS
            // The thread may be otside of its original scheduler. Check the recall request.
            if ( &recall_flag != my_arena_slot->my_scheduler_is_recalled ) {
                __TBB_ASSERT( my_arena_slot->my_scheduler_is_recalled != NULL, "A broken recall flag" );
                if ( *my_arena_slot->my_scheduler_is_recalled ) {
                    if ( !resume_original_scheduler() ) {
                        // We are requested to finish the current coroutine before the resume.
                        __TBB_ASSERT( !my_properties.genuine && my_properties.outermost,
                            "Only a coroutine on outermost level can be left." );
                        // Restore the old properties for the coroutine reusage (leave in a valid state)
                        my_innermost_running_task = old_innermost_running_task;
                        my_properties = old_properties;
                        my_wait_task = old_wait_task;
                        return;
                    }
                }
            }
#endif
            // Retrieve the task from local task pool.
            __TBB_ASSERT( is_task_pool_published() || is_quiescent_local_task_pool_reset(), NULL );
            t = is_task_pool_published() ? get_task( __TBB_ISOLATION_EXPR( isolation ) ) : NULL;
            assert_task_pool_valid();

            if ( !t ) // No tasks in the local task pool. Go to stealing loop.
                break;
        }; // end of local task pool retrieval loop

#if __TBB_HOARD_NONLOCAL_TASKS
        // before stealing, previously stolen task objects are returned
        for (; my_nonlocal_free_list; my_nonlocal_free_list = t ) {
            t = my_nonlocal_free_list->prefix().next;
            free_nonlocal_small_task( *my_nonlocal_free_list );
        }
#endif
        if ( cleanup ) {
            __TBB_ASSERT( !is_task_pool_published() && is_quiescent_local_task_pool_reset(), NULL );
            __TBB_ASSERT( !worker_outermost_level(), NULL );
            my_innermost_running_task = old_innermost_running_task;
            my_properties = old_properties;
#if __TBB_TASK_PRIORITY
            my_ref_top_priority = old_ref_top_priority;
            if(my_ref_reload_epoch != old_ref_reload_epoch)
                my_local_reload_epoch = *old_ref_reload_epoch-1;
            my_ref_reload_epoch = old_ref_reload_epoch;
#endif /* __TBB_TASK_PRIORITY */
#if __TBB_PREVIEW_RESUMABLE_TASKS
            if (&recall_flag != my_arena_slot->my_scheduler_is_recalled) {
                // The recall point
                __TBB_ASSERT(!recall_flag, NULL);
                tbb::task::suspend(recall_functor(&recall_flag));
                if (my_inbox.is_idle_state(true))
                    my_inbox.set_is_idle(false);
                continue;
            }
            __TBB_ASSERT(&recall_flag == my_arena_slot->my_scheduler_is_recalled, NULL);
            __TBB_ASSERT(!(my_wait_task->prefix().ref_count & internal::abandon_flag), NULL);
            my_wait_task = old_wait_task;
#endif
            return;
        }
        t = receive_or_steal_task( __TBB_ISOLATION_ARG( parent.prefix().ref_count, isolation ) );
        if ( !t ) {
#if __TBB_PREVIEW_RESUMABLE_TASKS
            if ( *my_arena_slot->my_scheduler_is_recalled )
                continue;
            // Done if either original thread enters or we are on the nested level or attached the same arena
            if ( &recall_flag == my_arena_slot->my_scheduler_is_recalled || old_wait_task != NULL )
                goto done;
            // The recall point. Continue dispatch loop because recalled thread may have tasks in it's task pool.
            __TBB_ASSERT(!recall_flag, NULL);
            tbb::task::suspend( recall_functor(&recall_flag) );
            if ( my_inbox.is_idle_state(true) )
                my_inbox.set_is_idle(false);
#else
            // Just exit dispatch loop
            goto done;
#endif
        }
    } // end of infinite stealing loop
#if TBB_USE_EXCEPTIONS
    __TBB_ASSERT( false, "Must never get here" );
    } // end of try-block
    TbbCatchAll( my_innermost_running_task->prefix().context );
    t = my_innermost_running_task;
    // Complete post-processing ...
    if( t->state() == task::recycle
#if __TBB_RECYCLE_TO_ENQUEUE
        // TODO: the enqueue semantics gets lost below, consider reimplementing
        ||  t->state() == task::to_enqueue
#endif
      ) {
        // ... for recycled tasks to atomically decrement ref_count
        t->prefix().state = task::allocated;
        if( SchedulerTraits::itt_possible )
            ITT_NOTIFY(sync_releasing, &t->prefix().ref_count);
        if( __TBB_FetchAndDecrementWrelease(&t->prefix().ref_count)==1 ) {
            if( SchedulerTraits::itt_possible )
                ITT_NOTIFY(sync_acquired, &t->prefix().ref_count);
        }else{
            t = NULL;
        }
    }
    } // end of infinite EH loop
    __TBB_ASSERT( false, "Must never get here too" );
#endif /* TBB_USE_EXCEPTIONS */
done:
#if __TBB_PREVIEW_RESUMABLE_TASKS
    __TBB_ASSERT(!(parent.prefix().ref_count & internal::abandon_flag), NULL);
    my_wait_task = old_wait_task;
    if (my_wait_task == NULL) {
        __TBB_ASSERT(outermost_level(), "my_wait_task could be NULL only on outermost level");
        if (&recall_flag != my_arena_slot->my_scheduler_is_recalled) {
            // The recall point.
            __TBB_ASSERT(my_properties.genuine, NULL);
            __TBB_ASSERT(!recall_flag, NULL);
            tbb::task::suspend(recall_functor(&recall_flag));
            if (my_inbox.is_idle_state(true))
                my_inbox.set_is_idle(false);
        }
        __TBB_ASSERT(my_arena_slot->my_scheduler == this, NULL);
        my_arena_slot->my_scheduler_is_recalled = NULL;
        my_current_is_recalled = NULL;
    }

#endif /* __TBB_PREVIEW_RESUMABLE_TASKS */
    my_innermost_running_task = old_innermost_running_task;
    my_properties = old_properties;
#if __TBB_TASK_PRIORITY
    my_ref_top_priority = old_ref_top_priority;
    if(my_ref_reload_epoch != old_ref_reload_epoch)
        my_local_reload_epoch = *old_ref_reload_epoch-1;
    my_ref_reload_epoch = old_ref_reload_epoch;
#endif /* __TBB_TASK_PRIORITY */
    if ( !ConcurrentWaitsEnabled(parent) ) {
        if ( parent.prefix().ref_count != 1) {
            // This is a worker that was revoked by the market.
            __TBB_ASSERT( worker_outermost_level(),
                "Worker thread exits nested dispatch loop prematurely" );
            return;
        }
        parent.prefix().ref_count = 0;
    }
#if TBB_USE_ASSERT
    parent.prefix().extra_state &= ~es_ref_count_active;
#endif /* TBB_USE_ASSERT */
#if __TBB_TASK_GROUP_CONTEXT
    __TBB_ASSERT(parent.prefix().context && default_context(), NULL);
    task_group_context* parent_ctx = parent.prefix().context;
    if ( parent_ctx->my_cancellation_requested ) {
        task_group_context::exception_container_type *pe = parent_ctx->my_exception;
        if ( master_outermost_level() && parent_ctx == default_context() ) {
            // We are in the outermost task dispatch loop of a master thread, and
            // the whole task tree has been collapsed. So we may clear cancellation data.
            parent_ctx->my_cancellation_requested = 0;
            // TODO: Add assertion that master's dummy task context does not have children
            parent_ctx->my_state &= ~(uintptr_t)task_group_context::may_have_children;
        }
        if ( pe ) {
            // On Windows, FPU control settings changed in the helper destructor are not visible
            // outside a catch block. So restore the default settings manually before rethrowing
            // the exception.
            context_guard.restore_default();
            TbbRethrowException( pe );
        }
    }
    __TBB_ASSERT(!is_worker() || !CancellationInfoPresent(*my_dummy_task),
        "Worker's dummy task context modified");
    __TBB_ASSERT(!master_outermost_level() || !CancellationInfoPresent(*my_dummy_task),
        "Unexpected exception or cancellation data in the master's dummy task");
#endif /* __TBB_TASK_GROUP_CONTEXT */
    assert_task_pool_valid();
}

} // namespace internal
} // namespace tbb

#endif /* _TBB_custom_scheduler_H */
