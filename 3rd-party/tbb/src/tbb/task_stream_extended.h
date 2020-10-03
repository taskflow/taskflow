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

#ifndef _TBB_task_stream_extended_H
#define _TBB_task_stream_extended_H

//! This file is a possible future replacement for the task_stream class implemented in
//! task_stream.h. It refactors the code and extends task_stream capabilities by moving lane
//! management during operations on caller side. Despite the fact that new implementation should not
//! affect performance of the original task stream, analysis on this subject was not made at the
//! time it was developed. In addition, it is not clearly seen at the moment that this container
//! would be suitable for critical tasks due to linear time complexity on its operations.


#if _TBB_task_stream_H
#error Either task_stream.h or this file can be included at the same time.
#endif

#if !__TBB_CPF_BUILD
#error This code bears a preview status until it proves its usefulness/performance suitability.
#endif

#include "tbb/tbb_stddef.h"
#include <deque>
#include <climits>
#include "tbb/atomic.h" // for __TBB_Atomic*
#include "tbb/spin_mutex.h"
#include "tbb/tbb_allocator.h"
#include "scheduler_common.h"
#include "tbb_misc.h" // for FastRandom

namespace tbb {
namespace internal {

//! Essentially, this is just a pair of a queue and a mutex to protect the queue.
/** The reason std::pair is not used is that the code would look less clean
    if field names were replaced with 'first' and 'second'. **/
template< typename T, typename mutex_t >
struct queue_and_mutex {
    typedef std::deque< T, tbb_allocator<T> > queue_base_t;

    queue_base_t my_queue;
    mutex_t      my_mutex;

    queue_and_mutex () : my_queue(), my_mutex() {}
    ~queue_and_mutex () {}
};

typedef uintptr_t population_t;
const population_t one = 1;

inline void set_one_bit( population_t& dest, int pos ) {
    __TBB_ASSERT( pos>=0, NULL );
    __TBB_ASSERT( pos<int(sizeof(population_t)*CHAR_BIT), NULL );
    __TBB_AtomicOR( &dest, one<<pos );
}

inline void clear_one_bit( population_t& dest, int pos ) {
    __TBB_ASSERT( pos>=0, NULL );
    __TBB_ASSERT( pos<int(sizeof(population_t)*CHAR_BIT), NULL );
    __TBB_AtomicAND( &dest, ~(one<<pos) );
}

inline bool is_bit_set( population_t val, int pos ) {
    __TBB_ASSERT( pos>=0, NULL );
    __TBB_ASSERT( pos<int(sizeof(population_t)*CHAR_BIT), NULL );
    return (val & (one<<pos)) != 0;
}

struct random_lane_selector :
#if __INTEL_COMPILER == 1110 || __INTEL_COMPILER == 1500
        no_assign
#else
        no_copy
#endif
{
    random_lane_selector( FastRandom& random ) : my_random( random ) {}
    unsigned operator()( unsigned out_of ) const {
        __TBB_ASSERT( ((out_of-1) & out_of) == 0, "number of lanes is not power of two." );
        return my_random.get() & (out_of-1);
    }
private:
    FastRandom& my_random;
};

struct lane_selector_base :
#if __INTEL_COMPILER == 1110 || __INTEL_COMPILER == 1500
        no_assign
#else
        no_copy
#endif
{
    unsigned& my_previous;
    lane_selector_base( unsigned& previous ) : my_previous( previous ) {}
};

struct subsequent_lane_selector : lane_selector_base {
    subsequent_lane_selector( unsigned& previous ) : lane_selector_base( previous ) {}
    unsigned operator()( unsigned out_of ) const {
        __TBB_ASSERT( ((out_of-1) & out_of) == 0, "number of lanes is not power of two." );
        return (++my_previous &= out_of-1);
    }
};

struct preceding_lane_selector : lane_selector_base {
    preceding_lane_selector( unsigned& previous ) : lane_selector_base( previous ) {}
    unsigned operator()( unsigned out_of ) const {
        __TBB_ASSERT( ((out_of-1) & out_of) == 0, "number of lanes is not power of two." );
        return (--my_previous &= (out_of-1));
    }
};

class task_stream_base : no_copy {
protected:
    typedef queue_and_mutex <task*, spin_mutex> lane_t;
};

enum task_stream_accessor_type { front_accessor = 0, back_nonnull_accessor };

//! Specializes from which side of the underlying container elements are retrieved. Method must be
//! called under corresponding mutex locked.
template<task_stream_accessor_type accessor>
class task_stream_accessor : public task_stream_base {
protected:
    using task_stream_base::lane_t;
    task* get_item( lane_t::queue_base_t& queue ) {
        task* result = queue.front();
        queue.pop_front();
        return result;
    }
};

template<>
class task_stream_accessor< back_nonnull_accessor > : public task_stream_base {
protected:
    task* get_item( lane_t::queue_base_t& queue ) {
        task* result = NULL;
        do {
            result = queue.back();
            queue.pop_back();
        } while( !result && !queue.empty() );
        return result;
    }
};

//! The container for "fairness-oriented" aka "enqueued" tasks.
template<int Levels, task_stream_accessor_type accessor>
class task_stream : public task_stream_accessor< accessor > {
    typedef typename task_stream_accessor<accessor>::lane_t lane_t;
    population_t population[Levels];
    padded<lane_t>* lanes[Levels];
    unsigned N;

public:
    task_stream() : N() {
        for(int level = 0; level < Levels; level++) {
            population[level] = 0;
            lanes[level] = NULL;
        }
    }

    void initialize( unsigned n_lanes ) {
        const unsigned max_lanes = sizeof(population_t) * CHAR_BIT;

        N = n_lanes>=max_lanes ? max_lanes : n_lanes>2 ? 1<<(__TBB_Log2(n_lanes-1)+1) : 2;
        __TBB_ASSERT( N==max_lanes || N>=n_lanes && ((N-1)&N)==0, "number of lanes miscalculated");
        __TBB_ASSERT( N <= sizeof(population_t) * CHAR_BIT, NULL );
        for(int level = 0; level < Levels; level++) {
            lanes[level] = new padded<lane_t>[N];
            __TBB_ASSERT( !population[level], NULL );
        }
    }

    ~task_stream() {
        for(int level = 0; level < Levels; level++)
            if (lanes[level]) delete[] lanes[level];
    }

    //! Returns true on successful push, otherwise - false.
    bool try_push( task* source, int level, unsigned lane_idx ) {
        __TBB_ASSERT( 0 <= level && level < Levels, "Incorrect lane level specified." );
        spin_mutex::scoped_lock lock;
        if( lock.try_acquire( lanes[level][lane_idx].my_mutex ) ) {
            lanes[level][lane_idx].my_queue.push_back( source );
            set_one_bit( population[level], lane_idx ); // TODO: avoid atomic op if the bit is already set
            return true;
        }
        return false;
    }

    //! Push a task into a lane. Lane selection is performed by passed functor.
    template<typename lane_selector_t>
    void push( task* source, int level, const lane_selector_t& next_lane ) {
        bool succeed = false;
        unsigned lane = 0;
        do {
            lane = next_lane( /*out_of=*/N );
            __TBB_ASSERT( lane < N, "Incorrect lane index." );
        } while( ! (succeed = try_push( source, level, lane )) );
    }

    //! Returns pointer to task on successful pop, otherwise - NULL.
    task* try_pop( int level, unsigned lane_idx ) {
        __TBB_ASSERT( 0 <= level && level < Levels, "Incorrect lane level specified." );
        if( !is_bit_set( population[level], lane_idx ) )
            return NULL;
        task* result = NULL;
        lane_t& lane = lanes[level][lane_idx];
        spin_mutex::scoped_lock lock;
        if( lock.try_acquire( lane.my_mutex ) && !lane.my_queue.empty() ) {
            result = this->get_item( lane.my_queue );
            if( lane.my_queue.empty() )
                clear_one_bit( population[level], lane_idx );
        }
        return result;
    }

    //! Try finding and popping a task using passed functor for lane selection. Last used lane is
    //! updated inside lane selector.
    template<typename lane_selector_t>
    task* pop( int level, const lane_selector_t& next_lane ) {
        task* popped = NULL;
        unsigned lane = 0;
        do {
            lane = next_lane( /*out_of=*/N );
            __TBB_ASSERT( lane < N, "Incorrect lane index." );
        } while( !empty( level ) && !(popped = try_pop( level, lane )) );
        return popped;
    }

    // TODO: unify '*_specific' logic with 'pop' methods above
    task* look_specific( __TBB_ISOLATION_ARG(task_stream_base::lane_t::queue_base_t& queue, isolation_tag isolation) ) {
        __TBB_ASSERT( !queue.empty(), NULL );
        // TODO: add a worst-case performance test and consider an alternative container with better
        // performance for isolation search.
        typename lane_t::queue_base_t::iterator curr = queue.end();
        do {
            // TODO: consider logic from get_task to simplify the code.
            task* result = *--curr;
            if( result __TBB_ISOLATION_EXPR( && result->prefix().isolation == isolation ) ) {
                if( queue.end() - curr == 1 )
                    queue.pop_back(); // a little of housekeeping along the way
                else
                    *curr = 0;      // grabbing task with the same isolation
                // TODO: move one of the container's ends instead if the task has been found there
                return result;
            }
        } while( curr != queue.begin() );
        return NULL;
    }

    //! Try finding and popping a related task.
    task* pop_specific( int level, __TBB_ISOLATION_ARG(unsigned& last_used_lane, isolation_tag isolation) ) {
        task* result = NULL;
        // Lane selection is round-robin in backward direction.
        unsigned idx = last_used_lane & (N-1);
        do {
            if( is_bit_set( population[level], idx ) ) {
                lane_t& lane = lanes[level][idx];
                spin_mutex::scoped_lock lock;
                if( lock.try_acquire(lane.my_mutex) && !lane.my_queue.empty() ) {
                    result = look_specific( __TBB_ISOLATION_ARG(lane.my_queue, isolation) );
                    if( lane.my_queue.empty() )
                        clear_one_bit( population[level], idx );
                    if( result )
                        break;
                }
            }
            idx=(idx-1)&(N-1);
        } while( !empty(level) && idx != last_used_lane );
        last_used_lane = idx;
        return result;
    }

    //! Checks existence of a task.
    bool empty(int level) {
        return !population[level];
    }

    //! Destroys all remaining tasks in every lane. Returns the number of destroyed tasks.
    /** Tasks are not executed, because it would potentially create more tasks at a late stage.
        The scheduler is really expected to execute all tasks before task_stream destruction. */
    intptr_t drain() {
        intptr_t result = 0;
        for(int level = 0; level < Levels; level++)
            for(unsigned i=0; i<N; ++i) {
                lane_t& lane = lanes[level][i];
                spin_mutex::scoped_lock lock(lane.my_mutex);
                for(typename lane_t::queue_base_t::iterator it=lane.my_queue.begin();
                    it!=lane.my_queue.end(); ++it, ++result)
                {
                    __TBB_ASSERT( is_bit_set( population[level], i ), NULL );
                    task* t = *it;
                    tbb::task::destroy(*t);
                }
                lane.my_queue.clear();
                clear_one_bit( population[level], i );
            }
        return result;
    }
}; // task_stream

} // namespace internal
} // namespace tbb

#endif /* _TBB_task_stream_extended_H */
