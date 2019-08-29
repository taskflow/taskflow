#include "universe.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/flow_graph.h>
#include <tbb/partitioner.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


struct UpdateStressBody {
  Universe & u_;
  UpdateStressBody(Universe & u):u_(u){}
  void operator()( const tbb::blocked_range<int>& range ) const {
    u_.UpdateStress(Universe::Rectangle(0, range.begin(), u_.UniverseWidth-1, range.size()));
  }
};

struct UpdateVelocityBody {
  Universe & u_;
  UpdateVelocityBody(Universe & u):u_(u){}
  void operator()( const tbb::blocked_range<int>& y_range ) const {
    u_.UpdateVelocity(Universe::Rectangle(1, y_range.begin(), u_.UniverseWidth-1, y_range.size()));
  }
};


// the wavefront computation
void seismic_tbb(unsigned num_threads, unsigned num_frames, Universe& u) {

  using namespace tbb;
  using namespace tbb::flow;
  
  tbb::task_scheduler_init init(num_threads);

  static tbb::affinity_partitioner affinity;

  for(unsigned i=0u; i<num_frames; ++i ) {
    u.UpdatePulse();
    tbb::parallel_for(tbb::blocked_range<int>( 0, u.UniverseHeight-1 ), // Index space for loop
                      UpdateStressBody(u),                            // Body of loop
                      affinity);                                      // Affinity hint
    tbb::parallel_for(tbb::blocked_range<int>( 1, u.UniverseHeight ), // Index space for loop
                      UpdateVelocityBody(u),                        // Body of loop
                      affinity);                                    // Affinity hint
  }
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads, unsigned num_frames, Universe& u) {
  auto beg = std::chrono::high_resolution_clock::now();
  seismic_tbb(num_threads, num_frames, u);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


