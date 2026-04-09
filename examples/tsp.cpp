// tsp.cpp: parallel branch-and-bound solver for the Travelling Salesman Problem
// using tf::TaskGroup for recursive task parallelism.
//
// The example uses a hand-constructed 4-city instance:
//
//       A    B    C    D
//   A [  0   10   15   20 ]
//   B [ 10    0   35   25 ]
//   C [ 15   35    0   30 ]
//   D [ 20   25   30    0 ]
//
// Optimal tour: A -> B -> D -> C -> A = 80

#include <taskflow/taskflow.hpp>

// ── problem constants ────────────────────────────────────────────────────────
// Use NUM_CITIES instead of N to avoid shadowing template parameter 'N'
// inside Taskflow's internal headers (e.g. SmallVector, UnboundedWSQ).

static constexpr int NUM_CITIES = 4;
static constexpr int INF        = std::numeric_limits<int>::max();

static const int DIST[NUM_CITIES][NUM_CITIES] = {
  {  0, 10, 15, 20 },   // A
  { 10,  0, 35, 25 },   // B
  { 15, 35,  0, 30 },   // C
  { 20, 25, 30,  0 },   // D
};

// ── shared state ─────────────────────────────────────────────────────────────

static tf::Executor        executor;
static std::atomic<int>    best_cost{ INF };

// ── lower bound ──────────────────────────────────────────────────────────────
//
// Any valid completion of the partial tour must include:
//   (1) at least one edge leaving the current city to some unvisited city
//   (2) at least one edge touching each remaining unvisited city
//
// We use the cheapest available option for each of these, giving a floor on
// the best possible tour cost from this node onward.
// If this floor already meets or exceeds best_cost, the subtree is pruned.
//
// Example: at A->C (cost=15, best_cost=80 after first leaf):
//   (1) min edge from C to unvisited {B,D}: min(35,30) = 30
//   (2) min edge from B to anywhere:        min(10,35,25) = 10
//       min edge from D to anywhere:        min(20,25,30) = 20
//   lb = 15 + 30 + 10 + 20 = 75  <  80  => not pruned
//
// At A->C->B (cost=50):
//   (1) min edge from B to unvisited {D}: 25
//   (2) min edge from D to anywhere:      20
//   lb = 50 + 25 + 20 = 95  >= 80  => PRUNED
//
static int lower_bound(
  int                       current,
  const std::vector<bool>&  visited,
  int                       cost_so_far
) {
  int bound = cost_so_far;

  // (1) cheapest edge out of the current city to any unvisited city
  int min_curr = INF;
  for(int v = 0; v < NUM_CITIES; v++) {
    if(!visited[v]) {
      min_curr = std::min(min_curr, DIST[current][v]);
    }
  }
  if(min_curr == INF) return bound;   // no unvisited cities: tour is complete
  bound += min_curr;

  // (2) cheapest outgoing edge from each unvisited city to any other city
  for(int u = 0; u < NUM_CITIES; u++) {
    if(visited[u]) continue;
    int min_u = INF;
    for(int v = 0; v < NUM_CITIES; v++) {
      if(v != u) {
        min_u = std::min(min_u, DIST[u][v]);
      }
    }
    bound += min_u;
  }
  return bound;
}

// ── branch and bound ─────────────────────────────────────────────────────────
//
// Each call represents one node in the TSP search tree.
// It creates a tf::TaskGroup and spawns one async task per unvisited city
// (except the first, which runs on the current worker to avoid task overhead).
// tg.corun() cooperatively waits for all spawned branches without blocking
// the worker thread.
//
static void branch_and_bound(
  int                 current,     // city we are currently at
  std::vector<bool>   visited,     // which cities have been visited
  int                 cost_so_far, // accumulated tour cost so far
  int                 depth        // number of cities visited so far
) {
  // compute lower bound and prune if it cannot improve on best_cost
  int lb = lower_bound(current, visited, cost_so_far);
  if(lb >= best_cost.load(std::memory_order_relaxed)) return;

  // all cities visited: close the tour by returning to city 0
  if(depth == NUM_CITIES) {
    int total = cost_so_far + DIST[current][0];
    // update best_cost if this tour is cheaper; compare_exchange_weak retries
    // on spurious failure so the true minimum is always stored correctly
    int cur = best_cost.load(std::memory_order_relaxed);
    while(total < cur &&
          !best_cost.compare_exchange_weak(
            cur, total,
            std::memory_order_relaxed,
            std::memory_order_relaxed));
    return;
  }

  // create a task group for parallel branching.
  // task_group() is valid here because every call to branch_and_bound runs
  // inside a worker thread: either the root executor.async() task or a
  // tg.silent_async() task spawned by a parent call.
  auto tg = executor.task_group();

  bool first = true;
  for(int v = 0; v < NUM_CITIES; v++) {
    if(visited[v]) continue;

    // prepare state for the branch where city v is visited next
    std::vector<bool> new_visited = visited;
    new_visited[v] = true;
    int new_cost = cost_so_far + DIST[current][v];

    if(first) {
      // run the first unvisited branch directly on this worker to avoid
      // unnecessary task creation overhead
      first = false;
      branch_and_bound(v, new_visited, new_cost, depth + 1);
    }
    else {
      // spawn remaining branches as independent async tasks in the group.
      // all spawned tasks share best_cost, so a better solution found in any
      // branch immediately tightens the pruning bound for all other branches.
      tg.silent_async([=]() {
        branch_and_bound(v, new_visited, new_cost, depth + 1);
      });
    }
  }

  // cooperatively wait for all spawned branches to complete.
  // corun() re-enters the executor's work-stealing loop rather than blocking,
  // preventing worker starvation in deep recursion.
  tg.corun();
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {

  std::vector<bool> visited(NUM_CITIES, false);
  visited[0] = true;   // start from city A (index 0)

  // the root call must be submitted as an async task so it runs inside a
  // worker thread, which is required for executor.task_group() to be valid
  executor.async([&]() {
    branch_and_bound(0, visited, 0, 1);
  }).get();

  const char* names[] = { "A", "B", "C", "D" };
  printf("Optimal tour cost: %d\n", best_cost.load());
  (void)names;

  return 0;
}
