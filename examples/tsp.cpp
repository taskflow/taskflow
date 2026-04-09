#include <taskflow/taskflow.hpp>
 
const int INF = std::numeric_limits<int>::max();
 
// ── distance matrix (4 cities: A=0, B=1, C=2, D=3) ───────────────────────
const int N = 4;
const int dist[N][N] = {
  {  0, 10, 15, 20 },
  { 10,  0, 35, 25 },
  { 15, 35,  0, 30 },
  { 20, 25, 30,  0 },
};
 
// ── lower bound ──────────────────────────────────────────────────────────────
//
// The key insight: any valid completion of the partial tour must include
// at least one edge leaving the current city (to reach the next unvisited
// city) and at least one edge touching each remaining unvisited city
// (to enter or leave it).
//
// We do not know which edges those will be, but we know they cannot be
// shorter than the cheapest available option for each city.
// So the cheapest possible completion is at minimum:
//
//   (1) the shortest edge from the current city to any unvisited city
//   (2) for each unvisited city, its shortest outgoing edge to any other city
//
// Adding these minimum costs to the cost already spent gives a floor: the
// final tour cannot possibly cost less than this bound, regardless of how
// the remaining cities are visited.
// If this floor already meets or exceeds the best complete tour found so
// far, the entire subtree can be pruned safely.
//
// Example: at partial tour A->C (cost=15) with best_cost=80:
//   (1) min edge from C to unvisited {B,D}: min(35,30) = 30
//   (2) min edge from B to anywhere:        min(10,35,25) = 10
//       min edge from D to anywhere:        min(20,25,30) = 20
//   lb = 15 + 30 + 10 + 20 = 75  <  80  =>  not pruned yet
//
// Later at A->C->B (cost=50):
//   (1) min edge from B to unvisited {D}: 25
//   (2) min edge from D to anywhere:      20
//   lb = 50 + 25 + 20 = 95  >=  80  =>  PRUNED
//
int lower_bound(int current, const std::vector<bool>& visited, int cost_so_far) {
  int bound = cost_so_far;
 
  // (1) cheapest edge we can take out of the current city
  // to reach any city we have not yet visited
  int min_curr = INF;
  for(int v = 0; v < N; v++) {
    if(!visited[v]) min_curr = std::min(min_curr, dist[current][v]);
  }
  if(min_curr == INF) return bound;   // no unvisited cities: tour is complete
  bound += min_curr;
 
  // (2) for each unvisited city, add the cost of its cheapest outgoing edge
  // to any other city (visited or not, as long as it is not the city itself).
  // This accounts for the fact that we must enter or leave each unvisited city
  // at some point, and the cheapest way to do so is via its minimum edge.
  for(int u = 0; u < N; u++) {
    if(visited[u]) continue;
    int min_u = INF;
    for(int v = 0; v < N; v++) {
      if(v != u) min_u = std::min(min_u, dist[u][v]);
    }
    bound += min_u;
  }
  return bound;
}
 
tf::Executor executor;
std::atomic<int> best_cost{INF};
 
void branch_and_bound(
  int current,
  std::vector<bool> visited,
  int cost_so_far,
  int depth
) {
  // compute a lower bound on the best possible tour through this node.
  // if the bound already meets or exceeds the best complete tour found so
  // far by any worker, this entire subtree cannot improve on the best known
  // solution and can be safely discarded.
  int lb = lower_bound(current, visited, cost_so_far);
  if(lb >= best_cost.load(std::memory_order_relaxed)) return;
 
  // all N cities have been visited: close the tour by returning to city 0.
  // try to update best_cost if this tour is cheaper than any found so far.
  // compare_exchange_weak retries if another worker concurrently updates
  // best_cost, ensuring only the true minimum is stored.
  if(depth == N) {
    int total = cost_so_far + dist[current][0];
    int cur = best_cost.load(std::memory_order_relaxed);
    while(total < cur &&
          !best_cost.compare_exchange_weak(cur, total,
            std::memory_order_relaxed, std::memory_order_relaxed));
    return;
  }
 
  // create a task group for parallel branching.
  // task_group() is valid here because this function always runs inside a
  // worker thread of the executor (either the root executor.async task or
  // a tg.silent_async task spawned by a parent call).
  auto tg = executor.task_group();
 
  bool first = true;
  for(int v = 0; v < N; v++) {
    if(visited[v]) continue;
 
    // prepare the state for the branch where we visit city v next
    std::vector<bool> new_visited = visited;
    new_visited[v] = true;
    int new_cost = cost_so_far + dist[current][v];
 
    if(first) {
      // run the first unvisited branch directly on the current worker.
      // this avoids task creation overhead for at least one branch per node
      // and keeps the current worker busy without yielding to the scheduler.
      first = false;
      branch_and_bound(v, new_visited, new_cost, depth + 1);
    }
    else {
      // spawn the remaining branches as independent async tasks in the group.
      // each task explores a different city as the next stop in the tour.
      // all spawned tasks share the same best_cost atomic, so a good solution
      // found in any branch immediately tightens the pruning bound for all others.
      tg.silent_async([=]() {
        branch_and_bound(v, new_visited, new_cost, depth + 1);
      });
    }
  }
 
  // cooperatively wait for all spawned branches to finish.
  // corun() does not block the worker thread: it re-enters the executor's
  // work-stealing loop, executing other pending tasks while waiting.
  // this prevents worker starvation and avoids deadlock in deep recursion.
  tg.corun();
}
 
int main() {
 
  std::vector<bool> visited(N, false);
  visited[0] = true;   // start from city A (index 0)
 
  // the root call must run inside a worker so that task_group() is valid
  executor.async([&]() {
    branch_and_bound(0, visited, 0, 1);
  }).get();
 
  const char* names[] = {"A", "B", "C", "D"};
  printf("Optimal tour cost: %d\n", best_cost.load());
 
  return 0;
}
