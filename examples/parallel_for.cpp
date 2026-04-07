// This program demonstrates loop-based parallelism in Taskflow using:
//
//   1. STL-iterator-based parallelism    (tf::Taskflow::for_each)
//   2. Scalar index-based parallelism    (tf::Taskflow::for_each_index)
//   3. 1D range-based parallelism        (tf::Taskflow::for_each_by_index, IndexRange<T>)
//   4. Multi-dimensional parallelism     (tf::Taskflow::for_each_by_index, IndexRange<T,N>)
//
// --------------------------------------------------------------------------
// Background: IndexRange<T, N>
// --------------------------------------------------------------------------
//
// tf::IndexRange<T, N> represents the Cartesian product of N independent 1D
// ranges, each defined by a (begin, end, step_size) triple. Iteration order
// is row-major (last dimension varies fastest), mirroring nested C-style loops:
//
//   for i in dim[0]:          // outermost / slowest
//     for j in dim[1]:
//       ...
//         for k in dim[N-1]:  // innermost / fastest
//
// Flat index 0 corresponds to (beg[0], beg[1], ..., beg[N-1]).
//
// When N == 1 (the default), IndexRange<T> is the familiar 1D range.
// When N > 1  (IndexRangeMDLike), the range represents a hyper-rectangular
// region of the iteration space.
//
// --------------------------------------------------------------------------
// How parallelism works for multi-dimensional ranges
// --------------------------------------------------------------------------
//
// Taskflow partitions the flat iteration space [0, N_total) among workers.
// The partitioner calls range.consume_chunk(flat_beg, chunk_size), which maps
// a contiguous flat slice back into the largest valid orthogonal sub-box
// (the "trailing zeros" rule ensures the sub-box is always a perfect
// hyper-rectangle).  Each worker's callback receives one such sub-box and is
// responsible for iterating it however it likes.
//
// This design means:
//   - No false sharing from stride misalignment.
//   - The callback controls whether to iterate, vectorise, or recurse.
//   - Step sizes, negative strides, and non-zero origins are all handled.

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

// ============================================================================
// 1. STL iterator parallelism
// ============================================================================
//
// for_each dispatches one callable per element of a container.
// The StaticPartitioner divides the range into W equal chunks (one per worker)
// before execution begins — ideal when each element costs roughly the same.

void demo_for_each(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> data(N);
  std::iota(data.begin(), data.end(), 0);

  taskflow.for_each(
    data.begin(), data.end(),
    [](int v) {
      printf("[for_each] element = %d\n", v);
    },
    tf::StaticPartitioner()
  );

  executor.run(taskflow).get();
}

// ============================================================================
// 2. Scalar index parallelism
// ============================================================================
//
// for_each_index is the scalar equivalent of for_each: the callback receives
// the raw index value (not an iterator or sub-range).  Useful when the index
// itself is what you need (e.g., addressing a flat array by position).
//
// Supports non-unit and negative step sizes:
//   for_each_index(beg, end, step, callback)

void demo_for_each_index(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // Forward traversal: indices 0, 2, 4, ..., < N
  taskflow.for_each_index(
    0, N, 2,
    [](int i) {
      printf("[for_each_index forward] i = %d\n", i);
    }
  );
  executor.run(taskflow).wait();

  // Reverse traversal: indices N-1, N-3, ..., > 0
  // (ensure N > 0 and beg > end for a valid negative-step range)
  if (N > 0) {
    taskflow.clear();
    taskflow.for_each_index(
      N - 1, -1, -2,
      [](int i) {
        printf("[for_each_index reverse] i = %d\n", i);
      }
    );
    executor.run(taskflow).wait();
  }
}

// ============================================================================
// 3. 1D range-based parallelism with IndexRange<T>
// ============================================================================
//
// for_each_by_index with an IndexRange<T> (N == 1) delivers sub-ranges rather
// than scalar indices.  Each worker receives an IndexRange<T> covering its
// assigned slice, so it can iterate with any loop structure it chooses.
//
// This is particularly useful when the body needs to know the stride, or when
// SIMD/vectorisation hints must be applied over a contiguous sub-range.

void demo_for_each_by_index_1d(int N) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // Forward 1D range: [0, N) step 2
  {
    tf::IndexRange<int> range(0, N, 2);

    taskflow.for_each_by_index(
      range,
      [](tf::IndexRange<int> sub) {
        // sub covers a contiguous slice of the original range
        for (int i = sub.begin(); i < sub.end(); i += sub.step_size()) {
          printf("[for_each_by_index 1D forward] i = %d\n", i);
        }
      },
      tf::GuidedPartitioner()  // adaptive chunk sizes — good for uneven work
    );

    executor.run(taskflow).wait();
  }

  // Negative-step 1D range: [N-1, -1) step -1  (i.e., N-1 down to 0)
  if (N > 0) {
    taskflow.clear();
    tf::IndexRange<int> range(N - 1, -1, -1);

    taskflow.for_each_by_index(
      range,
      [](tf::IndexRange<int> sub) {
        for (int i = sub.begin(); i > sub.end(); i += sub.step_size()) {
          printf("[for_each_by_index 1D reverse] i = %d\n", i);
        }
      }
    );

    executor.run(taskflow).wait();
  }
}

// ============================================================================
// 4. Multi-dimensional parallelism with IndexRange<T, N>
// ============================================================================
//
// for_each_by_index with an IndexRangeMDLike argument parallelises over the
// flat Cartesian product of N 1D ranges.  The callback receives one orthogonal
// sub-box (itself an IndexRange<T, N>) per call and is responsible for
// iterating its contents.
//
// The sub-box preserves the original step sizes for every dimension, so
// negative strides and non-zero origins are handled transparently.
//
// Example: a 2D grid [0,H) x [0,W) partitioned across workers:
//
//   Worker 0 might receive sub-box [0,2) x [0,W)   (rows 0-1, all columns)
//   Worker 1 might receive sub-box [2,4) x [0,W)   (rows 2-3, all columns)
//   ...
//
// The exact box boundaries depend on the partitioner and the geometry.

// --------------------------------------------------------------------------
// 4a. 2D example — image-like grid, unit steps
// --------------------------------------------------------------------------
//
// Typical use case: parallel traversal of a 2D array of size H x W.
// Each worker receives a rectangular tile; it iterates row-by-row over the
// tile and accesses data[row * W + col].

void demo_2d(int H, int W) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> data(H * W, 0);

  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(0, H, 1),   // rows
    tf::IndexRange<int>(0, W, 1)    // columns
  );

  printf("\n[2D] grid %d x %d  (total %zu elements)\n", H, W, range.size());

  std::atomic<int> tile_count{0};

  taskflow.for_each_by_index(
    range,
    [&](tf::IndexRange<int, 2> tile) {
      // Each invocation owns one rectangular tile of the grid.
      // dim(0) is the row sub-range; dim(1) is the column sub-range.
      int t = tile_count.fetch_add(1, std::memory_order_relaxed);
      printf("  tile %d: rows [%d,%d) cols [%d,%d)  (%zu elements)\n",
             t,
             tile.dim(0).begin(), tile.dim(0).end(),
             tile.dim(1).begin(), tile.dim(1).end(),
             tile.size());

      for (int r = tile.dim(0).begin(); r < tile.dim(0).end(); r += tile.dim(0).step_size()) {
        for (int c = tile.dim(1).begin(); c < tile.dim(1).end(); c += tile.dim(1).step_size()) {
          data[r * W + c] = r * W + c;
        }
      }
    },
    tf::DynamicPartitioner()
  );

  executor.run(taskflow).wait();

  // Verify: every element was written exactly once
  bool ok = true;
  for (int i = 0; i < H * W; i++) {
    if (data[i] != i) { ok = false; break; }
  }
  printf("  result: %s\n", ok ? "PASS" : "FAIL");
}

// --------------------------------------------------------------------------
// 4b. 3D example — volumetric grid, unit steps
// --------------------------------------------------------------------------
//
// Useful for finite-difference stencils, voxel grids, or tensor operations.
// The partitioner carves the 3D space into axis-aligned sub-volumes.

void demo_3d(int D, int H, int W) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<int> data(D * H * W, 0);

  tf::IndexRange<int, 3> range(
    tf::IndexRange<int>(0, D, 1),   // depth
    tf::IndexRange<int>(0, H, 1),   // height
    tf::IndexRange<int>(0, W, 1)    // width
  );

  printf("\n[3D] volume %d x %d x %d  (total %zu elements)\n", D, H, W, range.size());

  std::atomic<int> subvol_count{0};

  taskflow.for_each_by_index(
    range,
    [&](tf::IndexRange<int, 3> vol) {
      int v = subvol_count.fetch_add(1, std::memory_order_relaxed);
      printf("  sub-volume %d: depth [%d,%d) height [%d,%d) width [%d,%d)  (%zu elements)\n",
             v,
             vol.dim(0).begin(), vol.dim(0).end(),
             vol.dim(1).begin(), vol.dim(1).end(),
             vol.dim(2).begin(), vol.dim(2).end(),
             vol.size());

      for (int d = vol.dim(0).begin(); d < vol.dim(0).end(); d += vol.dim(0).step_size()) {
        for (int h = vol.dim(1).begin(); h < vol.dim(1).end(); h += vol.dim(1).step_size()) {
          for (int w = vol.dim(2).begin(); w < vol.dim(2).end(); w += vol.dim(2).step_size()) {
            data[d * H * W + h * W + w] = d * H * W + h * W + w;
          }
        }
      }
    },
    tf::GuidedPartitioner()
  );

  executor.run(taskflow).wait();

  bool ok = true;
  for (int i = 0; i < D * H * W; i++) {
    if (data[i] != i) { ok = false; break; }
  }
  printf("  result: %s\n", ok ? "PASS" : "FAIL");
}

// --------------------------------------------------------------------------
// 4c. 2D example — non-unit step sizes
// --------------------------------------------------------------------------
//
// Step sizes larger than 1 are useful for strided access patterns, e.g.:
//   - Processing every other row/column
//   - Red-black Gauss-Seidel sweeps
//   - Downsampling or decimation kernels
//
// The sub-box preserves the original step sizes, so the inner loops remain
// identical to what you would write for sequential code.

void demo_2d_strided(int H, int W) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // Only even rows (step 2) and every 3rd column (step 3)
  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(0, H, 2),   // rows:    0, 2, 4, ...
    tf::IndexRange<int>(0, W, 3)    // columns: 0, 3, 6, ...
  );

  printf("\n[2D strided] step=(2,3)  logical elements: %zu\n", range.size());

  std::atomic<size_t> total{0};

  taskflow.for_each_by_index(
    range,
    [&](tf::IndexRange<int, 2> tile) {
      size_t local = 0;
      for (int r = tile.dim(0).begin(); r < tile.dim(0).end(); r += tile.dim(0).step_size()) {
        for (int c = tile.dim(1).begin(); c < tile.dim(1).end(); c += tile.dim(1).step_size()) {
          (void)r; (void)c;
          local++;
        }
      }
      total.fetch_add(local, std::memory_order_relaxed);
    }
  );

  executor.run(taskflow).wait();

  printf("  elements visited: %zu  expected: %zu  %s\n",
         total.load(), range.size(),
         total.load() == range.size() ? "PASS" : "FAIL");
}

// --------------------------------------------------------------------------
// 4d. 2D example — negative step sizes
// --------------------------------------------------------------------------
//
// Negative step sizes traverse a dimension in reverse.  This is useful for:
//   - Back-substitution in triangular solvers
//   - Reverse wavefront traversals
//   - Algorithms that must process rows/columns in descending order
//
// Here both dimensions run in reverse: rows from H-1 down to 0,
// columns from W-1 down to 0.  The inner loop comparison must use '>'
// instead of '<' when the step is negative.

void demo_2d_negative_steps(int H, int W) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // Reverse both dimensions
  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(H - 1, -1, -1),  // rows:    H-1, H-2, ..., 0
    tf::IndexRange<int>(W - 1, -1, -1)   // columns: W-1, W-2, ..., 0
  );

  printf("\n[2D negative steps] %d x %d reversed  (total %zu elements)\n",
         H, W, range.size());

  std::atomic<size_t> total{0};

  taskflow.for_each_by_index(
    range,
    [&](tf::IndexRange<int, 2> tile) {
      size_t local = 0;
      // step_size() is negative, so the loop condition flips to '>'
      for (int r = tile.dim(0).begin(); r > tile.dim(0).end(); r += tile.dim(0).step_size()) {
        for (int c = tile.dim(1).begin(); c > tile.dim(1).end(); c += tile.dim(1).step_size()) {
          (void)r; (void)c;
          local++;
        }
      }
      total.fetch_add(local, std::memory_order_relaxed);
    }
  );

  executor.run(taskflow).wait();

  printf("  elements visited: %zu  expected: %zu  %s\n",
         total.load(), range.size(),
         total.load() == range.size() ? "PASS" : "FAIL");
}

// --------------------------------------------------------------------------
// 4e. Stateful 2D range — range values set at runtime
// --------------------------------------------------------------------------
//
// When the range bounds are not known at task-graph construction time, pass
// the range by std::ref.  An upstream init task fills the range before the
// parallel loop runs.  Taskflow's dependency graph guarantees the init task
// completes before any worker reads the range.

void demo_2d_stateful(int H, int W) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  // Placeholder range — will be filled by the init task
  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 0, 1)
  );

  std::vector<int> data;
  std::atomic<size_t> total{0};

  auto init = taskflow.emplace([&]() {
    // Bounds are only known here, e.g. after reading a file or receiving input
    range.dim(0).reset(0, H, 1);
    range.dim(1).reset(0, W, 1);
    data.assign(H * W, -1);
    printf("\n[2D stateful] range set to %d x %d\n", H, W);
  });

  auto loop = taskflow.for_each_by_index(
    std::ref(range),                    // capture by reference: reads the range at run time
    [&](tf::IndexRange<int, 2> tile) {
      for (int r = tile.dim(0).begin(); r < tile.dim(0).end(); r += tile.dim(0).step_size()) {
        for (int c = tile.dim(1).begin(); c < tile.dim(1).end(); c += tile.dim(1).step_size()) {
          data[r * W + c] = r * W + c;
          total.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  );

  init.precede(loop);   // init must finish before loop starts

  executor.run(taskflow).wait();

  bool ok = (total.load() == static_cast<size_t>(H * W));
  for (int i = 0; ok && i < H * W; i++) ok = (data[i] == i);
  printf("  result: %s  (%zu elements written)\n", ok ? "PASS" : "FAIL", total.load());
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: ./parallel_for N\n"
              << "  N  — controls iteration counts; must be > 0\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);
  if (N <= 0) {
    std::cerr << "N must be a positive integer\n";
    std::exit(EXIT_FAILURE);
  }

  printf("=== 1. STL iterator parallelism ===\n");
  demo_for_each(N);

  printf("\n=== 2. Scalar index parallelism ===\n");
  demo_for_each_index(N);

  printf("\n=== 3. 1D range-based parallelism ===\n");
  demo_for_each_by_index_1d(N);

  printf("\n=== 4a. 2D grid (unit steps) ===\n");
  demo_2d(N, N + 3);          // deliberately non-square to stress partitioner

  printf("\n=== 4b. 3D volume (unit steps) ===\n");
  demo_3d(N, N + 1, N + 2);   // three distinct dimensions

  printf("\n=== 4c. 2D strided (step 2 x step 3) ===\n");
  demo_2d_strided(N * 2, N * 3);

  printf("\n=== 4d. 2D negative steps (reverse traversal) ===\n");
  demo_2d_negative_steps(N, N + 1);

  printf("\n=== 4e. 2D stateful range ===\n");
  demo_2d_stateful(N, N + 2);

  return 0;
}
