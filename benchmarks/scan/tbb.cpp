#include "scan.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

template<typename T, typename Op>
T inclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op) {
	using range_type = tbb::blocked_range<size_t>;
	T sum = tbb::parallel_scan(range_type(0, in.size()), id,
		[&](const range_type &r, T sum, bool is_final_scan) {
			T tmp = sum;
			for (size_t i = r.begin(); i < r.end(); ++i) {
				tmp = op(tmp, in[i]);
				if (is_final_scan) {
					out[i] = tmp;
				}
			}
			return tmp;
		},
		[&](const T &a, const T &b) {
			return op(a, b);
		});
	return sum;
}

template<typename T, typename Op>
T exclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op) {
	// Exclusive scan is the same as inclusive, but shifted by one
	using range_type = tbb::blocked_range<size_t>;
	T sum = tbb::parallel_scan(range_type(0, in.size()), id,
		[&](const range_type &r, T sum, bool is_final_scan) {
			T tmp = sum;
			for (size_t i = r.begin(); i < r.end(); ++i) {
				tmp = op(tmp, in[i]);
				if (is_final_scan) {
					out[i + 1] = tmp;
				}
			}
			return tmp;
		},
		[&](const T &a, const T &b) {
			return op(a, b);
		});
	out.pop_back();
	return sum;
}

// scan_tbb
void scan_tbb(size_t num_threads) {
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );
  inclusive_scan(input, 0, output, std::multiplies<int>{});
}

std::chrono::microseconds measure_time_tbb(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  scan_tbb(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
