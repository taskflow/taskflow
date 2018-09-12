#include <atomic>
#include <chrono>
#include <random>
#include <map>
#include <algorithm>
#include <climits>

#include <taskflow/threadpool/threadpool.hpp>

constexpr auto min_num = -50;
constexpr auto max_num = 50;
constexpr auto tree_height = 20u;
constexpr auto total = 1u << tree_height;

void update_max(std::atomic<int>& max_val, int const& value)
{
  int old = max_val;
  while(old < value && !max_val.compare_exchange_weak(old, value));
}

int maxCrossingSum(std::vector<int>& vec, int l, int m, int r){  
  // Include elements on left of mid. 
  auto sum = 0; 
  auto left_sum = INT_MIN; 
  for (auto i = m; i >= l; i--){ 
    sum = sum + vec[i]; 
    if (sum > left_sum) 
      left_sum = sum; 
  } 

  // Include elements on right of mid 
  sum = 0; 
  auto right_sum = INT_MIN; 
  for (auto i = m+1; i <= r; i++) 
  { 
    sum = sum + vec[i]; 
    if (sum > right_sum) 
      right_sum = sum; 
  } 

  // Return sum of elements on left and right of mid 
  return left_sum + right_sum; 
} 

template<typename T>
void maxSubArraySum(std::vector<int>& vec, int l, int r, std::atomic<int>& max_num, T& tp, 
  std::atomic<size_t>& counter, std::promise<void>& promise) 
{ 
  // Base Case: Only one element 
  if (l == r) {
    update_max(max_num, vec[l]);  
    if(++counter == total*2-1){
      promise.set_value();
    }
    return ;
  }

  // Find middle point 
  int m = (l + r)/2; 

  tp.silent_async([&, l=l, m=m](){ maxSubArraySum(vec, l,   m, max_num, tp, counter, promise); });
  tp.silent_async([&, m=m, r=r](){ maxSubArraySum(vec, m+1, r, max_num, tp, counter, promise); });

  update_max(max_num, maxCrossingSum(vec, l, m, r));
  if(++counter == total*2-1){
    promise.set_value();
  }
} 

template<typename T>
auto subsum(std::vector<int>& vec){
  std::atomic<int> result {INT_MIN};
  std::atomic<size_t> counter{0};
  std::promise<void> promise;
  auto future = promise.get_future();

  auto start = std::chrono::high_resolution_clock::now();
  {
    T tp(std::thread::hardware_concurrency());
    maxSubArraySum(vec, 0, total-1, result, tp, counter, promise);
    future.get();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  //std::cout << counter << '\n';
  return elapsed.count();
}

// https://www.geeksforgeeks.org/divide-and-conquer-maximum-sum-subarray/
void benchmark_subsum(){
  std::vector<int> vec(total, 0);
  auto gen = [](){return rand()%(max_num-min_num) + min_num;};
  std::generate(std::begin(vec), std::end(vec), gen);

  std::cout << "Proactive threadpool takes: " 
            << subsum<tf::ProactiveThreadpool>(vec) << " ms\n";

  std::cout << "Simple threadpool takes: " 
            << subsum<tf::SimpleThreadpool>(vec) << " ms\n";
}


// Function: main
int main(int argc, char* argv[]) {
  ::srand(1);
  benchmark_subsum();
  return 0;
}


