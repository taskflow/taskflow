// 2018/09/13 - created by Chun-Xun Lin
//
// This example is from https://www.geeksforgeeks.org/divide-and-conquer-maximum-sum-subarray/

#include <climits>

#include <taskflow/taskflow.hpp>  // the only include you need

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
void maxSubArraySum(std::vector<int>& vec, int l, int r, std::atomic<int>& max_num, T& subflow) 
{ 
  // Base Case: Only one element 
  if (l == r) {
    update_max(max_num, vec[l]);  
    return ;
  }

  // Find middle point 
  int m = (l + r)/2; 

  subflow.silent_emplace(
    [l=l, m=m, &vec, &max_num](auto& subflow){
      maxSubArraySum(vec, l, m, max_num, subflow);
    }
  );

  subflow.silent_emplace(
    [r=r, m=m, &vec, &max_num](auto& subflow){
      maxSubArraySum(vec, m+1, r, max_num, subflow);
    }
  );

  update_max(max_num, maxCrossingSum(vec, l, m, r));
} 


// Function: main
int main(int argc, char* argv[]) {

  ::srand(1);

  std::vector<int> vec(total, 0);
  auto gen = [](){return rand()%(max_num-min_num) + min_num;};
  std::generate(std::begin(vec), std::end(vec), gen);
  std::atomic<int> result {INT_MIN};

  auto start = std::chrono::high_resolution_clock::now();
  {
    tf::Taskflow tf(std::thread::hardware_concurrency());
    tf.silent_emplace(
      [&vec, &result](auto &subflow){
        maxSubArraySum(vec, 0, total-1, result, subflow);
        subflow.detach();
      }
    );
    tf.wait_for_all();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Taskflow elapsed: " << elapsed.count() << " ms\n";
  //std::cout << result << '\n';

  return 0;
}


