# Reduce a Container of Items in Parallel

Parallel tasks normally produce some quantity that needs to be combined or *reduced*
through particular operations, for instance, sum.
In this example, we are going to demonstrate how to use Cpp-Taskflow
to parallelize a reduction workload.

+ [Reduce](#Reduce-through-an-Operator)
+ [Transform and Reduce](#Transform-and-Reduce)
+ [Example 1: Find the Min/Max Element](#Example-1-Find-the-Min-Max-Element)
+ [Example 2: Pipeline a Reducer Graph](#Example-2-Pipeline-a-Reducer-Graph)
+ [Example 3: Find the Minimum L1-norm](#Example-3-Find-the-Minimum-L1-norm)

# Reduce

The most basic reduction method, `reduce`, 
is to reduce a range of items through a binary operator.
The method applies the binary operator to iteratively combine two items 
and returns the final single result.

```cpp
 1: tf::Taskflow tf(4);
 2:
 3: std::vector<int> items {1, 2, 3, 4, 5, 6, 7, 8};
 4: int sum {0};
 5:
 6: auto [S, T] = tf.reduce(items.begin(), items.end(), sum, [] (int a, int b) {
 7:   return a + b;
 8: });
 9:
10: S.name("S");
11: T.name("T");
12:
13: tf.wait_for_all();
14:
15: std::cout << "sum = " << sum << std::endl;    // 36
```

Debrief:

+ Line 1 creates a taskflow object with 4 worker threads
+ Line 3 creates a vector of integers
+ Line 4 declares an integer variable `sum` and initializes it to zero
+ Line 6-8 constructs a reduction graph that sum up all integer itmes and stores the final 
  result in `sum`
+ Line 10-11 names the two synchronization points
+ Line 13 dispatches the graph for execution
+ Line 15 prints out the final reduction value

The task dependency graph of this example is shown below:

![](reduce1.png)

Taskflow partitions and distributes the workload evenly across all workers
for all reduction methods.
In this example, each internal node applies reduction to two items and 
the target node `T` will reduce all results returned by the internal nodes to a single value.

# Transform and Reduce

It is common to transform each item into a new data type and
then perform reduction on the transformed sequences.
Taskflow provides a method, `transform_reduce`, 
that fuses these two operators together to save memory reads and writes.
The example below takes a string and transforms each digit to an integer number,
and then applies reduction to sum up all integer numbers.

```cpp
 1: std::string str = "12345678";
 2: int sum {0};
 3:
 4: auto [S, T] = tf.transform_reduce(str.begin(), str.end(), sum,
 5:   [] (int a, int b) {
 6:     return a + b;
 7:   },  
 8:   [] (char c) -> int {
 9:     return c - '0';
10:   }   
11: ); 
12:
13: // sum will be 36 after execution
```

Debrief:

+ Line 1 creates a string of eight digits
+ Line 2 declares an integer variables and initializes it to zero
+ Line 4-11 constructs a reduction graph that converts each character of the string 
  into an integer and computes the sum of all integers

The method `transform_reduce` has another overload that takes one additional binary operator
to combine one raw item and the transformed one into the reduction type.
This is useful when extra computation is required during the reduction process.

```cpp
 1: std::string str = "12345678";
 2:
 3: double sum {0};
 4:
 5: auto [S, T] = tf.transform_reduce(str.begin(), str.end(), sum,
 6:   [] (double a, double b) {
 7:     return a + b;
 8:   },  
 9:   [] (double a, char c) -> double {
10:     return a + (c - '0');
11:   },  
12:   [] (char c) -> double {
13:     return static_cast<double>(c - '0');
14:   }   
15: );  
16:
17: // sum will be 36 after execution
```

Debrief:

+ Line 1 creates a string of eight digits
+ Line 3 declares an integer variable and initializes it to zero
+ Line 5 constructs a reduction graph that represents the reduction workload
+ Line 6-8 takes a binary operator to combine two transformed data
+ Line 9-11 takes a binary operator to combine one raw data together with a transformed data
+ Line 12-14 takes a unary operator to transform one raw data to the reduced data type 

The difference between the two overloads appears in the second binary operator.
Instead of converting every item to the reduced data type,
this binary operator provides a more fine-grained control over reduction.

---

# Example 1: Find the Min/Max Element

One common workload of using reduce is to find the minimum or the maximum
element in a range of items.
This example demonstrates how to use the method `reduce` to find the 
minimum element out of a set of items.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items {4, 2, 1, 3, 7, 8, 6, 5};
 8:   int min = std::numeric_limits<int>::max();
 9:
10:   tf.reduce(items.begin(), items.end(), min, [] (int a, int b) {
11:     return std::min(a, b);
12:   });
13:
14:   tf.wait_for_all();
15:
16:   std::cout << min << std::endl;  // 1
17:
18:   return 0;
19: }
```

Similarly, the example below uses the method `reduce` to find
the maximum element in an item set.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items {4, 2, 1, 3, 7, 8, 6, 5};
 8:   int max = std::numeric_limits<int>::min();
 9:
10:   tf.reduce(items.begin(), items.end(), max, [] (int a, int b) {
11:     return std::max(a, b);
12:   });
13:
14:   tf.wait_for_all();
15:
16:   std::cout << max << std::endl;  // 8
17:
18:   return 0;
19: }
```

# Example 2: Pipeline a Reducer Graph

The `reduce` method returns a task pair as two synchronization points
which can be used to pipeline with other tasks.
The example below demonstrates how to pipeline a reducer with two tasks.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items{1024};
 8:   int min = std::numeric_limits<int>::max();
 9:
10:   auto T1 = tf.silent_emplace([&] () {  // modifier task
11:     for(auto& item : items) {
12:       item = ::rand();
13:     }
14:   });
15:
16:   auto [S, T] = tf.reduce(items.begin(), items.end(), min, [] (int a, int b) {
17:     return std::min(a, b);
18:   });
19:
20:   auto T2 = tf.silent_emplace([&] () {  // printer task
21:     std::cout << "min is " << min << std::endl;
22:   });
23:
24:   T1.precede(S);    // modifier task precedes the reducer
25:   T.precede(T2);    // reducer precedes the printer task
26:
27:   tf.wait_for_all();
28:
29:   return 0;
30: }
```

Debrief:
+ Line 5 creates a taskflow object with four worker threads
+ Line 7 creates a vector of 1024 uninitialized integers
+ Line 8 creates an integer value initialized to the maximum value of its range
+ Line 10-14 creates a modifier task that initializes the vector to random integer values
+ Line 16-18 creates a reducer graph to find the minimum element in the vector
+ Line 20-22 creates a task that prints the minimum value found after reducer finishes
+ Line 24-25 adds two dependency links to implement our control flow
+ Line 27 dispatches the dependency graph into threads and waits until the execution completes

Each worker thread will apply the give reduce operator
to a partition of 1024/4 = 512 items inside the reducer graph,
and store the minimum element in the variable `min`.
Since the variable `min` also participates in the reduce operation,
it is your responsibility to initialize it to a proper value.

# Example 3: Find the Minimum L1-norm

The example below applies the method `transform_reduce`
to find the minimum L1-norm out of a point set.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2: 
 3: struct Point {
 4:   int x1 {::rand() % 10 - 5};   // random value
 5:   int x2 {::rand() % 10 - 5};   // random value
 6: };
 7: 
 8: int main() {
 9: 
10:   tf::Taskflow tf(4);
11: 
12:   std::vector<Point> points {1024};
13:   int min = std::numeric_limits<int>::max();
14: 
15:   tf.transform_reduce(points.begin(), points.end(), min,
16:     [] (int a, int b) {
17:       return std::min(a, b);
18:     },
19:     [] (const Point& point) -> int {    // Find the L1-norm of the point
20:       return std::abs(point.x1) + std::abs(point.x2);
21:     }
22:   );
23: 
24:   tf.wait_for_all();
25: 
26:   return 0;
27: }
```
