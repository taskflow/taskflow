# Reduce a Container of Items in Parallel

Parallel tasks normally produce some quantity that needs to be combined or *reduced*
through particular operations, for instance, sum.
In this example, we are going to demonstrate how to use Cpp-Taskflow
to parallelize a reduction workload.

+ [Reduce](#Reduce-through-an-Operator)
+ [Transform and Reduce](#Transform-and-Reduce)

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





