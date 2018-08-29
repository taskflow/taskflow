# Frequently Asked Questions

This page summarizes a list of frequently asked questions about Cpp-Taskflow.
If you cannot find a solution here, please post an issue [here][Github issues].

## General Questions

#### Q: How do I use Cpp-Taskflow in my projects?

**A:** Cpp-Taskflow is a header-only library with zero dependencies. 
The only thing you need is a [C++17][C++17] compiler.
To use Cpp-Taskflow, simply drop the folder 
[taskflow](../taskflow) to your project and include [taskflow.hpp](../taskflow/taskflow.hpp).

#### Q: What is the difference between static tasking and dynamic tasking?

**A:** Static tasking refers to those tasks created before execution,
while dynamic tasking refers to those tasks created during the execution of static tasks
or dynamic tasks (nested).
Dynamic tasks created by the same task node are grouped together to a subflow.

| Static Tasking | Dynamic Tasking |
| :------------: | :-------------: |
| ![](../image/static_graph.png) | ![](../image/dynamic_graph.png) |



#### Q: How many tasks can Cpp-Taskflow handle?

**A:** Cpp-Taskflow is a very lightweight and efficient tasking library.
It has been applied in many academic and industry projects to scale up their existing workload.
A research project [OpenTimer][OpenTimer] has used Cpp-Taskflow to deal with hundreds of millions of tasks.

#### Q: What are the differences between Cpp-Taskflow and other tasking libraries?

**A:** From our humble opinion, Cpp-Taskflow is superior in its tasking API, interface, and performance.
In most cases, users can quickly master Cpp-Taskflow to create large and complex dependency graphs
in just a few minutes.
The performance scales very well and is comparable to hard-coded multi-threading.
Of course, the judge is always left for users -:)

## Compile Issues

#### Q: I can't get Cpp-Taskflow compiled in my project!

**A:** Please make sure your compile supports the latest version of [C++17][C++17]. 
Make sure your project meets the System Requirements described at [README][README].

#### Q: Clang can't compile due to the use of std::variant.

**A:** Cpp-Taskflow uses `std::variant` to enable uniform interface between static and dynamic tasking.
However it has been reported in 
[Clang Bug 31852](https://bugs.llvm.org/show_bug.cgi?id=31852) that
clang-6.0 (and before) does not correctly pick up the friend declaration with auto-deduced return type.
While this can be fixed in the future version of clang, 
we use the [variant patch](../patch/clang_variant.hpp) posted
[here](https://gcc.gnu.org/viewcvs/gcc?view=revision&revision=258854) as a workaround.
For clang users, you will need to use this patch in `taskflow.hpp` as follows:

```cpp
#if defined(__clang__)
  #include <patch/clang_variant.hpp>
#else
  #include <variant>
#endif
```

## Programming Questions

#### Q: What is the difference between Cpp-Taskflow threads and workers?

**A:** The master thread owns the thread pool and can spawn workers to run tasks 
or shutdown the pool. 
Giving taskflow `N` threads means using `N` threads to do the works, 
and there is a total of `N+1` threads (including the master threads) in the program.

```cpp
tf::Taskflow(N);    // N workers, N+1 threads in the program.
```

If there is no worker threads in the pool, the master thread will do all the works by itself.

#### Q: Is taskflow thread-safe?
**A:** No, the taskflow object is not thread-safe. You can't create tasks from multiple threads
at the same time.

#### Q: My program hangs and never returns after dispatching a taskflow graph. What's wrong?

**A:** When the program hangs forever it is very likely your taskflow graph has a cycle.
Try the `dump` method to debug the graph before dispatching your taskflow graph.
If there is no cycle, make sure you are using `future.get()` in the right way, 
i.e., not blocking your control flow.

#### Q: In the following example where B spawns a joined subflow of two tasks B1 and B2, do they run concurrently with task A?

<p>
<img src="../image/dynamic_graph.png" width="60%">
</p>

**A:** No. The subflow is spawned during the execution of B, and at this point A must finish
because A precedes B. This gives rise to the fact B1 and B2 must run after A. 
This graph may looks strange because B seems to run twice!
However, Cpp-Taskflow will schedule B only once to create its subflow.
Whether this subflow joins or detaches from B only affects the future object returned from B.





* * *
[Github issues]:         https://github.com/cpp-taskflow/cpp-taskflow/issues
[OpenTimer]:             https://github.com/OpenTimer/OpenTimer
[README]:                ../README.md
[C++17]:                 https://en.wikipedia.org/wiki/C%2B%2B17



