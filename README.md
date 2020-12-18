# Taskflow <img align="right" width="10%" src="image/taskflow_logo.png">

<!--[![Linux Build Status](https://travis-ci.com/taskflow/taskflow.svg?branch=master)](https://travis-ci.com/taskflow/taskflow)-->
[![taskflow](https://circleci.com/gh/taskflow/taskflow.svg?style=svg)](https://app.circleci.com/pipelines/github/taskflow)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/rbjl16i6c9ahxr16?svg=true)](https://ci.appveyor.com/project/tsung-wei-huang/taskflow)
[![Ubuntu](https://github.com/taskflow/taskflow/workflows/Ubuntu/badge.svg)](https://github.com/taskflow/taskflow/actions?query=workflow%3AUbuntu)
[![macOS](https://github.com/taskflow/taskflow/workflows/macOS/badge.svg)](https://github.com/taskflow/taskflow/actions?query=workflow%3AmacOS)
[![Windows](https://github.com/taskflow/taskflow/workflows/Windows/badge.svg)](https://github.com/taskflow/taskflow/actions?query=workflow%3AWindows)
[![Wiki](image/api-doc.svg)][documentation]
[![TFProf](image/tfprof.svg)](https://taskflow.github.io/tfprof/)
[![Cite](image/cite-ipdps.svg)][IPDPS19]

Taskflow helps you quickly write parallel and heterogeneous tasks programs in modern C++

# Why Taskflow?

Taskflow is faster, more expressive, and easier for drop-in integration
than many of existing task programming frameworks
in handling complex parallel workloads.

![](image/performance.png)

Taskflow lets you quickly implement task decomposition strategies
that incorporate both regular and irregular compute patterns,
together with an efficient *work-stealing* scheduler to optimize your multithreaded performance.

| [Static Tasking](#get-started-with-taskflow) | [Dynamic Tasking](#dynamic-tasking) |
| :------------: | :-------------: |
| ![](image/static_graph.svg) | <img align="right" src="image/dynamic_graph.svg" width="100%"> |

Taskflow supports conditional tasking for you to make rapid control-flow decisions
across dependent tasks to implement cycles and conditions that were otherwise difficult to do
with existing tools.

| [Conditional Tasking](#conditional-tasking) |
| :-----------------: |
| ![](image/condition.svg) |

Taskflow is composable. You can create large parallel graphs through
composition of modular and reusable blocks that are easier to optimize
at an individual scope.

| [Taskflow Composition](#composable-tasking) |
| :---------------: |
|![](image/framework.svg)|

Taskflow supports heterogeneous tasking for you to
accelerate a wide range of scientific computing applications
by harnessing the power of CPU-GPU collaborative computing.

| [Concurrent CPU-GPU Tasking](#concurrent-cpu-gpu-tasking) |
| :-----------------: |
| ![](image/cudaflow.svg) |


Taskflow provides visualization and tooling needed for profiling Taskflow programs.

| [Taskflow Profiler](https://taskflow.github.io/tfprof) |
| :-----------------: |
| ![](image/tfprof.png) |

We are committed to support trustworthy developments for both academic and industrial research projects
in parallel computing. Check out [Who is Using Taskflow](#who-is-using-taskflow) and what our users say:

+ *"Taskflow is the cleanest Task API I've ever seen." [Damien Hocking @Corelium Inc](http://coreliuminc.com)*
+ *"Taskflow has a very simple and elegant tasking interface. The performance also scales very well." [Glen Fraser][totalgee]*
+ *"Taskflow lets me handle parallel processing in a smart way." [Hayabusa @Learning](https://cpp-learning.com/cpp-taskflow/)*
+ *"Taskflow improves the throughput of our graph engine in just a few hours of coding." [Jean-Michaël @KDAB](https://ossia.io/)*
+ *"Best poster award for open-source parallel programming library." [Cpp Conference 2018][Cpp Conference 2018]*
+ *"Second Prize of Open-source Software Competition." [ACM Multimedia Conference 2019](https://tsung-wei-huang.github.io/img/mm19-ossc-award.jpg)*

See a quick [presentation](https://taskflow.github.io/) and
visit the [documentation][documentation] to learn more about Taskflow.
Technical details can be referred to our [IPDPS paper][IPDPS19].

# Start Your First Taskflow Program

The following program (`simple.cpp`) creates four tasks 
`A`, `B`, `C`, and `D`, where `A` runs before `B` and `C`, and `D`
runs after `B` and `C`.
When `A` finishes, `B` and `C` can run in parallel.



```cpp
#include <taskflow/taskflow.hpp>  // Taskflow is header-only

int main(){
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create 4 tasks
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; } 
  );                                  
                                      
  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C
                                      
  executor.run(taskflow).wait(); 

  return 0;
}
```

Taskflow is *header-only* and there is no wrangle with installation.
To compile the program, clone the Taskflow project and 
tell the compiler to include the [headers](./taskflow/).

```bash
~$ git clone https://github.com/taskflow/taskflow.git  # clone it only once
~$ g++ -std=c++17 simple.cpp -I taskflow/taskflow -O2 -pthread -o simple
~$ ./simple
TaskA
TaskC 
TaskB 
TaskD
```

# Visualize Your First Taskflow Program

Taskflow comes with a built-in profiler, 
[TFProf](https://taskflow.github.io/tfprof/), 
for you to profile and visualize taskflow programs
in an easy-to-use web-based interface.

![](doxygen/images/tfprof.png)

```bash
# run the program with the environment variable TF_ENABLE_PROFILER enabled
~$ TF_ENABLE_PROFILER=simple.json ./simple
~$ cat simple.json
[
{"executor":"0","data":[{"worker":0,"level":0,"data":[{"span":[172,186],"name":"0_0","type":"static"},{"span":[187,189],"name":"0_1","type":"static"}]},{"worker":2,"level":0,"data":[{"span":[93,164],"name":"2_0","type":"static"},{"span":[170,179],"name":"2_1","type":"static"}]}]}
]
# paste the profiling json data to https://taskflow.github.io/tfprof/
```

In addition to execution diagram, you can dump the graph to a DOT format 
and visualize it using a number of free [GraphViz][GraphViz] tools.

```
// dump the taskflow graph to a DOT format through std::cout
taskflow.dump(std::cout); 
```

<p align="center"><img src="doxygen/images/simple.svg"></p>

# Supported Compilers

To use Taskflow, you only need a compiler that supports C++17:

+ GNU C++ Compiler at least v7.0 with -std=c++17
+ Clang C++ Compiler at least v6.0 with -std=c++17
+ Microsoft Visual Studio at least v19.27 with /std:c++17
+ AppleClang Xode Version at least v12.0 with -std=c++17
+ Nvidia CUDA Toolkit and Compiler (nvcc) at least v11.1 with -std=c++17
+ Intel C++ Compiler (nvcc) at least v19.0.1 with -std=c++17

Taskflow works on Linux, Windows, and Mac OS X.

# Learn More about Taskflow

Visit our [project website][Project Website] and [documentation][documentation]
to learn more about Taskflow. To get involved:

  + See [release notes][release notes] to stay up-to-date with newest versions
  + Read the step-by-step tutorial at [cookbook][cookbook]
  + Submit an issue at [GitHub issues][GitHub issues]
  + Find out our technical details at [references][references]
  + Watch our technical talks at YouTube

| [CppCon20 Tech Talk][cppcon20 talk] | [MUC++ Tech Talk](https://www.youtube.com/watch?v=u8Mc_WgGwVY) |
| :------------: | :-------------: |
| ![](doxygen/images/cppcon20-thumbnail.jpg) | <img align="right" src="doxygen/images/muc++20-thumbnail.jpg" width="100%"> |

We are committed to support trustworthy developments for 
both academic and industrial research projects in parallel 
and heterogeneous computing. 
At the same time, we appreciate all Taskflow [contributors][contributors]!

# License

Taskflow is licensed with the [MIT License](./LICENSE). 
You are completely free to re-distribute your work derived from Taskflow.

* * *

[Tsung-Wei Huang]:       https://tsung-wei-huang.github.io/
[Chun-Xun Lin]:          https://github.com/clin99
[Martin Wong]:           https://ece.illinois.edu/directory/profile/mdfwong
[Gitter badge]:          ./image/gitter_badge.svg
[GitHub releases]:       https://github.com/taskflow/taskflow/releases
[GitHub issues]:         https://github.com/taskflow/taskflow/issues
[GitHub insights]:       https://github.com/taskflow/taskflow/pulse
[GitHub pull requests]:  https://github.com/taskflow/taskflow/pulls
[GraphViz]:              https://www.graphviz.org/
[Project Website]:       https://taskflow.github.io/
[cppcon20 talk]:         https://www.youtube.com/watch?v=MX15huP5DsM
[contributors]:          https://taskflow.github.io/taskflow/contributors.html
[OpenMP Tasking]:        https://www.openmp.org/spec-html/5.0/openmpsu99.html 
[TBB FlowGraph]:         https://www.threadingbuildingblocks.org/tutorial-intel-tbb-flow-graph
[OpenTimer]:             https://github.com/OpenTimer/OpenTimer
[DtCraft]:               https://github.com/tsung-wei-huang/DtCraft
[totalgee]:              https://github.com/totalgee
[damienhocking]:         https://github.com/damienhocking
[ForgeMistress]:         https://github.com/ForgeMistress
[Patrik Huber]:          https://github.com/patrikhuber
[KingDuckZ]:             https://github.com/KingDuckZ
[NSF]:                   https://www.nsf.gov/
[UIUC]:                  https://illinois.edu/
[CSL]:                   https://csl.illinois.edu/
[UofU]:                  https://www.utah.edu/
[documentation]:         https://taskflow.github.io/taskflow/index.html
[release notes]:         https://taskflow.github.io/taskflow/Releases.html
[cookbook]:              https://taskflow.github.io/taskflow/pages.html
[references]:            https://taskflow.github.io/taskflow/References.html
[PayMe]:                 https://www.paypal.me/twhuang/10
[C++17]:                 https://en.wikipedia.org/wiki/C%2B%2B17
[C++14]:                 https://en.wikipedia.org/wiki/C%2B%2B14
[email me]:              mailto:twh760812@gmail.com
[Cpp Conference 2018]:   https://github.com/CppCon/CppCon2018
[IPDPS19]:               https://tsung-wei-huang.github.io/papers/ipdps19.pdf

[cuda-zone]:             https://developer.nvidia.com/cuda-zone
[nvcc]:                  https://developer.nvidia.com/cuda-llvm-compiler
[cudaGraph]:             https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html

[Firestorm]:             https://github.com/ForgeMistress/Firestorm
[Shiva]:                 https://shiva.gitbook.io/project/shiva
[PID Framework]:         http://pid.lirmm.net/pid-framework/index.html
[NovusCore]:             https://github.com/novuscore/NovusCore
[SA-PCB]:                https://github.com/choltz95/SA-PCB

