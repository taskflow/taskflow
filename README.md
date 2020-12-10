# Taskflow <img align="right" width="10%" src="image/taskflow_logo.png">

[![Linux Build Status](https://travis-ci.com/taskflow/taskflow.svg?branch=master)](https://travis-ci.com/taskflow/taskflow)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/rbjl16i6c9ahxr16?svg=true)](https://ci.appveyor.com/project/tsung-wei-huang/taskflow)
[![Wiki](image/api-doc.svg)][wiki]
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
visit the [documentation][wiki] to learn more about Taskflow.
Technical details can be referred to our [IPDPS paper][IPDPS19].

# Start Your First Taskflow Program

The following program (`simple.cpp`) creates four tasks 
`A`, `B`, `C`, and `D`, where `A` runs before `B` and `C`, and `D`
runs after `B` and `C`.
When `A` finishes, `B` and `C` can run in parallel.

<p align="center">
<img src="doxygen/images/simple.svg">
</p>

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

Visit [documentation](https://taskflow.github.io/taskflow/index.html) 
for a complete overview of Taskflow.

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
[GitHub contributors]:   https://github.com/taskflow/taskflow/graphs/contributors
[GraphViz]:              https://www.graphviz.org/
[AwesomeGraphViz]:       https://dreampuf.github.io/GraphvizOnline/
[OpenMP Tasking]:        https://www.openmp.org/spec-html/5.0/openmpsu99.html 
[TBB FlowGraph]:         https://www.threadingbuildingblocks.org/tutorial-intel-tbb-flow-graph
[OpenTimer]:             https://github.com/OpenTimer/OpenTimer
[DtCraft]:               https://github.com/tsung-wei-huang/DtCraft
[totalgee]:              https://github.com/totalgee
[damienhocking]:         https://github.com/damienhocking
[ForgeMistress]:         https://github.com/ForgeMistress
[Patrik Huber]:          https://github.com/patrikhuber
[DARPA IDEA]:            https://www.darpa.mil/news-events/2017-09-13
[KingDuckZ]:             https://github.com/KingDuckZ
[NSF]:                   https://www.nsf.gov/
[UIUC]:                  https://illinois.edu/
[CSL]:                   https://csl.illinois.edu/
[UofU]:                  https://www.utah.edu/
[wiki]:                  https://taskflow.github.io/taskflow/index.html
[release notes]:         https://taskflow.github.io/taskflow/Releases.html
[PayMe]:                 https://www.paypal.me/twhuang/10
[C++17]:                 https://en.wikipedia.org/wiki/C%2B%2B17
[C++14]:                 https://en.wikipedia.org/wiki/C%2B%2B14
[email me]:              mailto:twh760812@gmail.com
[Cpp Conference 2018]:   https://github.com/CppCon/CppCon2018
[ChromeTracing]:         https://www.chromium.org/developers/how-tos/trace-event-profiling-tool
[IPDPS19]:               https://tsung-wei-huang.github.io/papers/ipdps19.pdf
[WorkStealing Wiki]:     https://en.wikipedia.org/wiki/Work_stealing

[std::invoke]:           https://en.cppreference.com/w/cpp/utility/functional/invoke
[std::future]:           https://en.cppreference.com/w/cpp/thread/future

[cuda-zone]:             https://developer.nvidia.com/cuda-zone
[nvcc]:                  https://developer.nvidia.com/cuda-llvm-compiler
[cuda-toolkit]:          https://developer.nvidia.com/cuda-toolkit
[cudaGraph]:             https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html

[Firestorm]:             https://github.com/ForgeMistress/Firestorm
[Shiva]:                 https://shiva.gitbook.io/project/shiva
[PID Framework]:         http://pid.lirmm.net/pid-framework/index.html
[NovusCore]:             https://github.com/novuscore/NovusCore
[SA-PCB]:                https://github.com/choltz95/SA-PCB

[Presentation]:          https://taskflow.github.io/

