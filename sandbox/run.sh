#!/bin/bash

# x: TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE
# y: TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE

for((x=6; x<=12; x=x+1)) do
  for((y=6; y<=12; y=y+1)) do
    cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS="-DTF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE=$x -DTF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE=$y -DTF_ENABLE_ATOMIC_NOTIFIER=1" &> /dev/null;
    
    #echo "Compiling y=$y ...";
    make -j 16 &> /dev/null;
  
    #echo "Testing y=$y ...";
    make test &> /dev/null;
  
    for((i=0;i<20;i=i+1)) do
      make test | grep "Total" | grep -oP '\d+(\.\d+)?' >> result-$x-$y ;
    done
  done
done
