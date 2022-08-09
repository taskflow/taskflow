#!/bin/bash -l
#SBATCH -w gorgon1

WORK_DIR=/home/xiongzc/taskflow/scripts
EXEC=/home/xiongzc/taskflow/build/benchmarks/data_pipeline
PERF="perf stat -d"
TIME="/usr/bin/time -v"

if [ ! -d "$WORK_DIR/tf_vs_tbb" ]; then
  mkdir $WORK_DIR/tf_vs_tbb
fi

cd $WORK_DIR
PLIST="ssss sppp ssssssss sppppppp ssssssssssssssss sppppppppppppppp"
NLIST="1 2 4 24"
for p in $PLIST; do
  if [ -e "tf_vs_tbb/$p.txt" ]; then
    rm "tf_vs_tbb/$p.txt"
  fi
  for n in $NLIST; do
    for m in tf tbb; do
      $PERF $EXEC -t $n -r 3 -m $m -l $n -p $p >> "tf_vs_tbb/$p.txt"
    done
  done
done 