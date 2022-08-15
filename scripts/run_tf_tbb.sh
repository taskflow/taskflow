#!/bin/bash -l
#SBATCH -w gorgon1

WORK_DIR=/home/phrygiangates/taskflow/scripts
EXEC=/home/phrygiangates/taskflow/build/benchmarks/data_pipeline
PERF="perf stat -d"
TIME="/usr/bin/time -v"

if [ ! -d "$WORK_DIR/tf_vs_tbb" ]; then
  mkdir $WORK_DIR/tf_vs_tbb
fi

cd $WORK_DIR
PLIST="ssss spsp sppp ssssssss spspspsp sppppppp ssssssssssssssss spspspspspspspsp sppppppppppppppp"
NLIST="4 8 16 32 64 80"
for p in $PLIST; do
  if [ -e "tf_vs_tbb/$p.txt" ]; then
    rm "tf_vs_tbb/$p.txt"
  fi
  for n in $NLIST; do
    for m in tf tbb; do
      $EXEC -t $n -r 3 -m $m -l $n -p $p >> "tf_vs_tbb/$p.txt"
    done
  done
done 