#!/bin/bash -l
#SBATCH -w gorgon1

WORK_DIR=/home/xiongzc/taskflow/scripts
EXEC=/home/xiongzc/taskflow/build/benchmarks/data_pipeline_dev

if [ ! -d "$WORK_DIR/normal_vs_efficient_string" ]; then
  mkdir $WORK_DIR/normal_vs_efficient_string
fi

cd $WORK_DIR
PLIST="ssss spsp sppp ssssssss spspspsp sppppppp ssssssssssssssss spspspspspspspsp sppppppppppppppp"
NLIST="4 8 16 24"
for p in $PLIST; do
  if [ -e "normal_vs_efficient_string/$p.txt" ]; then
    rm "normal_vs_efficient_string/$p.txt"
  fi
  for n in $NLIST; do
    for m in normal efficient; do
      $EXEC -t $n -r 3 -m $m -l $n -p $p -d string >> "normal_vs_efficient_string/$p.txt"
    done
  done
done 