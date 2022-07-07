WORK_DIR=/home/xiongzc/taskflow/scripts
EXEC=/home/xiongzc/taskflow/build/benchmarks/data_pipeline

if [ ! -d "$WORK_DIR/data" ]; then
  mkdir $WORK_DIR/data
fi

cd $WORK_DIR
PLIST="ssss sppp ssssssss sppppppp ssssssssssssssss sppppppppppppppp"
NLIST="4 8 16 24"
for p in $PLIST; do
  if [ -e "data/$p.txt" ]; then
    rm "data/$p.txt"
  fi
  for n in $NLIST; do
    for m in tf tbb; do
      $EXEC -t $n -r 3 -m $m -l $n -p $p >> "data/$p.txt"
    done
  done
done 