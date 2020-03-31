#!/bin/bash

TIME="/usr/bin/time -f \"elapsed_%e_maxmem_%M\""

#FILES="graph/graph50.txt    \
#       graph/graph2550.txt  \
#       graph/graph5050.txt"

FILES="graph/graph50.txt    \
       graph/graph2550.txt  \
       graph/graph5050.txt  \
       graph/graph7550.txt  \
       graph/graph10050.txt \
       graph/graph12550.txt \
       graph/graph15050.txt \
       graph/graph17550.txt \
       graph/graph20050.txt \
       graph/graph22550.txt \
       graph/graph25050.txt \
       graph/graph27550.txt \
       graph/graph30050.txt \
       graph/graph32550.txt \
       graph/graph35050.txt \
       graph/graph37550.txt \
       graph/graph40050.txt \
       graph/graph42550.txt \
       graph/graph45050.txt \
       graph/graph47550.txt \
       graph/graph50050.txt \
       graph/graph52550.txt \
       graph/graph55050.txt \
       graph/graph57550.txt \
       graph/graph60050.txt \
       graph/graph62550.txt \
       graph/graph65050.txt \
       graph/graph67550.txt \
       graph/graph70050.txt \
       graph/graph72550.txt \
       graph/graph75050.txt \
       graph/graph77550.txt \
       graph/graph80050.txt \
       graph/graph82550.txt \
       graph/graph85050.txt \
       graph/graph87550.txt \
       graph/graph90050.txt \
       graph/graph92550.txt \
       graph/graph95050.txt \
       graph/graph97550.txt \
       graph/graph100050.txt"



echo "tf"
for graph in $FILES; do
  res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 10 -f $graph -m tf > /dev/null" 2>&1`
  cpu="$(cut -d'_' -f2 <<<"$res")"
  mem="$(cut -d'_' -f4 <<<"$res")"
  echo $mem
done

echo "omp"
for graph in $FILES; do
  res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 10 -f $graph -m omp > /dev/null" 2>&1`
  cpu="$(cut -d'_' -f2 <<<"$res")"
  mem="$(cut -d'_' -f4 <<<"$res")"
  echo $mem
done

echo "tbb"
for graph in $FILES; do
  res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 10 -f $graph -m tbb > /dev/null" 2>&1`
  cpu="$(cut -d'_' -f2 <<<"$res")"
  mem="$(cut -d'_' -f4 <<<"$res")"
  echo $mem
done

echo "hpx"
for graph in $FILES; do
  res=`eval "${TIME} ./hpx $graph 4 1 10 > /dev/null" 2>&1`
  cpu="$(cut -d'_' -f2 <<<"$res")"
  mem="$(cut -d'_' -f4 <<<"$res")"
  echo $mem
done

echo "starpu"
for graph in $FILES; do
  res=`eval "${TIME} ./starpu $graph 4 1 10 > /dev/null" 2>&1`
  cpu="$(cut -d'_' -f2 <<<"$res")"
  mem="$(cut -d'_' -f4 <<<"$res")"
  echo $mem
done

#echo "starpu"
#for graph in $FILES; do
#  #echo "running $graph"
#  #${TIME} $1 $graph $2 $3 $4
#  #$1 $graph $2 $3 $4 2>&1
#  ./starpu $graph 4 1 10 2>&1
#done

#echo "hpx"
#for graph in $FILES; do
#  #${TIME} $1 $graph $2 $3 $4
#  #$1 $graph $2 $3 $4 2>&1
#  ${TIME} ./hpx $graph 4 1 10 2>&1
#done

#
#echo "omp"
#./hetero_traversal -t 4 -g 1 -r 10 -m omp 2>&1
# 
# echo "tf"
#./hetero_traversal -t 4 -g 1 -r 10 -m tf 2>&1
#
#echo "tbb"
#./hetero_traversal -t 4 -g 1 -r 10 -m tbb 2>&1



