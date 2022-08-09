#!/bin/bash

TIME="/usr/bin/time -f \"elapsed_%e_maxmem_%M\""
PERF="perf stat -e power/energy-pkg/"

#FILES="graph/graph50.txt    \
#       graph/graph2550.txt  \
#       graph/graph5050.txt"

FILES="graph/graph50.txt    \
       graph/graph5050.txt  \
       graph/graph10050.txt \
       graph/graph15050.txt \
       graph/graph20050.txt \
       graph/graph25050.txt \
       graph/graph30050.txt \
       graph/graph35050.txt \
       graph/graph40050.txt \
       graph/graph45050.txt \
       graph/graph50050.txt \
       graph/graph55050.txt \
       graph/graph60050.txt \
       graph/graph65050.txt \
       graph/graph70050.txt \
       graph/graph75050.txt \
       graph/graph80050.txt \
       graph/graph85050.txt \
       graph/graph90050.txt \
       graph/graph95050.txt \
       graph/graph100050.txt"



#echo "tf"
#for graph in $FILES; do
#  #res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 1 -f $graph -m tf > /dev/null" 2>&1`
#  #cpu="$(cut -d'_' -f2 <<<"$res")"
#  #mem="$(cut -d'_' -f4 <<<"$res")"
#  #echo $mem
#
#  sumj=0;
#  sump=0;
#  sumt=0;
#  R=10
#  for((r=1; r<=$R; r=r+1)) do
#    res=`eval "${PERF} ./hetero_traversal -t 4 -g 1 -r 1 -f $graph -m tf > /dev/null" 2>&1`    
#    j=$(echo "$res" | grep "power" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    t=$(echo "$res" | grep "elapsed" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    p=`bc -l <<< $j/$t`
#    sumj=`bc -l <<< $sumj+$j`
#    sumt=`bc -l <<< $sumt+$t`
#    sump=`bc -l <<< $sump+$p`
#  done;
#  avgj=`bc -l <<< $sumj/$R`
#  avgt=`bc -l <<< $sumt/$R`
#  avgp=`bc -l <<< $sump/$R`
#
#  echo "$avgj $avgt $avgp"
#
#  sleep 5
#done

#
#echo "omp"
#for graph in $FILES; do
##  res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 10 -f $graph -m omp > /dev/null" 2>&1`
##  cpu="$(cut -d'_' -f2 <<<"$res")"
##  mem="$(cut -d'_' -f4 <<<"$res")"
##  echo $mem
#  sumj=0;
#  sump=0;
#  sumt=0;
#  R=10
#  for((r=1; r<=$R; r=r+1)) do
#    res=`eval "${PERF} ./hetero_traversal -t 4 -g 1 -r 1 -f $graph -m omp > /dev/null" 2>&1`    
#    j=$(echo "$res" | grep "power" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    t=$(echo "$res" | grep "elapsed" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    p=`bc -l <<< $j/$t`
#    sumj=`bc -l <<< $sumj+$j`
#    sumt=`bc -l <<< $sumt+$t`
#    sump=`bc -l <<< $sump+$p`
#  done;
#  avgj=`bc -l <<< $sumj/$R`
#  avgt=`bc -l <<< $sumt/$R`
#  avgp=`bc -l <<< $sump/$R`
#
#  echo "$avgj $avgt $avgp"
#
#  sleep 5
#done
#

#echo "tbb"
#for graph in $FILES; do
##  res=`eval "${TIME} ./hetero_traversal -t 4 -g 1 -r 10 -f $graph -m tbb > /dev/null" 2>&1`
##  cpu="$(cut -d'_' -f2 <<<"$res")"
##  mem="$(cut -d'_' -f4 <<<"$res")"
##  echo $mem
#
#  sumj=0;
#  sump=0;
#  sumt=0;
#  R=10
#  for((r=1; r<=$R; r=r+1)) do
#    res=`eval "${PERF} ./hetero_traversal -t 4 -g 1 -r 1 -f $graph -m tbb > /dev/null" 2>&1`    
#    j=$(echo "$res" | grep "power" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    t=$(echo "$res" | grep "elapsed" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    p=`bc -l <<< $j/$t`
#    sumj=`bc -l <<< $sumj+$j`
#    sumt=`bc -l <<< $sumt+$t`
#    sump=`bc -l <<< $sump+$p`
#  done;
#  avgj=`bc -l <<< $sumj/$R`
#  avgt=`bc -l <<< $sumt/$R`
#  avgp=`bc -l <<< $sump/$R`
#
#  echo "$avgj $avgt $avgp"
#
#  sleep 5
#done

#
echo "hpx"
for graph in $FILES; do
#  res=`eval "${TIME} ./hpx $graph 4 1 10 > /dev/null" 2>&1`
#  cpu="$(cut -d'_' -f2 <<<"$res")"
#  mem="$(cut -d'_' -f4 <<<"$res")"
#  echo $mem
  sumj=0;
  sump=0;
  sumt=0;
  R=2
  for((r=1; r<=$R; r=r+1)) do
    res=`eval "${PERF} ./hpx $graph 4 1 1 > /dev/null" 2>&1`    
    j=$(echo "$res" | grep "power" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    t=$(echo "$res" | grep "elapsed" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
    p=`bc -l <<< $j/$t`
    sumj=`bc -l <<< $sumj+$j`
    sumt=`bc -l <<< $sumt+$t`
    sump=`bc -l <<< $sump+$p`
  done;
  avgj=`bc -l <<< $sumj/$R`
  avgt=`bc -l <<< $sumt/$R`
  avgp=`bc -l <<< $sump/$R`

  echo "$avgj $avgt $avgp"

  sleep 3
done

#echo "starpu"
#for graph in $FILES; do
##  res=`eval "${TIME} ./starpu $graph 4 1 10 > /dev/null" 2>&1`
##  cpu="$(cut -d'_' -f2 <<<"$res")"
##  mem="$(cut -d'_' -f4 <<<"$res")"
##  echo $mem
#  sumj=0;
#  sump=0;
#  sumt=0;
#  R=10
#  for((r=1; r<=$R; r=r+1)) do
#    res=`eval "${PERF} ./starpu $graph 4 1 1 > /dev/null" 2>&1`    
#    j=$(echo "$res" | grep "power" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    t=$(echo "$res" | grep "elapsed" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
#    p=`bc -l <<< $j/$t`
#    sumj=`bc -l <<< $sumj+$j`
#    sumt=`bc -l <<< $sumt+$t`
#    sump=`bc -l <<< $sump+$p`
#  done;
#  avgj=`bc -l <<< $sumj/$R`
#  avgt=`bc -l <<< $sumt/$R`
#  avgp=`bc -l <<< $sump/$R`
#
#  echo "$avgj $avgt $avgp"
#
#  sleep 3
#done

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



