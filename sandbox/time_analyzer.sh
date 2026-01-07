#!/bin/bash

# Usage: ./sum.sh input.txt
# Parses runtime numbers and reports min, max, sum, average, std

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

awk '
{
  v = $(NF-1)
  sum += v
  sumsq += v * v
  if (count == 0 || v < min) min = v
  if (count == 0 || v > max) max = v
  count++
}
END {
  if (count == 0) {
    print "No valid data found"
    exit 1
  }

  avg = sum / count
  std = sqrt(sumsq / count - avg * avg)

  printf "Count : %d\n", count
  printf "Min   : %.2f sec\n", min
  printf "Max   : %.2f sec\n", max
  printf "Sum   : %.2f sec\n", sum
  printf "Avg   : %.2f sec\n", avg
  printf "Std   : %.2f sec\n", std
}
' "$1"
