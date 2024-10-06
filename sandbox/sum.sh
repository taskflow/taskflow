#!/bin/bash

# x: TF_DEFAULT_BOUNDED_TASK_QUEUE_LOG_SIZE
# y: TF_DEFAULT_UNBOUNDED_TASK_QUEUE_LOG_SIZE

for ((x=6; x<=12; x=x+1)) do
  for ((y=6; y<=12; y=y+1)) do
    filename="result-$x-$y"
    
    if [[ -f "$filename" ]]; then
      # Read the numbers from the file
      numbers=$(cat "$filename")
      
      # Calculate the sum
      sum=$(echo "$numbers" | awk '{sum+=$1} END {print sum}')
      
      # Calculate the mean
      count=$(echo "$numbers" | wc -l)
      mean=$(echo "$sum / $count" | bc -l)
      
      # Calculate the standard deviation
      stddev=$(echo "$numbers" | awk -v mean="$mean" '{sum+=($1-mean)*($1-mean)} END {print sqrt(sum/NR)}')
      
      # Calculate the min and max
      min=$(echo "$numbers" | sort -n | head -n 1)
      max=$(echo "$numbers" | sort -n | tail -n 1)
      
      # Output the results
      printf "%s %.2f %.2f %.2f %.2f %.2f\n" "$filename" "$sum" "$mean" "$stddev" "$min" "$max"
    else
      echo "File: $filename does not exist."
    fi
  done
done
