#!/bin/bash

m="SimHash SimHashRound SimHashFunction" 
s="200 400 600 800" 
# o="0.01 0.1 0.5 1" 
o="0.1"

for overlap in $o; do
    for storage_size in $s; do
        for sketch_method in $m; do
            log_file="./results/new_simhash_results/${sketch_method}.csv"
            echo "Running: python experiment_simhash.py -sketch_methods $sketch_method -overlap $overlap -iterations 100 -storage_size $storage_size -log true"
            python src/experiment_simhash.py -sketch_methods $sketch_method -overlap $overlap -iterations 100 -storage_size $storage_size -log true -log_name $log_file
        done
    done
done

echo "All tasks are done!"