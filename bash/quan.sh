#!/bin/bash

s="1000 2000 3000 4000 5000"
o="0.01 0.1 0.5 1" 


for overlap in $o; do
    for sketch_size in $s; do
        for m in JL SimHash PrioritySampling; do
            log_file="./results/quantization/${overlap}"
            echo "Running: python src/experiment_quan.py  -sketch_methods $m -overlap $overlap -iterations 100 -sketch_size $sketch_size -log true -log_name $log_file"
            python src/experiment_quan.py  -sketch_methods $m -overlap $overlap -iterations 100 -sketch_size $sketch_size -log true -log_name $log_file
        done
    done
done

echo "All tasks are done!"