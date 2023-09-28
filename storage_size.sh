#!/bin/bash

for s in 200 400 600 800 1000; do
    for m in JL SimHash PrioritySampling; do
        echo "Running: python experiment.py -sketch_methods $m -iterations 100 -storage_size $s -log true"
        python experiment.py -sketch_methods $m -iterations 100 -storage_size $s -log true
    done
done

echo "All tasks are done!"