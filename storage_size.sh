#!/bin/bash

for s in 100 200 300; do
    for m in JL SimHashM SimHash PrioritySampling; do
        echo "Running: python experiment.py -sketch_methods $m -iterations 100 -storage_size $s -log true"
        python experiment.py -sketch_methods $m -iterations 100 -storage_size $s -log true
    done
done

echo "All tasks are done!"