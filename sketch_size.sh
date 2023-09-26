#!/bin/bash

for s in 1000 2000 3000 4000 5000; do
    for m in JL SimHash PrioritySampling SimHashM; do
        echo "Running: python experiment.py -sketch_methods $m -iterations 100 -sketch_size $s -log true"
        python experiment.py -sketch_methods $m -iterations 100 -sketch_size $s -log true
    done
done

# for s in 1000 2000 3000 4000 5000; do
#     echo "Running: python experiment.py -sketch_methods SimHash -iterations 100 -sketch_size $s"
#     python experiment.py -sketch_methods SimHash -iterations 100 -sketch_size $s
# done

echo "All tasks are done!"