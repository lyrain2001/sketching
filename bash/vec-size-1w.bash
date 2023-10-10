#!/bin/bash
#SBATCH --job-name=sketching      
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=1         
#SBATCH --mem=4GB                 
#SBATCH --time=10:00:00            
#SBATCH --output=sketching_%j.out 
#SBATCH --error=sketching_%j.err  

export PATH=$PATH:/scratch/yl6624/sketching

m="JL SimHash PrioritySampling UnweightedPrioritySampling UnweightedPrioritySamplingSimHash" 
s="200 400 600 800 1000" 
o="0.01 0.1 0.5 1" 

for sketch_method in $m; do
    for storage_size in $s; do
        for overlap in $o; do
            log_file="./results/olp${overlap}_${sketch_method}.csv"
            echo "Running: python experiment.py -sketch_methods $sketch_method -iterations 100 -storage_size $storage_size -log true"
            python src/experiment.py -sketch_methods $sketch_method -overlap $overlap -iterations 100 -storage_size $storage_size -log true -log_name $log_file
        done
    done
done
