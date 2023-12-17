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

# print current directory
pwd

s="1000 2000 3000 4000 5000"
o="0.01 0.1 0.5 1" 


for overlap in $o; do
    for sketch_size in $s; do
        log_file="./results/binary_zero/${overlap}_"
        echo "Running: python src/experiment_binary.py  -overlap $overlap -zeroes 0.5 -iterations 100 -sketch_size $sketch_size -log true -log_name $log_file"
        python src/experiment_binary.py  -overlap $overlap -zeroes 0.5 -iterations 100 -sketch_size $sketch_size -log true -log_name $log_file
    done
done

echo "All tasks are done!"