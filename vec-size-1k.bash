#!/bin/bash
#SBATCH --job-name=sketching      
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=1         
#SBATCH --mem=4GB                 
#SBATCH --time=1:00:00            
#SBATCH --output=sketching_%j.out 
#SBATCH --error=sketching_%j.err  


m="JL SimHash PrioritySampling" 
s="200 400 600 800 1000" 
o="0.01 0.1 0.5 1" 

for sketch_method in $m; do
    for storage_size in $s; do
        for overlap in $o; do
            log_file="./results/olp${overlap}_${sketch_method}.csv"

            sbatch <<EOT
#!/bin/bash
python experiment.py -sketch_methods $sketch_method -overlap $overlap -iterations 100 -storage_size $storage_size -log true -log $log_file
EOT
        done
    done
done
