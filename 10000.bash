#!/bin/bash
#SBATCH --job-name=experiment      # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --mem=4GB                 # Memory per node
#SBATCH --time=1:00:00            # Maximum runtime (hh:mm:ss)
#SBATCH --partition=your_partition # Specify the partition or queue name
#SBATCH --output=experiment_%j.out # Output file name (%j expands to job ID)
#SBATCH --error=experiment_%j.err  # Error file name (%j expands to job ID)

# Load any necessary modules
# module load your_module(s)

# Define your variables
m="JL" # Change this to "SimHash", "PrioritySampling", or use an array if you want to test multiple values
s="200 400 600 800 1000" # Storage size values
o="0.01 0.1 0.5 1" # Overlap values

# Loop through the parameter combinations and submit jobs
for sketch_method in $m; do
    for storage_size in $s; do
        for overlap in $o; do
            # Construct the log file name
            log_file="/results/olp${overlap}_${sketch_method}.csv"

            # Submit the Python script as a job
            sbatch <<EOT
#!/bin/bash
python experiment.py -sketch_methods $sketch_method -overlap $overlap -iterations 100 -storage_size $storage_size -log true -log $log_file
EOT
        done
    done
done
