#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short

# The default run (wall-clock) time is 1 minute
#SBATCH --time=60

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=10

# The default number of CPUs per task is 1 (note: CPUs are always allocated to jobs per 2)
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
#SBATCH --cpus-per-task=1

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu ins tead)
#SBATCH --mem-per-cpu=85

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Use this simple command to check that your sbatch settings are working
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load miniconda

conda activate MCTS-Time-Management

# Computations should be started with 'srun'. For example:
#srun --ntasks=1 --exclusive python ../scripts/CreateDataset.py --size 100 --temperature 1 --simulations 1,16 --environment CartPole-v1 --dataset-name 1Cluster_1-16_100-Cartpole --horizon -1 &
#srun --ntasks=1 --exclusive python ../scripts/CreateDataset.py --size 100 --temperature 1 --simulations 1,16 --environment CartPole-v1 --dataset-name 2Cluster_1-16_100-Cartpole --horizon -1 &
#srun --ntasks=1 --exclusive python ../scripts/CreateDataset.py --size 100 --temperature 1 --simulations 1,16 --environment CartPole-v1 --dataset-name 3Cluster_1-16_100-Cartpole --horizon -1 &
#srun --ntasks=1 --exclusive python ../scripts/CreateDataset.py --size 100 --temperature 1 --simulations 1,16 --environment CartPole-v1 --dataset-name 4Cluster_1-16_100-Cartpole --horizon -1 &
for ((i=1; i<=10; i++)); do
   
    dataset_name="states_Cluster_${i}_1600_s1_16-t500-CartPole-v1" 

    srun --ntasks=1 --exclusive python ../scripts/CreateDataset.py --size 1600 --temperature 500 --simulations 1,16 --environment CartPole-v1 --dataset-name "${dataset_name}" --horizon -1 &
done

wait
