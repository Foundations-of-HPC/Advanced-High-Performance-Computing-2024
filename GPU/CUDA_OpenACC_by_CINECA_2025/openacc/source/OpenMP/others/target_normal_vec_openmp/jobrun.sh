#!/bin/bash

# ----------------------------------
#       Account details
# ----------------------------------
#SBATCH -A cin_staff 
#SBATCH --partition=m100_usr_prod
#SBATCH --no-requeue

# ----------------------------------
# 	Job info 
# ----------------------------------
#SBATCH -J slurm_multinode_prova
#SBATCH --out=out.txt 
#SBATCH --err=err.txt

# ----------------------------------
#       Resource allocation
# ----------------------------------
#SBATCH --mem=246000
#SBATCH --time=00:10:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 

#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

# 1 node, 4 tasks, 4 GPUs, 1 GPU visible to each task 
###SBATCH --gpu-bind=map_gpu:0,1,2,3

export SLURM_CPU_BIND="cores"

echo "submitting OpenMP parallel code"
echo "========================================="

OMP_NUM_THREADS to 1
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    # if it is not set, set it to one
    export OMP_NUM_THREADS=1
fi
./a.out



##export SLURM_CPU_BIND="cores"
##srun --ntasks=1 --cpus-per-task=1 --partition=gpu --gres=gpu:1 ./a.out 
echo "========================================="
echo "Successfull, done!"
echo "========================================="




