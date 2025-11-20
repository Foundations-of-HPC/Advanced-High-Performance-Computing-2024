#!/bin/bash

# ----------------------------------
#       Account details
# ----------------------------------
#SBATCH -A cin_staff 
#SBATCH --partition=m100_usr_prod
#SBATCH --no-requeue

# ----------------------------------
#       Job info 
# ----------------------------------
#SBATCH -J slurm_multinode_prova
#SBATCH --out=out2.txt 
#SBATCH --err=err2.txt

# ----------------------------------
#       Resource allocation
# ----------------------------------
#SBATCH --mem=246000
#SBATCH --time=00:10:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 

#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
export SLURM_CPU_BIND="cores"

srun ./a.out
