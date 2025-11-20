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

export SLURM_CPU_BIND="cores"

OMP_NUM_THREADS to 1
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    # if it is not set, set it to one
    export OMP_NUM_THREADS=1
fi
./a.out


### srun --cpu-bind=core ./a.out





