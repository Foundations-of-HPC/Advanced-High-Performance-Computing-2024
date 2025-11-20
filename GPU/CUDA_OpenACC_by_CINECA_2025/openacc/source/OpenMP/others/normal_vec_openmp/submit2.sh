#!/bin/bash
#SBATCH --mem=246000
#SBATCH -J slurm_multinode_prova
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

#SBATCH --no-requeue
#SBATCH --partition=m100_usr_prod
#SBATCH -A cin_staff 
##SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
##SBATCH --mail-user=n.shukla@cineca.it

#SBATCH --out=out.txt 
#SBATCH --err=err.txt

#SBATCH --ntasks-per-node=64

echo "submitting OpenMP parallel code"
echo "========================================="

for a in 4 8 16 32 64 128 256
do
        export OMP_NUM_THREADS=$a
        export OMP_PROC_BIND=true
	srun --cpu-bind=core ./a.out
done

echo "========================================="
echo "Successfull, done!"
echo "========================================="







