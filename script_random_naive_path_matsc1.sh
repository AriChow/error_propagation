#!/bin/sh
#SBATCH --job-name=random_naive_path_matsc1
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python random_mcmc_main_naive.py matsc_dataset1 barn 0 5