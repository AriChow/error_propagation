#!/bin/sh
#SBATCH --job-name=random_parallel
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python random_mcmc_main_parallel.py breast barn
python random_mcmc_main_parallel.py matsc_dataset1 barn
python random_mcmc_main_parallel.py matsc_dataset2 barn
python random_mcmc_main_parallel.py brain barn
python random_mcmc_main_parallel.py bone barn