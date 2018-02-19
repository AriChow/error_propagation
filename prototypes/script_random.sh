#!/bin/sh
#SBATCH --job-name=random
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python random_mcmc_main.py breast barn 0 5
python random_mcmc_main.py matsc_dataset1 barn 0 5
python random_mcmc_main.py matsc_dataset2 barn 0 5
python random_mcmc_main.py brain barn 0 5
python random_mcmc_main.py bone barn 0 5