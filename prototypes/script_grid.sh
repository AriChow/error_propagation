#!/bin/sh
#SBATCH --job-name=grid
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python grid_mcmc_main.py breast barn
python grid_mcmc_main.py matsc_dataset1 barn
python grid_mcmc_main.py matsc_dataset2 barn
python grid_mcmc_main.py brain barn
python grid_mcmc_main.py bone barn
