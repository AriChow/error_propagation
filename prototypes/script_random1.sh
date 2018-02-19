#!/bin/sh
#SBATCH --job-name=random1
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation 0 5
python random_mcmc_main1.py breast barn 0 5
python random_mcmc_main1.py matsc_dataset1 barn 0 5
python random_mcmc_main1.py matsc_dataset2 barn 0 5
python random_mcmc_main1.py brain barn 0 5
python random_mcmc_main1.py bone barn 0 5
