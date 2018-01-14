#!/bin/sh
#SBATCH --job-name=reinforcement1
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python reinforcement_mcmc_main1.py breast barn
python reinforcement_mcmc_main1.py matsc_dataset1 barn
python reinforcement_mcmc_main1.py matsc_dataset2 barn
python reinforcement_mcmc_main1.py brain barn
python reinforcement_mcmc_main1.py bone barn