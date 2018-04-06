#!/bin/sh
#SBATCH --job-name=random1
#SBATCH -t 35:59:00
#SBATCH -D /home/aritra/Documents/research/EP_project/error_propagation

cd /home/aritra/Documents/research/EP_project/error_propagation/
python random_mcmc_main1.py breast
python random_mcmc_main1.py matsc_dataset1
python random_mcmc_main1.py matsc_dataset2
python random_mcmc_main1.py brain
