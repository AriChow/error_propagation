#!/bin/sh
#SBATCH --job-name=random1
#SBATCH -t 35:59:00
#SBATCH -D /home/aritra/Documents/research/EP_project/error_propagation

cd /home/aritra/Documents/research/EP_project/error_propagation 0 5
python test_analysis_sample.py breast Documents/research 0 5
python random_mcmc_main1.py matsc_dataset1 Documents/research 0 5
python random_mcmc_main1.py matsc_dataset2 Documents/research 0 5
python random_mcmc_main1.py brain Documents/research 0 5
python random_mcmc_main1.py bone Documents/research 0 5
