#!/bin/sh
#SBATCH --job-name=grid
#SBATCH -t 35:59:00
#SBATCH -D /home/aritra/Documents/research/EP_project/error_propagation

cd /home/aritra/Documents/research/EP_project/error_propagation
python grid_mcmc_main.py breast Documents/research 0 1
python grid_mcmc_main.py matsc_dataset1 Documents/research 0 1
python grid_mcmc_main.py matsc_dataset2 Documents/research 0 1
python grid_mcmc_main.py brain Documents/research 0 1
python grid_mcmc_main.py bone Documents/research 0 1
