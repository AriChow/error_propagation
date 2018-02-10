#!/bin/sh
#SBATCH --job-name=reinforcement2
#SBATCH -t 35:59:00
#SBATCH -D /home/aritra/Documents/research/EP_project/error_propagation

cd /home/aritra/Documents/research/EP_project/error_propagation
python reinforcement_mcmc_main2.py breast Documents/research 0 5
python reinforcement_mcmc_main2.py matsc_dataset1 Documents/research 0 5
python reinforcement_mcmc_main2.py matsc_dataset2 Documents/research 0 5
python reinforcement_mcmc_main2.py brain Documents/research 0 5
python reinforcement_mcmc_main2.py bone Documents/research 0 5