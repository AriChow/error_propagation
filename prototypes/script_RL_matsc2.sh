#!/bin/sh
#SBATCH --job-name=RL_matsc2
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python reinforcement_mcmc_main.py matsc_dataset2 barn 0 5