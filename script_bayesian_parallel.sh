#!/bin/sh

cd /home/aritra/Documents/research/EP_project/error_propagation
python bayesian_mcmc_main_parallel.py breast Documents/research 0 5
python bayesian_mcmc_main_parallel.py matsc_dataset1 Documents/research 0 5
python bayesian_mcmc_main_parallel.py matsc_dataset2 Documents/research 0 5
python bayesian_mcmc_main_parallel.py brain Documents/research 0 5
python bayesian_mcmc_main_parallel.py bone Documents/research 0 5