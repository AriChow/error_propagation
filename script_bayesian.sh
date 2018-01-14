#!/bin/sh

cd /home/aritra/Documents/research/EP_project/error_propagation
python bayesian_mcmc_main.py breast Documents/research
python bayesian_mcmc_main.py matsc_dataset1 Documents/research
python bayesian_mcmc_main.py matsc_dataset2 Documents/research
python bayesian_mcmc_main.py brain Documents/research
python bayesian_mcmc_main.py bone Documents/research