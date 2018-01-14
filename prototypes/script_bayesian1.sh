#!/bin/sh

cd /home/aritra/Documents/research/EP_project/error_propagation
python bayesian_mcmc_main1.py breast Documents/research
python bayesian_mcmc_main1.py matsc_dataset1 Documents/research
python bayesian_mcmc_main1.py matsc_dataset2 Documents/research
python bayesian_mcmc_main1.py brain Documents/research
python bayesian_mcmc_main1.py bone Documents/research