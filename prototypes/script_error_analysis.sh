#!/bin/sh

cd /home/aritra/Documents/research/EP_project/error_propagation/prototypes
#python error_analysis.py breast grid_MCMC
#python error_analysis.py matsc_dataset1 grid_MCMC
#python error_analysis.py matsc_dataset2 grid_MCMC
#python error_analysis.py brain grid_MCMC
#python error_analysis.py bone grid_MCMC

#python error_analysis.py breast bayesian_MCMC
#python error_analysis.py matsc_dataset1 bayesian_MCMC
#python error_analysis.py matsc_dataset2 bayesian_MCMC
#python error_analysis.py brain bayesian_MCMC
#python error_analysis.py bone bayesian_MCMC

/home/aritra/anaconda2/bin/python2.7 error_analysis.py breast random_MCMC 1 5
/home/aritra/anaconda2/bin/python2.7 error_analysis.py matsc_dataset1 random_MCMC 1 4
/home/aritra/anaconda2/bin/python2.7 error_analysis.py matsc_dataset2 random_MCMC 1 5
/home/aritra/anaconda2/bin/python2.7 error_analysis.py brain random_MCMC 1 5
/home/aritra/anaconda2/bin/python2.7 error_analysis.py bone random_MCMC 1 5

#python error_analysis.py breast RL_MCMC
#python error_analysis.py matsc_dataset1 RL_MCMC
#python error_analysis.py matsc_dataset2 RL_MCMC
#python error_analysis.py brain RL_MCMC
#python error_analysis.py bone RL_MCMC

#python error_analysis1.py breast
#python error_analysis1.py matsc_dataset1
#python error_analysis1.py matsc_dataset2
#python error_analysis1.py brain
#python error_analysis1.py bone