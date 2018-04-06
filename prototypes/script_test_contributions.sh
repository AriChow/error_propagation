#!/bin/sh
#SBATCH --job-name=contributions
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/EP_project/error_propagation
python test_analysis_sample.py breast barn
python test_analysis_sample.py matsc_dataset1 barn
python test_analysis_sample.py matsc_dataset2 barn
python test_analysis_sample.py brain barn
python test_analysis_sample.py bone barn