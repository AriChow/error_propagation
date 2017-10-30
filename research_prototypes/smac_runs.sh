#!/usr/bin/env bash

/home/aritra/anaconda2/envs/py36/bin/python3.6 smac_breast_cancer.py
/home/aritra/anaconda2/envs/py36/bin/python3.6 smac_pipeline_agnostic_feature_extraction.py
/home/aritra/anaconda2/envs/py36/bin/python3.6 smac_pipeline_agnostic_dimensionality_reduction.py
/home/aritra/anaconda2/envs/py36/bin/python3.6 smac_pipeline_agnostic_learning_algorithm.py