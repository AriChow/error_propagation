'''
This file is for analysing results on test data samples.
'''
import numpy as np
import os
import sys
import pickle

home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'][:-1] + pipeline['dimensionality_reduction'][:-1] + pipeline['learning_algorithm'][:-1]
lenp = len(pipeline['all'])

# Random1
start = 1
stop = 2
type1 = 'random_MCMC'

E1 = np.zeros((stop - 1, lenp))
E2 = np.zeros((stop - 1, lenp))
E3 = np.zeros((stop - 1, lenp))
all_paths = []
for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')
	pipelines = obj.pipelines
	print