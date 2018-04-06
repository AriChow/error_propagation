'''
This file is for analysing results on test data samples.
'''
import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline1 import image_classification_pipeline

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
stop = 5
type1 = 'random_MCMC'

hypers = ['haralick_distance', 'pca_whiten', 'n_components', 'n_neighbors', 'n_estimators', 'max_features', 'svm_gamma', 'svm_C']

all_paths = []

for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')
	pipelines = obj.pipelines
	test_pipelines = []
	for i in range(len(pipelines)):
		pi = pipelines[i]
		hyper = pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1', fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		test_pipelines.append(g)

	pickle.dump(test_pipelines, open(results_home + 'intermediate/random_MCMC/' + type1 + '_' + data_name + '_run_' + str(run) + '_full_naive_test.pkl', 'wb'))