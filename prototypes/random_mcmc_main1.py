import numpy as np
import os
from prototypes.random_based_mcmc2 import random_MCMC

if __name__ == '__main__':
	home = os.path.expanduser('~')
	dataset = 'matsc_dataset1'
	data_home = home + '/Documents/research/EP_project/data/'
	results_home = home + '/Documents/research/EP_project/results/'
	num_iters = 51
	# Gradient calculation
	# Empty features directory
	import glob
	files = glob.glob(data_home + 'features/random1/*.npz')
	for f in files:
		os.remove(f)
	pipeline = {}
	pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	pipeline['learning_algorithm'] = ["SVM", "RF"]
	pipeline['haralick_distance'] = range(1, 4)
	pipeline['pca_whiten'] = [True, False]
	pipeline['n_neighbors'] = range(3, 8)
	pipeline['n_components'] = range(2, 5)
	pipeline['n_estimators'] = np.round(np.linspace(8, 300, 10))
	pipeline['max_features'] = np.arange(0.3, 0.8, 0.1)
	pipeline['svm_gamma'] = np.linspace(0.01, 8, 10)
	pipeline['svm_C'] = np.linspace(0.1, 100, 10)

	# CONTROL
	type1 = 'random_MCMC'
	for i in range(5):
		rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, run=i+1, type1=type1, pipeline=pipeline, iters=num_iters)
		rm.populate_paths()
		rm.randomMcmc()
