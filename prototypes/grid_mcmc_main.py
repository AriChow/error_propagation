import numpy as np
import os
from prototypes.grid_based_mcmc import grid_MCMC
import pickle

if __name__ == '__main__':
	home = os.path.expanduser('~')
	dataset = 'breast'
	data_home = home + '/barn/EP_project/data/'
	results_home = home + '/barn/EP_project/results/'
	# num_iters = 21
	# Empty features directory
	import glob
	files = glob.glob(data_home + 'features/grid/*.npz')
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
	pipeline['n_estimators'] = np.round(np.linspace(8, 300, 5))
	pipeline['max_features'] = np.arange(0.3, 0.8, 0.1)
	pipeline['svm_gamma'] = np.linspace(0.01, 8, 5)
	pipeline['svm_C'] = np.linspace(0.1, 100, 5)

	# CONTROL
	type1 = 'grid_MCMC'
	for i in range(5):
		rm = grid_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, run=i+1, type1=type1, pipeline=pipeline)
		rm.populate_paths()
		rm.gridMcmc()
