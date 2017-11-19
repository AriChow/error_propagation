import numpy as np
import os
from prototypes.random_based_mcmc import random_MCMC
import pickle

if __name__=='__main__':
	home = os.path.expanduser('~')
	dataset = 'breast'
	data_home = home + '/Documents/research/EP_project/data/'
	results_home = home + '/Documents/research/EP_project/results/'
	# Gradient calculation
	# Empty features directory
	import glob
	files = glob.glob(data_home + 'features/*.npz')
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
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20, iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC.pkl', 'wb'))

	# FEATURE EXTRACTION AGNOSTIC
	type1 = 'random_MCMC_VGG'
	pipeline['feature_extraction'] = ['VGG']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_VGG.pkl', 'wb'))

	type1 = 'random_MCMC_inception'
	pipeline['feature_extraction'] = ['inception']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_inception.pkl', 'wb'))

	type1 = 'random_MCMC_haralick'
	pipeline['feature_extraction'] = ['haralick']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_haralick.pkl', 'wb'))

	# DIMENSIONALITY REDUCTION AGNOSTIC
	type1 = 'random_MCMC_PCA'
	pipeline['dimensionality_reduction'] = ['PCA']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_PCA.pkl', 'wb'))

	type1 = 'random_MCMC_ISOMAP'
	pipeline['dimensionality_reduction'] = ['ISOMAP']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_ISOMAP.pkl', 'wb'))

	# LEARNING ALGORITHM AGNOSTIC
	type1 = 'random_MCMC_RF'
	pipeline['learning_algorithm'] = ['RF']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_RF.pkl', 'wb'))

	type1 = 'random_MCMC_SVM'
	pipeline['learning_algorithm'] = ['SVM']
	rm = random_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=50)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/random_MCMC/random_MCMC_SVM.pkl', 'wb'))
