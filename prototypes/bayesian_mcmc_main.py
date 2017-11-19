import numpy as np
import os
from prototypes.bayesian_based_mcmc import bayesian_MCMC
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
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20, iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC.pkl', 'wb'))

	# FEATURE EXTRACTION AGNOSTIC
	pipeline['feature_extraction'] = ['VGG']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_VGG.pkl', 'wb'))

	pipeline['feature_extraction'] = ['inception']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_inception.pkl', 'wb'))

	pipeline['feature_extraction'] = ['haralick']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_haralick.pkl', 'wb'))

	# DIMENSIONALITY REDUCTION AGNOSTIC
	pipeline['dimensionality_reduction'] = ['PCA']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_PCA.pkl', 'wb'))

	pipeline['dimensionality_reduction'] = ['ISOMAP']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_ISOMAP.pkl', 'wb'))

	# LEARNING ALGORITHM AGNOSTIC
	pipeline['learning_algorithm'] = ['RF']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_RF.pkl', 'wb'))

	pipeline['learning_algorithm'] = ['SVM']
	rm = bayesian_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline, path_resources=12, hyper_resources=20,
				 iters=1)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.bayesianmcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_SVM.pkl', 'wb'))
