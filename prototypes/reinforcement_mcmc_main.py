import numpy as np
import os
from prototypes.reinforcement_based_mcmc import RL_MCMC
import pickle

if __name__=='__main__':
	home = os.path.expanduser('~')
	dataset = 'matsc_dataset2'
	data_home = home + '/Documents/research/EP_project/data/'
	results_home = home + '/Documents/research/EP_project/results/'
	num_iters = 21
	# Gradient calculation
	# Empty features directory
	import glob
	files = glob.glob(data_home + 'features/RL/*.npz')
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
	type1 = 'RL_MCMC'
	rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20, iters=num_iters)
	rm.populate_paths()
	best_pipeline, best_error, times = rm.rlMcmc()
	pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_' + dataset + '.pkl', 'wb'))

	# # FEATURE EXTRACTION AGNOSTIC
	# type1 = 'RL_MCMC_VGG'
	# pipeline['feature_extraction'] = ['VGG']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_VGG_' + dataset + '.pkl', 'wb'))
	#
	# type1 = 'RL_MCMC_inception'
	# pipeline['feature_extraction'] = ['inception']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_inception_' + dataset + '.pkl', 'wb'))
	#
	# type1 = 'RL_MCMC_haralick'
	# pipeline['feature_extraction'] = ['haralick']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_haralick_' + dataset + '.pkl', 'wb'))
	#
	# # DIMENSIONALITY REDUCTION AGNOSTIC
	# type1 = 'RL_MCMC_PCA'
	# pipeline['dimensionality_reduction'] = ['PCA']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_PCA_' + dataset + '.pkl', 'wb'))
	#
	# type1 = 'RL_MCMC_ISOMAP'
	# pipeline['dimensionality_reduction'] = ['ISOMAP']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_ISOMAP_' + dataset + '.pkl', 'wb'))
	#
	# # LEARNING ALGORITHM AGNOSTIC
	# type1 = 'RL_MCMC_RF'
	# pipeline['learning_algorithm'] = ['RF']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_RF_' + dataset + '.pkl', 'wb'))
	#
	# type1 = 'RL_MCMC_SVM'
	# pipeline['learning_algorithm'] = ['SVM']
	# rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, type1=type1, pipeline=pipeline, path_resources=12, hyper_resources=20,
	# 			 iters=num_iters)
	# rm.populate_paths()
	# best_pipeline, best_error, times = rm.rlMcmc()
	# pickle.dump([rm, best_pipeline, best_error, times], open(results_home + 'intermediate/RL_MCMC/RL_MCMC_SVM_' + dataset + '.pkl', 'wb'))
