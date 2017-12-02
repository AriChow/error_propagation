import pickle
import os
import numpy as np
from prototypes.reinforcement_based_mcmc import RL_MCMC

home = os.path.expanduser('~')
dataset = 'matsc_dataset2'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
max_iters = 21

pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
rm = RL_MCMC(data_name=dataset, data_loc=data_home, results_loc=results_home, pipeline=pipeline)
rm.populate_paths()
paths = rm.paths
errors = np.zeros((max_iters, 4))
for i in range(max_iters):
	[obj, best_pipeline, best_error, time] = pickle.load(open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	pipelines = obj.best_pipelines
	potential = obj.potential
	# Final Errors
	errors[i, 3] = best_error

	# Feature extraction
	p = pipeline['feature_extraction']
	errs = []
	for j in range(len(p)):
		alg = p[j]
		err = []
		for k in range(len(pipelines)):
			if paths[k][0] == alg:
				err.append(potential[k])
		errs.append(min(err))
	errors[i, 0] = np.mean(errs)

	# Dimensionality reduction
	p = pipeline['dimensionality_reduction']
	errs = []
	for j in range(len(p)):
		alg = p[j]
		err = []
		for k in range(len(pipelines)):
			if paths[k][1] == alg:
				err.append(potential[k])
		errs.append(min(err))
	errors[i, 1] = np.mean(errs)

	# Learning algorithms
	p = pipeline['learning_algorithm']
	errs = []
	for j in range(len(p)):
		alg = p[j]
		err = []
		for k in range(len(pipelines)):
			if paths[k][2] == alg:
				err.append(potential[k])
		errs.append(min(err))
	errors[i, 2] = np.mean(errs)

pickle.dump(errors, open(results_home + 'intermediate/bayesian_MCMC/errors_' + dataset + '.pkl', 'wb'))
