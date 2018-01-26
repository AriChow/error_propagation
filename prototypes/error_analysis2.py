import pickle
import os
import matplotlib.pyplot as plt
import sys
from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np

home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Errors
random_f1 = []
bayesian_f1 = []
mcmc_f1 = []
random_acc = []
bayesian_acc = []
mcmc_acc = []

hypers = ['haralick_distance', 'pca_whiten', 'n_components', 'n_neighbors', 'n_estimators', 'max_features', 'svm_gamma', 'svm_C']
for run in range(1, 2):
	type1 = 'bayesian_MCMC'
	bayesian_obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'))
	pipeline = bayesian_obj.best_pipelines[0]._values
	hyper = {}
	for v in pipeline.keys():
		for h in hypers:
			s = v.find(h)
			if s != -1:
				hyper[h] = pipeline[v]
	g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
									  data_loc=data_home, type1='bayesian1', fe=pipeline['feature_extraction'], dr=pipeline['dimensionality_reduction'], la=pipeline['learning_algorithm'],
									  val_splits=3, test_size=0.2)
	g.run()
	bayesian_f1.append(g.get_f1_score())
	bayesian_acc.append(g.get_accuracy())

	type1 = 'random_MCMC'
	random_obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'), encoding='latin1')

	pipeline = random_obj.best_pipelines[-1]
	g = image_classification_pipeline(pipeline.kwargs, ml_type='testing', data_name=data_name,
									  data_loc=data_home, type1='random1',
									  fe=pipeline.feature_extraction.decode('utf-8', 'ignore'),
									  dr=pipeline.dimensionality_reduction.decode('utf-8', 'ignore'),
									  la=pipeline.learning_algorithm.decode('utf-8', 'ignore'),
									  val_splits=3, test_size=0.2)
	g.run()
	random_f1.append(g.get_f1_score())
	random_acc.append(g.get_accuracy())

	type1 = 'RL_MCMC'
	mcmc_obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full_test.pkl', 'rb'), encoding='latin1')
	pipeline = mcmc_obj.best_pipelines[-1]
	g = image_classification_pipeline(pipeline.kwargs, ml_type='testing', data_name=data_name,
									  data_loc=data_home, type1='RL1', fe=pipeline.feature_extraction,
									  dr=pipeline.dimensionality_reduction, la=pipeline.learning_algorithm,
									  val_splits=3, test_size=0.2)
	g.run()
	mcmc_f1.append(g.get_f1_score())
	mcmc_acc.append(g.get_accuracy())
std_error = np.asarray([np.std(random_f1), np.std(bayesian_f1), np.std(mcmc_f1)])
error = np.asarray([np.mean(random_f1), np.mean(bayesian_f1), np.mean(mcmc_f1)])
error = error.astype('float32')
std_error = std_error.astype('float32')
x = ['Random', 'Bayesian', 'MCMC']
plt.figure(1, figsize=(12, 4))

plt.subplot(121)
plt.errorbar(range(1, 4), error, std_error, linestyle='None', marker='^', capsize=3)
plt.axhline(y=1)
plt.title('F1-scores')
plt.xlabel('Algorithm')
plt.ylabel('F1-score')
plt.xticks(range(1, 4), x)


std_error = np.asarray([np.std(random_acc), np.std(bayesian_acc), np.std(mcmc_acc)])
error = np.asarray([np.mean(random_acc), np.mean(bayesian_acc), np.mean(mcmc_acc)])
error = error.astype('float32')
std_error = std_error.astype('float32')
x = ['Random', 'Bayesian', 'MCMC']
plt.figure(1, figsize=(12, 4))

plt.subplot(122)
plt.errorbar(range(1, 4), error, std_error, linestyle='None', marker='^', capsize=3)
plt.axhline(y=1)
plt.title('Accuracy')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(range(1, 4), x)


plt.savefig(results_home + 'figures/classification_metrics' + data_name + '.jpg')
plt.close()
