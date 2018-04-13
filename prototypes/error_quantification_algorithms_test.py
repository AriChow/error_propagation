import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
datasets = ['breast', 'brain', 'matsc_dataset1', 'matsc_dataset2']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["haralick"]
pipeline['dimensionality_reduction'] = ["PCA"]
pipeline['learning_algorithm'] = ["RF"]

pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']

# Random search
start = 1
stop = 6
type1 = 'random_MCMC'

for data_name in datasets:
	# alg_error = np.zeros((stop-1, 3))
	# alg_error_test = np.zeros((stop-1, 3))
	# for run in range(start, stop):
	# 	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
	# 						   str(run) + '.pkl','rb'), encoding='latin1')
	# 	pipelines = obj.pipelines
	# 	paths = obj.paths
	# 	path_pipelines = []
	# 	for i in range(len(pipelines)):
	# 		path = paths[i]
	# 		if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
	# 			path_pipelines = pipelines[i]
	# 			break
	#
	#
	# 	fe_params = set()
	# 	dr_params = set()
	# 	la_params = set()
	# 	for i in range(len(path_pipelines)):
	# 		p = path_pipelines[i]
	# 		fe_params.add(p.haralick_distance)
	# 		dr_params.add(p.pca_whiten)
	# 		la_params.add((p.n_estimators, p.max_features))
	#
	# 	fe_params = list(fe_params)
	# 	dr_params = list(dr_params)
	# 	la_params = list(la_params)
	# 	min_err = 1000000
	# 	min_fe = [1000000] * len(fe_params)
	# 	min_dr = [1000000] * len(dr_params)
	# 	min_la = [1000000] * len(la_params)
	# 	min_fe_pi = [None] * len(fe_params)
	# 	min_dr_pi = [None] * len(dr_params)
	# 	min_la_pi = [None] * len(la_params)
	# 	best_pi = None
	# 	for i in range(len(path_pipelines)):
	# 		p = path_pipelines[i]
	# 		err = p.get_error()
	# 		if err < min_err:
	# 			min_err = err
	# 			best_pi = p
	# 		for j in range(len(fe_params)):
	# 			if p.haralick_distance == fe_params[j]:
	# 				if min_fe[j] > err:
	# 					min_fe[j] = err
	# 					min_fe_pi[j] = p
	#
	# 		for j in range(len(dr_params)):
	# 			if p.pca_whiten == dr_params[j]:
	# 				if min_dr[j] > err:
	# 					min_dr[j] = err
	# 					min_dr_pi[j] = p
	#
	# 		for j in range(len(la_params)):
	# 			if (p.n_estimators, p.max_features) == la_params[j]:
	# 				if min_la[j] > err:
	# 					min_la[j] = err
	# 					min_la_pi[j] = p
	#
	# 	## Test results for all test samples
	# 	hyper = best_pi.kwargs
	# 	g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 									  data_loc=data_home, type1='random1',
	# 									  fe=best_pi.feature_extraction,
	# 									  dr=best_pi.dimensionality_reduction,
	# 									  la=best_pi.learning_algorithm,
	# 									  val_splits=3, test_size=0.2)
	# 	g.run()
	# 	min_err_test = g.get_error()
	#
	# 	min_fe_test = [0] * len(fe_params)
	# 	for j in range(len(min_fe_pi)):
	# 		hyper = min_fe_pi[j].kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=min_fe_pi[j].feature_extraction,
	# 										  dr=min_fe_pi[j].dimensionality_reduction,
	# 										  la=min_fe_pi[j].learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_fe_test[j] = g.get_error()
	#
	# 	min_dr_test = [0] * len(dr_params)
	# 	for j in range(len(min_dr_pi)):
	# 		hyper = min_dr_pi[j].kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=min_dr_pi[j].feature_extraction,
	# 										  dr=min_dr_pi[j].dimensionality_reduction,
	# 										  la=min_dr_pi[j].learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_dr_test[j] = g.get_error()
	#
	# 	min_la_test = [0] * len(la_params)
	# 	for j in range(len(min_la_pi)):
	# 		hyper = min_la_pi[j].kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=min_la_pi[j].feature_extraction,
	# 										  dr=min_la_pi[j].dimensionality_reduction,
	# 										  la=min_la_pi[j].learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_la_test[j] = g.get_error()
	# 	errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
	# 	errors = np.asarray(errors)
	# 	alg_error[run - 1, :] = errors
	# 	errors = [np.mean(min_fe_test) - min_err_test, np.mean(min_dr_test) - min_err_test, np.mean(min_la_test) - min_err_test]
	# 	errors = np.asarray(errors)
	# 	alg_error_test[run - 1, :] = errors
	# std_error = np.std(alg_error, 0)
	# step_error = np.mean(alg_error, 0)
	# random_alg_error = step_error.astype('float32')
	# random_alg_std_error = std_error.astype('float32')
	# std_error = np.std(alg_error_test, 0)
	# step_error = np.mean(alg_error_test, 0)
	# random_alg_error_test = step_error.astype('float32')
	# random_alg_std_error_test = std_error.astype('float32')
	# test_results = [random_alg_error_test, random_alg_std_error_test]
	# val_results = [random_alg_error, random_alg_std_error]
	# pickle.dump([val_results, test_results], open(results_home + 'intermediate/' + data_name + '_test_error_contribution_algorithms.pkl', 'wb'))

	val_results, test_results = pickle.load(open(results_home + 'intermediate/' + data_name + '_test_error_contribution_algorithms.pkl', 'rb'))
	random_alg_error_test, random_alg_std_error_test = test_results
	random_alg_error, random_alg_std_error = val_results

	import matplotlib.pyplot as plt

	x = pipeline['all']

	_, axs = plt.subplots(nrows=1, ncols=1)
	x1 = [1, 2, 3]
	axs.errorbar(x1, random_alg_error, random_alg_std_error, linestyle='None', marker='^', capsize=3, color='b', label='validation')
	axs.plot(x1, random_alg_error, color='b')
	axs.errorbar(x1, random_alg_error_test, random_alg_std_error_test, linestyle='None', marker='^', capsize=3, color='k', label='testing')
	axs.plot(x1, random_alg_error_test, color='k')
	labels = []
	cnt = 1
	for item in axs.get_xticklabels():
		if len(item.get_text()) == 0:
			labels.append('')
		elif int(float(item.get_text())) == cnt:
			labels.append(x[cnt-1])
			cnt += 1
		else:
			labels.append('')

	box = axs.get_position()
	axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
	axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
	axs.axhline(y=0)
	axs.set_title('Algorithm error contributions')
	axs.set_xlabel('Agnostic algorithm')
	axs.set_ylabel('Error contributions')
	axs.set_xticklabels(labels)
	plt.savefig(results_home + 'figures/agnostic_error_alg_' + data_name + '_test.jpg')
	plt.close()





