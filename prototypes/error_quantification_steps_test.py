'''
This file is for getting error contributions on test data samples.
'''
import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']
lenp = len(pipeline['all'])
datasets = ['breast', 'brain', 'matsc_dataset1', 'matsc_dataset2']

# Random1
type1 = 'random_MCMC'
start = 1
stop = 6

for data_name in datasets:
	step_error_val = np.zeros((stop - 1, 3))
	alg_error_val = np.zeros((stop - 1, 7))
	alg1_error_val = np.zeros((stop - 1, 7))
	step_error_test = np.zeros((stop - 1, 3))
	alg_error_test = np.zeros((stop - 1, 7))
	alg1_error_test = np.zeros((stop - 1, 7))

	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		min_err_val = 100000000
		min_fe_val = [1000000] * len(pipeline['feature_extraction'])
		min_fe_pi = [None] * len(pipeline['feature_extraction'])
		min_dr_val = [1000000] * len(pipeline['dimensionality_reduction'])
		min_dr_pi = [None] * len(pipeline['dimensionality_reduction'])
		min_la_val = [1000000] * len(pipeline['learning_algorithm'])
		min_la_pi = [None] * len(pipeline['learning_algorithm'])

		min_all_val = [100000] * len(pipeline['all'])
		min_all_pi = [None] * len(pipeline['all'])
		min_all1_val = [100000] * len(pipeline['all'])
		min_all1_pi = [None] * len(pipeline['all'])
		min_alls_val = [0] * len(pipeline['all'])
		min_alls_test = [0] * len(pipeline['all'])
		min_alls_pi = [None] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])
		pipelines = obj.pipelines
		best_pi = None
		for i in range(len(pipelines)):
			pi = pipelines[i]
			pie = pi.get_error()
			if pie < min_err_val:
				min_err_val = pie
				best_pi = pi

			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if pi.feature_extraction.decode('latin1') == alg:
					if pie < min_fe_val[j]:
						min_fe_val[j] = pie
						min_fe_pi[j] = pi

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if pi.dimensionality_reduction.decode('latin1') == alg:
					if pie < min_dr_val[j]:
						min_dr_val[j] = pie
						min_dr_pi[j] = pi

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if pi.learning_algorithm.decode('latin1') == alg:
					if pie < min_la_val[j]:
						min_la_val[j] = pie
						min_la_pi[j] = pi

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if pi.learning_algorithm.decode('latin1') == alg or pi.feature_extraction.decode(
						'latin1') == alg or pi.dimensionality_reduction.decode('latin1') == alg:
					if pie < min_all1_val[j]:
						min_all1_val[j] = pie
						min_all1_pi[j] = pi

					min_alls_val[j] += pie
					hyper = pi.kwargs
					g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
													  data_loc=data_home, type1='random1',
													  fe=pi.feature_extraction.decode('latin1'),
													  dr=pi.dimensionality_reduction.decode('latin1'),
													  la=pi.learning_algorithm.decode('latin1'),
													  val_splits=3, test_size=0.2)
					g.run()
					min_alls_test[j] += g.get_error()
					s[j] += 1
				else:
					if pie < min_all_val[j]:
						min_all_val[j] = pie
						min_all_pi[j] = pi

		## Test results for all test samples
		hyper = best_pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=best_pi.feature_extraction.decode('latin1'),
										  dr=best_pi.dimensionality_reduction.decode('latin1'),
										  la=best_pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_err_test = g.get_error()

		min_fe_test = [0] * len(pipeline['feature_extraction'])
		for j in range(len(min_fe_pi)):
			hyper = min_fe_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_fe_pi[j].feature_extraction.decode('latin1'),
											  dr=min_fe_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_fe_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_fe_test[j] = g.get_error()

		min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
		for j in range(len(min_dr_pi)):
			hyper = min_dr_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_dr_pi[j].feature_extraction.decode('latin1'),
											  dr=min_dr_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_dr_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_dr_test[j] = g.get_error()

		min_la_test = [0] * len(pipeline['learning_algorithm'])
		for j in range(len(min_la_pi)):
			hyper = min_la_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_la_pi[j].feature_extraction.decode('latin1'),
											  dr=min_la_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_la_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_la_test[j] = g.get_error()


		min_all_test = [0] * len(pipeline['all'])
		min_all1_test = [0] * len(pipeline['all'])
		for j in range(len(min_all_pi)):
			hyper = min_all_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_all_pi[j].feature_extraction.decode('latin1'),
											  dr=min_all_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_all_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_all_test[j] = g.get_error()


			hyper = min_all1_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_all1_pi[j].feature_extraction.decode('latin1'),
											  dr=min_all1_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_all1_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_all1_test[j] = g.get_error()

		for j in range(len(s)):
			if s[j] == 0:
				min_alls_test[j] = min_all1_val[j]
				min_alls_val[j] = min_all1_test[j]
			else:
				min_alls_val[j] /= s[j]
				min_alls_test[j] /= s[j]
		errors = [np.mean(min_fe_val) - min_err_val, np.mean(min_dr_val) - min_err_val, np.mean(min_la_val) - min_err_val]
		errors = np.asarray(errors)
		step_error_val[run - 1, :] = errors
		min_all_val = np.asarray(min_all_val)
		min_all1_val = np.asarray(min_all1_val)
		min_alls_val = np.asarray(min_alls_val)
		alg_error_val[run - 1, :] = min_all_val - np.asarray([min_err_val] * 7)
		alg1_error_val[run - 1, :] = min_alls_val - min_all1_val

		errors = [np.mean(min_fe_test) - min_err_test, np.mean(min_dr_test) - min_err_test,
				  np.mean(min_la_test) - min_err_test]
		errors = np.asarray(errors)
		step_error_test[run - 1, :] = errors
		min_all_test = np.asarray(min_all_test)
		min_all1_test = np.asarray(min_all1_test)
		min_alls_test = np.asarray(min_alls_test)
		alg_error_test[run - 1, :] = min_all_test - np.asarray([min_err_test] * 7)
		alg1_error_test[run - 1, :] = min_alls_test - min_all1_test

	std_error = np.std(step_error_val, 0)
	step_error = np.mean(step_error_val, 0)
	random1_step_error_val = step_error.astype('float32')
	random1_std_error_val = std_error.astype('float32')
	std_error = np.std(alg_error_val, 0)
	step_error = np.mean(alg_error_val, 0)
	random1_alg_error_val = step_error.astype('float32')
	random1_alg_std_error_val = std_error.astype('float32')
	std_error = np.std(alg1_error_val, 0)
	step_error = np.mean(alg1_error_val, 0)
	random1_alg1_error_val = step_error.astype('float32')
	random1_alg1_std_error_val = std_error.astype('float32')

	std_error = np.std(step_error_test, 0)
	step_error = np.mean(step_error_test, 0)
	random1_step_error_test = step_error.astype('float32')
	random1_std_error_test = std_error.astype('float32')
	std_error = np.std(alg_error_test, 0)
	step_error = np.mean(alg_error_test, 0)
	random1_alg_error_test = step_error.astype('float32')
	random1_alg_std_error_test = std_error.astype('float32')
	std_error = np.std(alg1_error_test, 0)
	step_error = np.mean(alg1_error_test, 0)
	random1_alg1_error_test = step_error.astype('float32')
	random1_alg1_std_error_test = std_error.astype('float32')

	test_results = [random1_step_error_test, random1_std_error_test, random1_alg_error_test, random1_alg_std_error_test, random1_alg1_error_test, random1_alg1_std_error_test]
	val_results = [random1_step_error_val, random1_std_error_val, random1_alg_error_val, random1_alg_std_error_val, random1_alg1_error_val, random1_alg1_std_error_val]
	pickle.dump([val_results, test_results], open(results_home + 'intermediate/' + data_name + '_test_error_contribution.pkl', 'wb'))

	# val_results, test_results = pickle.load(open(results_home + 'intermediate/' + data_name + '_test_error_contribution.pkl', 'rb'))
	random1_step_error_test, random1_std_error_test, random1_alg_error_test, random1_alg_std_error_test, random1_alg1_error_test, random1_alg1_std_error_test = test_results
	random1_step_error_val, random1_std_error_val, random1_alg_error_val, random1_alg_std_error_val, random1_alg1_error_val, random1_alg1_std_error_val = val_results

	import matplotlib.pyplot as plt
	x = ['Feature extraction', 'Feature transformation', 'Learning algorithm']

	x1 = [1, 2, 3]
	_, axs = plt.subplots(nrows=1, ncols=1)
	axs.errorbar(x1, random1_step_error_val, random1_std_error_val, linestyle='None', marker='^', capsize=3, color='b', label='Validation')
	axs.plot(x1, random1_step_error_val, color='b')
	axs.errorbar(x1, random1_step_error_test, random1_std_error_test, linestyle='None', marker='^', capsize=3, color='k', label='Testing')
	axs.plot(x1, random1_step_error_test, color='k')
	box = axs.get_position()
	axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
	axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
	axs.axhline(y=0)
	axs.set_title('Step error contributions')
	axs.set_xlabel('Agnostic step')
	axs.set_ylabel('Error contributions')
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

	axs.set_xticklabels(labels)
	plt.savefig(results_home + 'figures/agnostic_error_step_val_test_' + data_name + '.jpg')
	plt.close()

	x = pipeline['all']
	_, axs = plt.subplots(nrows=1, ncols=1)
	x1 = [1, 2, 3, 4, 5, 6, 7]
	axs.errorbar(x1, random1_alg_error_val, random1_alg_std_error_val, linestyle='None', marker='^', capsize=3, color='b', label='Validation')
	axs.plot(x1, random1_alg_error_val, color='b')
	axs.errorbar(x1, random1_alg_error_test, random1_alg_std_error_test, linestyle='None', marker='^', capsize=3, color='k', label='Testing')
	axs.plot(x1, random1_alg_error_test, color='k')
	box = axs.get_position()
	axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
	axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
	axs.axhline(y=0)
	axs.set_title('Algorithm error contributions (method 1)')
	axs.set_xlabel('Agnostic algorithm')
	axs.set_ylabel('Error contributions')
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

	axs.set_xticklabels(labels)
	plt.savefig(results_home + 'figures/agnostic_error_alg_1_val_test_' + data_name + '.jpg')
	plt.close()

	_, axs = plt.subplots(nrows=1, ncols=1)
	x1 = [1, 2, 3, 4, 5, 6, 7]
	axs.errorbar(x1, random1_alg1_error_val, random1_alg1_std_error_val, linestyle='None', marker='^', capsize=3, color='b', label='Validaton')
	axs.plot(x1, random1_alg1_error_val, color='b')
	axs.errorbar(x1, random1_alg1_error_test, random1_alg1_std_error_test, linestyle='None', marker='^', capsize=3, color='k', label='Testing')
	axs.plot(x1, random1_alg1_error_test, color='k')
	box = axs.get_position()
	axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
	axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
	axs.axhline(y=0)
	axs.set_title('Algorithm error contributions (method 2)')
	axs.set_xlabel('Agnostic algorithm')
	axs.set_ylabel('Error contributions')
	axs.set_xticklabels(labels)
	plt.savefig(results_home + 'figures/agnostic_error_alg_2_val_test_' + data_name + '.jpg')
	plt.close()
