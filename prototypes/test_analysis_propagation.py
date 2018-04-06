'''
This file is for quantification of error contributions from
different computational algorithms of an ML pipeline.
'''
import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'][:-1] + pipeline['dimensionality_reduction'][:-1] + pipeline[
																										'learning_algorithm'][
																									:-1]
lenp = len(pipeline['all'])

# Random1
start = 1
stop = 2
type1 = 'random_MCMC'

E1_val = np.zeros((stop - 1, lenp))
E2_val = np.zeros((stop - 1, lenp))
E3_val = np.zeros((stop - 1, lenp))
E1_test = np.zeros((stop - 1, lenp))
E2_test = np.zeros((stop - 1, lenp))
E3_test = np.zeros((stop - 1, lenp))
all_paths = []
for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')
	pipelines = obj.pipelines

	opt_opt_val = [1000000] * lenp
	naive_naive_val = [1000000] * lenp
	naive_opt_val = [100000] * lenp
	opt_naive_val = [100000] * lenp
	agnostic_opt_val = [100000] * lenp
	agnostic_naive_val = [100000] * lenp

	opt_opt_paths = [None] * lenp
	naive_naive_paths = [None] * lenp
	naive_opt_paths = [None] * lenp
	opt_naive_paths = [None] * lenp
	agnostic_opt_paths = [None] * lenp
	agnostic_naive_paths = [None] * lenp

	opt_opt_test = [1000000] * lenp
	naive_naive_test = [1000000] * lenp
	naive_opt_test = [100000] * lenp
	opt_naive_test = [100000] * lenp
	agnostic_opt_test = [100000] * lenp
	agnostic_naive_test = [100000] * lenp

	min_all = {}
	min_all_naive = {}
	for p in pipeline['all']:
		min_all[p] = []
		min_all_naive[p] = []

	path_errors = []
	for i in range(len(pipelines)):
		pi = pipelines[i]
		fe = pi.feature_extraction.decode('latin1')
		dr = pi.dimensionality_reduction.decode('latin1')
		la = pi.learning_algorithm.decode('latin1')
		path = (fe, dr, la)
		err = pi.get_error()
		path_errors.append((err, path, pi))
		all_paths.append(path)

	for err, path, pi in path_errors:
		# NAIVE -> NAIVE
		# For feature extraction algorithms
		if 'naive' in path[0] and 'naive' in path[1] and 'naive' in path[2]:
			if err < naive_naive_val[0]:
				naive_naive_val[0] = err
				naive_naive_val[1] = err
				naive_naive_val[2] = err
				naive_naive_paths[0] = pi
				naive_naive_paths[1] = pi
				naive_naive_paths[2] = pi

		# For dimensionality reduction algorithms
		if 'naive' not in path[0] and 'naive' in path[1] and 'naive' in path[2]:
			if err < naive_naive_val[3]:
				naive_naive_val[3] = err
				naive_naive_val[4] = err
				naive_naive_paths[3] = pi
				naive_naive_paths[4] = pi

		# For learning algorithms
		if 'naive' not in path[0] and 'naive' not in path[1] and 'naive' in path[2]:
			if err < naive_naive_val[5]:
				naive_naive_val[5] = err
				naive_naive_val[6] = err
				naive_naive_paths[5] = pi
				naive_naive_paths[6] = pi

		# NAIVE -> OPT
		# For feature extraction algorithms
		if 'naive' in path[0]:
			if 'naive' not in path[1] and 'naive' not in path[2]:
				if err < naive_opt_val[0]:
					naive_opt_val[0] = err
					naive_opt_val[1] = err
					naive_opt_val[2] = err
					naive_opt_paths[0] = pi
					naive_opt_paths[1] = pi
					naive_opt_paths[2] = pi

		# For dimensionality reduction algorithms
		if 'naive' in path[1]:
			if 'naive' not in path[0] and 'naive' not in path[2]:
				if err < naive_opt_val[3]:
					naive_opt_val[3] = err
					naive_opt_val[4] = err
					naive_opt_paths[3] = pi
					naive_opt_paths[4] = pi

		# For learning algorithms
		if 'naive' in path[2]:
			if 'naive' not in path[0] and 'naive' not in path[1]:
				if err < naive_opt_val[5]:
					naive_opt_val[5] = err
					naive_opt_val[6] = err
					naive_opt_paths[5] = pi
					naive_opt_paths[6] = pi

		# OPT -> NAIVE
		# For feature extraction algorithms
		for i, alg in enumerate(pipeline['all'][:3]):
			if path[0] == alg and 'naive' in path[1] and 'naive' in path[2]:
				if err < opt_naive_val[i]:
					opt_naive_val[i] = err
					opt_naive_paths[i] = pi

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][3:5]):
			if path[1] == alg and 'naive' not in path[0] and 'naive' in path[2]:
				if err < opt_naive_val[i + 3]:
					opt_naive_val[i + 3] = err
					opt_naive_paths[i + 3] = pi

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][5:]):
			if path[2] == alg and 'naive' not in path[0] and 'naive' not in path[1]:
				if err < opt_naive_val[i + 5]:
					opt_naive_val[i + 5] = err
					opt_naive_paths[i + 5] = pi

		# OPT -> OPT
		# For feature extraction algorithms
		for i, alg in enumerate(pipeline['all'][:3]):
			if path[0] == alg and 'naive' not in path[1] and 'naive' not in path[2]:
				if err < opt_opt_val[i]:
					opt_opt_val[i] = err
					opt_opt_paths[i] = err

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][3:5]):
			if path[1] == alg and 'naive' not in path[0] and 'naive' not in path[2]:
				if err < opt_opt_val[i + 3]:
					opt_opt_val[i + 3] = err
					opt_opt_paths[i + 3] = err

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][5:]):
			if path[2] == alg and 'naive' not in path[0] and 'naive' not in path[1]:
				if err < opt_opt_val[i + 5]:
					opt_opt_val[i + 5] = err
					opt_opt_paths[i + 5] = err

		# AGNOSTIC -> NAIVE
		# For feature extraction algorithms
		for i, alg in enumerate(pipeline['all'][:3]):
			if path[0] == alg and 'naive' in path[1] and 'naive' in path[2]:
				min_all_naive[alg].append((err, path, pi))

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][3:5]):
			if path[1] == alg and 'naive' not in path[0] and 'naive' in path[2]:
				min_all_naive[alg].append((err, path, pi))

		# For learning algorithms
		for i, alg in enumerate(pipeline['all'][5:]):
			if path[2] == alg and 'naive' not in path[0] and 'naive' not in path[1]:
				min_all_naive[alg].append((err, path, pi))

		# AGNOSTIC -> OPT
		# For feature extraction algorithms
		for i, alg in enumerate(pipeline['all'][:3]):
			if path[0] == alg and 'naive' not in path[1] and 'naive' not in path[2]:
				min_all[alg].append((err, path, pi))

		# For dimensionality reduction algorithms
		for i, alg in enumerate(pipeline['all'][3:5]):
			if path[1] == alg and 'naive' not in path[0] and 'naive' not in path[2]:
				min_all[alg].append((err, path, pi))

		# For learning algorithms
		for i, alg in enumerate(pipeline['all'][5:]):
			if path[2] == alg and 'naive' not in path[0] and 'naive' not in path[1]:
				min_all[alg].append((err, path, pi))

	# AGNOSTIC_OPT
	for i, alg in enumerate(pipeline['all']):
		paths = []
		for err, path, pi in min_all[alg]:
			paths.append(path)
		paths = list(set(paths))
		errs = [100000] * len(paths)
		pis = [None] * len(paths)
		for err, path, pi in min_all[alg]:
			for j, p in enumerate(paths):
				if p == path:
					if err < errs[j]:
						errs[j] = err
						pis[j] = pi
		agnostic_opt_val[i] = np.mean(errs)
		errs1 = []
		for pi1 in pis:
			hyper = pi1.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi1.feature_extraction.decode('latin1'),
											  dr=pi1.dimensionality_reduction.decode('latin1'),
											  la=pi1.learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			errs1.append(g.get_error())
		agnostic_opt_test[i] = np.mean(errs1)


	# AGNOSTIC_NAIVE
	for i, alg in enumerate(pipeline['all']):
		paths = []
		for err, path, pi in min_all_naive[alg]:
			paths.append(path)
		paths = list(set(paths))
		errs = [100000] * len(paths)
		pis = [None] * len(paths)
		for err, path, pi in min_all_naive[alg]:
			for j, p in enumerate(paths):
				if p == path:
					if err < errs[j]:
						errs[j] = err
						pis[j] = pi
		agnostic_naive_val[i] = np.mean(errs)
		errs1 = []
		for pi1 in pis:
			hyper = pi1.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi1.feature_extraction.decode('latin1'),
											  dr=pi1.dimensionality_reduction.decode('latin1'),
											  la=pi1.learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			errs1.append(g.get_error())
		agnostic_naive_test[i] = np.mean(errs1)

	for i, pi1 in enumerate(opt_opt_paths):
		hyper = pi1.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi1.feature_extraction.decode('latin1'),
										  dr=pi1.dimensionality_reduction.decode('latin1'),
										  la=pi1.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		opt_opt_test[i] = g.get_error()

	for i, pi1 in enumerate(opt_naive_paths):
		hyper = pi1.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi1.feature_extraction.decode('latin1'),
										  dr=pi1.dimensionality_reduction.decode('latin1'),
										  la=pi1.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		opt_naive_test[i] = g.get_error()

	for i, pi1 in enumerate(naive_opt_paths):
		hyper = pi1.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi1.feature_extraction.decode('latin1'),
										  dr=pi1.dimensionality_reduction.decode('latin1'),
										  la=pi1.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		naive_opt_test[i] = g.get_error()

	for i, pi1 in enumerate(naive_naive_paths):
		hyper = pi1.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi1.feature_extraction.decode('latin1'),
										  dr=pi1.dimensionality_reduction.decode('latin1'),
										  la=pi1.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		naive_naive_test[i] = g.get_error()


	E1_val[run - 1, :] = np.asarray(agnostic_opt_val) - np.asarray(opt_opt_val)
	E2_val[run - 1, :] = np.asarray(agnostic_naive_val) - np.asarray(opt_naive_val)
	E3_val[run - 1, :] = np.asarray(naive_naive_val) - np.asarray(naive_opt_val)
	E1_test[run - 1, :] = np.asarray(agnostic_opt_val) - np.asarray(opt_opt_test)
	E2_test[run - 1, :] = np.asarray(agnostic_naive_test) - np.asarray(opt_naive_test)
	E3_test[run - 1, :] = np.asarray(naive_naive_test) - np.asarray(naive_opt_test)

E1 = E1_test
E2 = E2_test
E3 = E3_test

alpha = np.zeros(E1.shape)
gamma = np.zeros(E1.shape)
E_propagation = np.zeros(E1.shape)
for i in range(lenp):
	E11 = E1[:, i]
	E22 = E2[:, i]
	E33 = E3[:, i]
	for j in range(len(E11)):
		e1 = E11[j]
		e2 = E22[j]
		e3 = E33[j]
		if e3 == 0:
			g = 0
			a = e1
		else:
			a = e1 * e3 / (e2 + e3 - e1)
			g = (e2 - e1) / e3
		alpha[j, i] = a
		gamma[j, i] = g
		E_propagation[j, i] = a * g

E_total = []
E_total_std = []
E_propagation_mean = []
E_propagation_std = []
E_direct = []
E_direct_std = []
gamma_mean = []
gamma_std = []
for i in range(lenp):
	a = alpha[:, i]
	a1 = []
	for j in range(len(a)):
		if not np.isnan(a[j]) and not np.isinf(a[j]):
			a1.append(a[j])
	E_direct.append(np.mean(a1))
	E_direct_std.append(np.std(a1))

	a = E1[:, i]
	a1 = []
	for j in range(len(a)):
		if not np.isnan(a[j]) and not np.isinf(a[j]):
			a1.append(a[j])
	E_total.append(np.mean(a1))
	E_total_std.append(np.std(a1))

	a = E_propagation[:, i]
	a1 = []
	for j in range(len(a)):
		if not np.isnan(a[j]) and not np.isinf(a[j]):
			a1.append(a[j])
	E_propagation_mean.append(np.mean(a1))
	E_propagation_std.append(np.std(a1))

	a = gamma[:, i]
	a1 = []
	for j in range(len(a)):
		if not np.isnan(a[j]) and not np.isinf(a[j]):
			a1.append(a[j])
	gamma_mean.append(np.mean(a1))
	gamma_std.append(np.std(a1))

from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = pipeline['all']
x1 = np.asarray([1, 2, 3, 4, 5, 6, 7])
w = 0.2
d = w * np.ones(7)
x2 = x1 + d
plt.figure()
plt.bar(x1, E_total, width=w, align='center', color=colors[0], yerr=E_total_std, label='Total error')
plt.bar(x2, E_direct, width=w, align='center', color=colors[1], yerr=E_direct_std, label='Direct error')
plt.bar(x2, E_propagation_mean, width=w, bottom=E_direct, align='center', color=colors[2], yerr=E_propagation_std,
		label='Propagation error')
plt.title('Error quantification and propagation in algorithms for ' + data_name + ' data')
plt.xlabel('Algorithms')
plt.ylabel('Error contributions')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_random_pipeline_algorithms_' + data_name + '.jpg')
plt.close()

plt.figure()
plt.bar(x1, gamma_mean, width=w, color='r', yerr=gamma_std)
plt.title('Propagation factor in algorithms for ' + data_name + ' data')
plt.xlabel('Steps')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_algorithms_' + data_name + '.jpg')
plt.close()


