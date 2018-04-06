'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
data_home = home + '/' + sys.argv[1] + '/EP_project/data/'
results_home = home + '/' + sys.argv[1] + '/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]

datasets = ['breast', 'brain', 'matsc_dataset1', 'matsc_dataset2']

for data_name in datasets:
	start = 1
	stop = 6
	if data_name == 'breast':
		stop = 2
	if data_name == 'brain':
		stop = 4
	type1 = 'random_MCMC'
	# E1 = np.zeros((stop - 1, 3))
	# E2 = np.zeros((stop - 1, 3))
	# E3 = np.zeros((stop - 1, 3))
	#
	# E1_test = np.zeros((stop - 1, 3))
	# E2_test = np.zeros((stop - 1, 3))
	# E3_test = np.zeros((stop - 1, 3))
	#
	# for run in range(start, stop):
	# 	obj = pickle.load(
	# 		open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
	# 			 str(run) + '_full_naive.pkl', 'rb'))
	#
	# 	pipelines = obj.pipelines
	# 	opt_opt = [1000000] * 3
	# 	naive_naive = [1000000] * 3
	# 	naive_opt = [100000] * 3
	# 	opt_naive = [100000] * 3
	# 	agnostic_naive = [0] * 3
	# 	agnostic_opt = [0] * 3
	#
	# 	opt_opt_paths = [None] * 3
	# 	naive_naive_paths = [None] * 3
	# 	naive_opt_paths = [None] * 3
	# 	opt_naive_paths = [None] * 3
	# 	agnostic_naive_paths = [None] * 3
	# 	agnostic_opt_paths = [None] * 3
	#
	# 	opt_opt_test = [1000000] * 3
	# 	naive_naive_test = [1000000] * 3
	# 	naive_opt_test = [100000] * 3
	# 	opt_naive_test = [100000] * 3
	# 	agnostic_naive_test = [0] * 3
	# 	agnostic_opt_test = [0] * 3
	#
	# 	min_fe = {}
	# 	min_fe_naive = {}
	# 	min_fe_test = {}
	# 	min_fe_naive_test = {}
	# 	for p in pipeline['feature_extraction']:
	# 		if 'naive' not in p:
	# 			min_fe[p] = 1000000
	# 			min_fe_naive[p] = 1000000
	# 			min_fe_test[p] = None
	# 			min_fe_naive_test[p] = None
	# 	min_dr = {}
	# 	min_dr_naive = {}
	# 	min_dr_test = {}
	# 	min_dr_naive_test = {}
	# 	for p in pipeline['dimensionality_reduction']:
	# 		if 'naive' not in p:
	# 			min_dr[p] = 1000000
	# 			min_dr_naive[p] = 1000000
	# 			min_dr_test[p] = None
	# 			min_dr_naive_test[p] = None
	# 	min_la = {}
	# 	min_la_naive = {}
	# 	min_la_test = {}
	# 	min_la_naive_test = {}
	# 	for p in pipeline['learning_algorithm']:
	# 		if 'naive' not in p:
	# 			min_la[p] = 1000000
	# 			min_la_naive[p] = 1000000
	# 			min_la_test[p] = None
	# 			min_la_naive_test[p] = None
	#
	# 	for i in range(len(pipelines)):
	# 		pi = pipelines[i]
	# 		fe = pi.feature_extraction
	# 		dr = pi.dimensionality_reduction
	# 		la = pi.learning_algorithm
	# 		path = (fe, dr, la)
	# 		err = pi.get_error()
	#
	# 		#  Step 1: Feature extraction
	# 		p = pipeline['feature_extraction']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if alg == fe:
	# 				if 'naive' in alg:  # naive ->
	# 					if 'naive' in dr and 'naive' in la:
	# 						if err < naive_naive[0]:  # naive -> naive
	# 							naive_naive[0] = err
	# 							naive_naive_paths[0] = pi
	# 					if 'naive' not in dr and 'naive' not in la:  # naive -> opt
	# 						if err < naive_opt[0]:
	# 							naive_opt[0] = err
	# 							naive_opt_paths[0] = pi
	# 				else:  # opt -> and agnostic ->
	# 					if 'naive' in dr and 'naive' in la:
	# 						if err < opt_naive[0]:  # opt -> naive
	# 							opt_naive[0] = err
	# 							opt_naive_paths[0] = pi
	# 						if err < min_fe_naive[alg]:
	# 							min_fe_naive[alg] = err  # agnostic -> naive1
	# 							min_fe_naive_test[alg] = pi
	# 					if 'naive' not in dr and 'naive' not in la:
	# 						if err < opt_opt[0]:
	# 							opt_opt[0] = err
	# 							opt_opt_paths[0] = pi
	# 						if err < min_fe[alg]:  # agnostic -> opt
	# 							min_fe[alg] = err
	# 							min_fe_test[alg] = pi
	#
	# 		# Step 2: Dimensionality reduction
	# 		p = pipeline['dimensionality_reduction']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if alg == dr:
	# 				if 'naive' in alg:  # naive ->
	# 					if 'naive' not in fe and 'naive' in la:
	# 						if err < naive_naive[1]:  # naive -> naive
	# 							naive_naive[1] = err
	# 							naive_naive_paths[1] = pi
	# 					if 'naive' not in fe and 'naive' not in la:  # naive -> opt
	# 						if err < naive_opt[1]:
	# 							naive_opt[1] = err
	# 							naive_opt_paths[1] = pi
	# 				else:  # opt -> and agnostic ->
	# 					if 'naive' not in fe and 'naive' in la:
	# 						if err < opt_naive[1]:  # opt -> naive
	# 							opt_naive[1] = err
	# 							opt_naive_paths[1] = pi
	# 						if err < min_dr_naive[alg]:  # agnostic -> naive
	# 							min_dr_naive[alg] = err
	# 							min_dr_naive_test[alg] = pi
	# 					if 'naive' not in fe and 'naive' not in la:
	# 						if err < opt_opt[1]:
	# 							opt_opt[1] = err
	# 							opt_opt_paths[1] = pi
	# 						if err < min_dr[alg]:  # agnostic -> opt
	# 							min_dr[alg] = err
	# 							min_dr_test[alg] = pi
	#
	# 		# Step 2: Learning algorithm
	# 		p = pipeline['learning_algorithm']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if alg == la:
	# 				if 'naive' in alg:  # naive ->
	# 					if 'naive' not in fe and 'naive' not in dr:
	# 						if err < naive_naive[2]:  # naive -> naive
	# 							naive_naive[2] = err
	# 							naive_naive_paths[2] = pi
	# 					if 'naive' not in fe and 'naive' not in dr:  # naive -> opt
	# 						if err < naive_opt[2]:
	# 							naive_opt[2] = err
	# 							naive_opt_paths[2] = pi
	# 				else:  # opt -> and agnostic ->
	# 					if 'naive' not in fe and 'naive' not in dr:
	# 						if err < opt_naive[2]:  # opt -> naive
	# 							opt_naive[2] = err
	# 							opt_naive_paths[2] = pi
	# 						if err < min_la_naive[alg]:  # agnostic -> naive
	# 							min_la_naive[alg] = err
	# 							min_la_naive_test[alg] = pi
	# 					if 'naive' not in fe and 'naive' not in dr:
	# 						if err < opt_opt[2]:
	# 							opt_opt[2] = err
	# 							opt_opt_paths[2] = pi
	# 						if err < min_la[alg]:  # agnostic -> opt
	# 							min_la[alg] = err
	# 							min_la_test[alg] = pi
	#
	# 	s = 0
	# 	s_naive = 0
	# 	for k in min_fe.keys():
	# 		if min_fe[k] < 100000:
	# 			agnostic_opt[0] += min_fe[k]
	# 			pi = min_fe_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_opt_test[0] += g.get_error()
	# 			s += 1
	# 		if min_fe_naive[k] < 100000:
	# 			agnostic_naive[0] += min_fe_naive[k]
	# 			pi = min_fe_naive_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_naive_test[0] += g.get_error()
	# 			s_naive += 1
	# 	agnostic_opt[0] /= s
	# 	agnostic_naive[0] /= s_naive
	# 	agnostic_opt_test[0] /= s
	# 	agnostic_naive_test[0] /= s_naive
	#
	# 	s = 0
	# 	s_naive = 0
	# 	for k in min_dr.keys():
	# 		if min_dr[k] < 100000:
	# 			agnostic_opt[1] += min_dr[k]
	# 			pi = min_dr_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_opt_test[1] += g.get_error()
	# 			s += 1
	# 		if min_dr_naive[k] < 100000:
	# 			agnostic_naive[1] += min_dr_naive[k]
	# 			pi = min_dr_naive_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_naive_test[1] += g.get_error()
	# 			s_naive += 1
	# 	agnostic_opt[1] /= s
	# 	agnostic_naive[1] /= s_naive
	# 	agnostic_opt_test[1] /= s
	# 	agnostic_naive_test[1] /= s_naive
	#
	# 	s = 0
	# 	s_naive = 0
	# 	for k in min_la.keys():
	# 		if min_la[k] < 100000:
	# 			agnostic_opt[2] += min_la[k]
	# 			pi = min_la_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_opt_test[2] += g.get_error()
	# 			s += 1
	# 		if min_la_naive[k] < 100000:
	# 			agnostic_naive[2] += min_la_naive[k]
	# 			pi = min_la_naive_test[k]
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			agnostic_naive_test[2] += g.get_error()
	# 			s_naive += 1
	# 	agnostic_opt[2] /= s
	# 	agnostic_naive[2] /= s_naive
	# 	agnostic_opt_test[2] /= s
	# 	agnostic_naive_test[2] /= s_naive
	#
	# 	for i in range(len(opt_opt)):
	# 		pi = opt_opt_paths[i]
	# 		hyper = pi.kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=pi.feature_extraction,
	# 										  dr=pi.dimensionality_reduction,
	# 										  la=pi.learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		opt_opt_test[i] = g.get_error()
	#
	# 	for i in range(len(opt_naive)):
	# 		pi = opt_naive_paths[i]
	# 		hyper = pi.kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=pi.feature_extraction,
	# 										  dr=pi.dimensionality_reduction,
	# 										  la=pi.learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		opt_naive_test[i] = g.get_error()
	#
	# 	for i in range(len(naive_opt)):
	# 		pi = naive_opt_paths[i]
	# 		hyper = pi.kwargs
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='random1',
	# 										  fe=pi.feature_extraction,
	# 										  dr=pi.dimensionality_reduction,
	# 										  la=pi.learning_algorithm,
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		naive_opt_test[i] = g.get_error()
	#
	# 	for i in range(len(naive_naive)):
	# 		pi = naive_naive_paths[i]
	# 		if pi is not None:
	# 			hyper = pi.kwargs
	# 			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 											  data_loc=data_home, type1='random1',
	# 											  fe=pi.feature_extraction,
	# 											  dr=pi.dimensionality_reduction,
	# 											  la=pi.learning_algorithm,
	# 											  val_splits=3, test_size=0.2)
	# 			g.run()
	# 			naive_naive_test[i] = g.get_error()
	# 		else:
	# 			naive_naive_test[i] = 0
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = agnostic_opt[i] - opt_opt[i]
	# 	E1[run - 1, :] = np.asarray(e)
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = agnostic_naive[i] - opt_naive[i]
	# 	E2[run - 1, :] = np.asarray(e)
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = naive_naive[i] - naive_opt[i]
	# 	E3[run - 1, :] = np.asarray(e)
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = agnostic_opt_test[i] - opt_opt_test[i]
	# 	E1_test[run - 1, :] = np.asarray(e)
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = agnostic_naive_test[i] - opt_naive_test[i]
	# 	E2_test[run - 1, :] = np.asarray(e)
	#
	# 	e = [0] * 3
	# 	for i in range(3):
	# 		e[i] = naive_naive_test[i] - naive_opt_test[i]
	# 	E3_test[run - 1, :] = np.asarray(e)
	#
	# test_results = [E1, E2, E3, E1_test, E2_test, E3_test]
	# pickle.dump(test_results, open(results_home + 'intermediate/' + data_name + '_test_error_propagation.pkl', 'wb'))
	test_results = pickle.load(open(results_home + 'intermediate_CCNI/' + data_name + '_test_error_propagation.pkl', 'rb'), encoding='latin1')
	_, _, _, E1_test, E2_test, E3_test = test_results

	E1 = E1_test
	E2 = E2_test
	E3 = E3_test

	# rows = []
	# for run in range(start, stop):
	# 	if 0 in E3[run-1, :]:
	# 		rows.append(run-1)
	#
	# E1 = np.delete(E1, rows, 0)
	# E2 = np.delete(E2, rows, 0)
	# E3 = np.delete(E3, rows, 0)


	alpha = np.zeros(E1.shape)
	gamma = np.zeros(E1.shape)
	E_propagation = np.zeros(E1.shape)
	for i in range(3):
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

	E_total = np.mean(E1, 0)
	E_total_std = np.std(E1, 0)

	E_direct = np.mean(alpha, axis=0)
	E_direct_std = np.std(alpha, axis=0)

	E_propagation_mean = np.mean(E_propagation, axis=0)
	E_propagation_std = np.std(E_propagation, axis=0)

	gamma_mean = np.mean(gamma, 0)
	gamma_std = np.std(gamma, 0)



	from matplotlib import pyplot as plt

	colors = ['b', 'g', 'y']
	steps = ['Feature extraction', 'Feature transformation', 'Learning algorithms']
	x1 = np.asarray([1, 2, 3])
	w = 0.2
	d = w * np.ones(3)
	x2 = x1 + d
	plt.bar(x1, E_total.ravel(), width=w, align='center', color=colors[0], yerr=E_total_std.ravel(), label='Total error')
	plt.bar(x2, E_direct.ravel(), width=w, align='center', color=colors[1], yerr=E_direct_std.ravel(), label='Direct error')
	plt.bar(x2, E_propagation_mean.ravel(), width=w, bottom=E_direct.ravel(), align='center', color=colors[2],
			yerr=E_propagation_std.ravel(), label='Propagation error')
	plt.title('Error quantification and propagation in steps for ' + data_name + ' data')
	plt.xlabel('Steps')
	plt.ylabel('Error contributions')
	plt.xticks(x1, steps)
	plt.legend()
	plt.autoscale()
	plt.savefig(results_home + 'figures/error_propagation_random_pipeline_steps_test_' + data_name + '.jpg')
	plt.close()

	x1 = np.asarray([1, 2, 3])
	plt.bar(x1, gamma_mean.ravel(), width=w, color='r', yerr=gamma_std.ravel())
	plt.title('Propagation factor in steps for ' + data_name + ' data')
	plt.xlabel('Steps')
	plt.ylabel('Propagation factor')
	plt.xticks(x1, steps)
	plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_steps_test_' + data_name + '.jpg')
	plt.close()
