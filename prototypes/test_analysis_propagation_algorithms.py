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
data_name = 'matsc_dataset2'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["haralick", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'][:-1] + pipeline['dimensionality_reduction'][:-1] + pipeline['learning_algorithm'][:-1]

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
type2 = 'random_MCMC'
E1 = np.zeros((stop - 1, 3))
E2 = np.zeros((stop - 1, 3))
E3 = np.zeros((stop - 1, 3))
E1_test = np.zeros((stop - 1, 3))
E2_test = np.zeros((stop - 1, 3))
E3_test = np.zeros((stop - 1, 3))


if data_name == 'breast':
	stop = 2
if data_name == 'brain':
	stop = 4

obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(2) + '_full.pkl','rb'), encoding='latin1')
pipelines = obj.pipelines
path_pipelines = []
for i in range(len(pipelines)):
	pi = pipelines[i]
	path = (pi.feature_extraction.decode('latin1'), pi.dimensionality_reduction.decode('latin1'), pi.learning_algorithm.decode('latin1'))
	if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
		path_pipelines.append(pipelines[i])

fe_params = set()
dr_params = set()
la_params = set()
for i in range(len(path_pipelines)):
	p = path_pipelines[i]
	fe_params.add(p.haralick_distance)
	dr_params.add(p.pca_whiten)
	la_params.add((p.n_estimators, p.max_features))

fe_params = list(fe_params)
dr_params = list(dr_params)
la_params = list(la_params)

cnt = 0
agnostic_naive_all = np.zeros((1, 3))
naive_naive_all = np.zeros((1, 3))
opt_naive_all = np.zeros((1, 3))
naive_opt_all = np.zeros((1, 3))
agnostic_naive_all_test = np.zeros((1, 3))
naive_naive_all_test = np.zeros((1, 3))
opt_naive_all_test = np.zeros((1, 3))
naive_opt_all_test = np.zeros((1, 3))

for run in range(2, 3):
	obj = pickle.load(
		open(results_home + 'intermediate/' + type2 + '/' + type2 + '_' + data_name + '_run_' +
			 str(run) + '_full_final_naive.pkl', 'rb'), encoding='latin1')
	pipelines = obj.pipelines
	paths = obj.paths
	path_pipelines = []
	for i in range(len(pipelines)):
		pi = pipelines[i]
		path = (pi.feature_extraction, pi.dimensionality_reduction, pi.learning_algorithm)
		if ('naive' in path[0] or path[0] == pipeline['feature_extraction'][0]) and \
				('naive' in path[1] or path[1] == pipeline['dimensionality_reduction'][0]) and \
				('naive' in path[2] or path[2] == pipeline['learning_algorithm'][0]):
			path_pipelines.append((path, pipelines[i]))


	naive_naive = [1000000] * 3
	naive_opt = [100000] * 3
	opt_naive = [100000] * 3
	agnostic_naive = [0] * 3
	naive_naive_paths = [None] * 3
	naive_opt_paths = [None] * 3
	opt_naive_paths = [None] * 3
	agnostic_naive_paths = [None] * 3
	naive_naive_test = [1000000] * 3
	naive_opt_test = [100000] * 3
	opt_naive_test = [100000] * 3
	agnostic_naive_test = [0] * 3

	min_fe_naive = {}
	min_fe_naive_paths = {}
	for p in range(len(fe_params)):
		min_fe_naive[p] = 1000000
		min_fe_naive_paths[p] = None

	min_dr_naive = {}
	min_dr_naive_paths = {}
	for p in range(len(dr_params)):
		min_dr_naive[p] = 1000000
		min_dr_naive_paths[p] = None

	min_la_naive = {}
	min_la_naive_paths = {}
	for p in range(len(la_params)):
		min_la_naive[p] = 1000000
		min_la_naive_paths[p] = None


	#  Step 1: Feature extraction
	for i in range(len(path_pipelines)):
		path = path_pipelines[i][0]
		pi = path_pipelines[i][1]
		fe = path[0]
		dr = path[1]
		la = path[2]
		# errs = []
		# for j in range(len(pi)):
		# 	errs.append(pi.get_error())
		err = pi.get_error()

		p = pipeline['feature_extraction']
		for j in range(len(p)):
			alg = p[j]
			if alg == fe:
				if 'naive' in alg:  # naive ->
					if 'naive' in dr and 'naive' in la:
						if err < naive_naive[0]:  # naive -> naive
							naive_naive[0] = err
							naive_naive_paths[0] = pi
					if 'naive' not in dr and 'naive' not in la:  # naive -> opt
						if err < naive_opt[0]:
							naive_opt[0] = err
							naive_opt_paths[0] = pi
				else:  # opt -> and agnostic ->
					if 'naive' in dr and 'naive' in la:
						if err < opt_naive[0]:  # opt -> naive
							opt_naive[0] = err
							opt_naive_paths[0] = pi
						fe_param = pi.haralick_distance
						fe_ind = fe_params.index(fe_param)
						if err < min_fe_naive[fe_ind]:
							min_fe_naive[fe_ind] = err  # agnostic -> naive
							min_fe_naive_paths[fe_ind] = pi

		# Step 2: Dimensionality reduction
		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if alg == dr:
				if 'naive' in alg:  # naive ->
					if 'naive' not in fe and 'naive' in la:
						if err < naive_naive[1]:  # naive -> naive
							naive_naive[1] = err
							naive_naive_paths[1] = pi
					if 'naive' not in fe and 'naive' not in la:  # naive -> opt
						if err < naive_opt[1]:
							naive_opt[1] = err
							naive_opt_paths[1] = pi
				else:  # opt -> and agnostic ->
					if 'naive' not in fe and 'naive' in la:
						if err < opt_naive[1]:  # opt -> naive
							opt_naive[1] = err
							opt_naive_paths[1] = pi
						dr_param = pi.pca_whiten
						dr_ind = dr_params.index(dr_param)
						if err < min_dr_naive[dr_ind]:  # agnostic -> naive
							min_dr_naive[dr_ind] = err
							min_dr_naive_paths[dr_ind] = pi


		# Step 2: Learning algorithm
		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if alg == la:
				if 'naive' in alg:  # naive ->
					if 'naive' not in fe and 'naive' not in dr:
						if err < naive_naive[2]:  # naive -> naive
							naive_naive[2] = err
							naive_naive_paths[2] = pi
					if 'naive' not in fe and 'naive' not in dr:  # naive -> opt
						if err < naive_opt[2]:
							naive_opt[2] = err
							naive_opt_paths[2] = pi
				else:  # opt -> and agnostic ->
					if 'naive' not in fe and 'naive' not in dr:
						if err < opt_naive[2]:  # opt -> naive
							opt_naive[2] = err
							opt_naive_paths[2] = pi
						la_param = (pi.n_estimators, pi.max_features)
						dists = []
						for k in range(len(la_params)):
							dists.append(np.linalg.norm(np.asarray(la_param) - np.asarray(la_params[k])))
						la_ind = np.argmin(dists)
						if err < min_la_naive[la_ind]:  # agnostic -> naive
							min_la_naive[la_ind] = err
							min_la_naive_paths[la_ind] = pi

	for i in range(len(opt_naive)):
		pi = opt_naive_paths[i]
		hyper = pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction,
										  dr=pi.dimensionality_reduction,
										  la=pi.learning_algorithm,
										  val_splits=3, test_size=0.2)
		g.run()
		opt_naive_test[i] = g.get_error()

	for i in range(len(naive_opt)):
		pi = naive_opt_paths[i]
		hyper = pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction,
										  dr=pi.dimensionality_reduction,
										  la=pi.learning_algorithm,
										  val_splits=3, test_size=0.2)
		g.run()
		naive_opt_test[i] = g.get_error()

	for i in range(len(naive_naive)):
		pi = naive_naive_paths[i]
		if pi is not None:
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			naive_naive_test[i] = g.get_error()
		else:
			naive_naive_test[i] = 0

	s_naive = 0
	for k in min_fe_naive.keys():
		if min_fe_naive[k] < 100000:
			agnostic_naive[0] += min_fe_naive[k]
			pi = min_fe_naive_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_naive_test[0] += g.get_error()
			s_naive += 1
	agnostic_naive[0] /= s_naive
	agnostic_naive_test[0] /= s_naive

	s_naive = 0
	for k in min_dr_naive.keys():
		if min_dr_naive[k] < 100000:
			agnostic_naive[1] += min_dr_naive[k]
			pi = min_dr_naive_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_naive_test[1] += g.get_error()
			s_naive += 1
	agnostic_naive[1] /= s_naive
	agnostic_naive_test[1] /= s_naive

	s_naive = 0
	for k in min_la_naive.keys():
		if min_la_naive[k] < 100000:
			agnostic_naive[2] += min_la_naive[k]
			pi = min_la_naive_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_naive_test[2] += g.get_error()
			s_naive += 1
	agnostic_naive[2] /= s_naive
	agnostic_naive_test[2] /= s_naive

	agnostic_naive_all[cnt, :] = agnostic_naive
	naive_naive_all[cnt, :] = naive_naive
	opt_naive_all[cnt, :] = opt_naive
	naive_opt_all[cnt, :] = naive_opt
	agnostic_naive_all_test[cnt, :] = agnostic_naive_test
	naive_naive_all_test[cnt, :] = naive_naive_test
	opt_naive_all_test[cnt, :] = opt_naive_test
	naive_opt_all_test[cnt, :] = naive_opt_test
	cnt += 1

agnostic_naive = np.mean(agnostic_naive_all, 0)
naive_naive = np.mean(naive_naive_all, 0)
opt_naive = np.mean(opt_naive_all, 0)
naive_opt = np.mean(naive_opt_all, 0)
agnostic_naive_test = np.mean(agnostic_naive_all_test, 0)
naive_naive_test = np.mean(naive_naive_all_test, 0)
opt_naive_test = np.mean(opt_naive_all_test, 0)
naive_opt_test = np.mean(naive_opt_all_test, 0)



for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full.pkl', 'rb'), encoding='latin1')

	pipelines = obj.pipelines
	path_pipelines = []
	for i in range(len(pipelines)):
		pi = pipelines[i]
		path = (pi.feature_extraction.decode('latin1'), pi.dimensionality_reduction.decode('latin1'), pi.learning_algorithm.decode('latin1'))
		if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
			path_pipelines.append((path, pipelines[i]))

	opt_opt = [100000] * 3
	agnostic_opt = [0] * 3
	opt_opt_paths = [None] * 3
	agnostic_opt_paths = [None] * 3
	opt_opt_test = [100000] * 3
	agnostic_opt_test = [0] * 3

	min_fe = {}
	min_fe_paths = {}
	for p in range(len(fe_params)):
		min_fe[p] = 1000000
		min_fe_paths[p] = None

	min_dr = {}
	min_dr_paths = {}
	for p in range(len(dr_params)):
		min_dr[p] = 1000000
		min_dr_paths[p] = None

	min_la = {}
	min_la_paths = {}
	for p in range(len(la_params)):
		min_la[p] = 1000000
		min_la_paths[p] = None

	# Step 1: Feature extraction
	for i in range(len(path_pipelines)):
		path = path_pipelines[i][0]
		pi = path_pipelines[i][1]
		fe = path[0]
		dr = path[1]
		la = path[2]
		# errs = []
		# for j in range(len(pi)):
		# 	errs.append(pi.get_error())
		err = pi.get_error()
		fe_param = pi.haralick_distance
		fe_ind = fe_params.index(fe_param)
		dr_param = pi.pca_whiten
		dr_ind = dr_params.index(dr_param)
		la_param = (pi.n_estimators, pi.max_features)
		dists = []
		for k in range(len(la_params)):
			dists.append(np.linalg.norm(np.asarray(la_param) - np.asarray(la_params[k])))
		la_ind = np.argmin(dists)
		p = pipeline['feature_extraction']
		for j in range(len(p)-1):
			alg = p[j]
			if alg == fe:
				if err < opt_opt[0]:  # opt -> naive
					opt_opt[0] = err
					opt_opt_paths[0] = pi
				if err < min_fe[fe_ind]:
					min_fe[fe_ind] = err  # agnostic -> naive
					min_fe_paths[fe_ind] = pi

		# Step 2: Dimensionality reduction
		p = pipeline['dimensionality_reduction']
		for j in range(len(p)-1):
			alg = p[j]
			if alg == dr:
				if err < opt_opt[1]:  # opt -> naive
					opt_opt[1] = err
					opt_opt_paths[1] = pi
				if err < min_dr[dr_ind]:  # agnostic -> naive
					min_dr[dr_ind] = err
					min_dr_paths[dr_ind] = pi


		# Step 2: Learning algorithm
		p = pipeline['learning_algorithm']
		for j in range(len(p)-1):
			alg = p[j]
			if alg == la:
				if err < opt_opt[2]:  # opt -> naive
					opt_opt[2] = err
					opt_opt_paths[2] = pi
				if err < min_la[la_ind]:  # agnostic -> naive
					min_la[la_ind] = err
					min_la_paths[la_ind] = pi

	for i in range(len(opt_opt)):
		pi = opt_opt_paths[i]
		hyper = pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction,
										  dr=pi.dimensionality_reduction,
										  la=pi.learning_algorithm,
										  val_splits=3, test_size=0.2)
		g.run()
		opt_opt_test[i] = g.get_error()

	s = 0
	for k in min_fe.keys():
		if min_fe[k] < 100000:
			agnostic_opt[0] += min_fe[k]
			pi = min_fe_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_opt_test[0] += g.get_error()
			s += 1
	agnostic_opt[0] /= s
	agnostic_opt_test[0] /= s

	s = 0
	for k in min_dr.keys():
		if min_dr[k] < 100000:
			agnostic_opt[1] += min_dr[k]
			pi = min_dr_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_opt_test[1] += g.get_error()
			s += 1
	agnostic_opt[1] /= s
	agnostic_opt_test[1] /= s

	s = 0
	for k in min_la.keys():
		if min_la[k] < 100000:
			agnostic_opt[2] += min_la[k]
			pi = min_la_paths[k]
			hyper = pi.kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=pi.feature_extraction,
											  dr=pi.dimensionality_reduction,
											  la=pi.learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			agnostic_opt_test[2] += g.get_error()
			s += 1
	agnostic_opt[2] /= s
	agnostic_opt_test[2] /= s

	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_opt[i] - opt_opt[i]
	E1[run - 1, :] = np.asarray(e)

	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_opt_test[i] - opt_opt_test[i]
	E1_test[run - 1, :] = np.asarray(e)

	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_naive[i] - opt_naive[i]
	E2[run - 1, :] = np.asarray(e)

	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_naive_test[i] - opt_naive_test[i]
	E2_test[run - 1, :] = np.asarray(e)

	e = [0] * 3
	for i in range(3):
		e[i] = naive_naive[i] - naive_opt[i]
	E3[run - 1, :] = np.asarray(e)

	e = [0] * 3
	for i in range(3):
		e[i] = naive_naive_test[i] - naive_opt_test[i]
	E3_test[run - 1, :] = np.asarray(e)

alpha = np.zeros(E1.shape, dtype='float64')
gamma = np.zeros(E1.shape, dtype='float64')
E_propagation = np.zeros(E1.shape, dtype='float64')
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
			g1 = 0
		else:
			a = e1 * e3 / (e2 + e3 - e1)
			g = (e2 - e1) / e3
			g1 = e1 * (e2 - e1) / (e2 + e3 - e1)
		alpha[j, i] = a
		gamma[j, i] = g
		E_propagation[j, i] = g1

E_total = []
E_total_std = []
for i in range(3):
	e1 = E1[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_total.append(np.mean(s1))
	E_total_std.append(np.std(s1))
E_total = np.asarray(E_total)
E_total_std = np.asarray(E_total_std)

E_direct = []
E_direct_std = []
for i in range(3):
	e1 = alpha[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_direct.append(np.mean(s1))
	E_direct_std.append(np.std(s1))
E_direct = np.asarray(E_direct)
E_direct_std = np.asarray(E_direct_std)

E_propagation_mean = []
E_propagation_std = []
for i in range(3):
	e1 = E_propagation[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_propagation_mean.append(np.mean(s1))
	E_propagation_std.append(np.std(s1))
E_propagation_mean = np.asarray(E_propagation_mean)
E_propagation_std = np.asarray(E_propagation_std)

gamma_mean = []
gamma_std = []
for i in range(3):
	e1 = gamma[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	gamma_mean.append(np.mean(s1))
	gamma_std.append(np.std(s1))
gamma_mean = np.asarray(gamma_mean)
gamma_std = np.asarray(gamma_std)






E1 = E1_test
E2 = E2_test
E3 = E3_test
alpha = np.zeros(E1.shape, dtype='float64')
gamma = np.zeros(E1.shape, dtype='float64')
E_propagation = np.zeros(E1.shape, dtype='float64')
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
			g1 = 0
		else:
			a = e1 * e3 / (e2 + e3 - e1)
			g = (e2 - e1) / e3
			g1 = e1 * (e2 - e1) / (e2 + e3 - e1)
		alpha[j, i] = a
		gamma[j, i] = g
		E_propagation[j, i] = g1

E_total_test = []
E_total_std_test = []
for i in range(3):
	e1 = E1[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_total_test.append(np.mean(s1))
	E_total_std_test.append(np.std(s1))
E_total_test = np.asarray(E_total_test)
E_total_std_test = np.asarray(E_total_std_test)

E_direct_test = []
E_direct_std_test = []
for i in range(3):
	e1 = alpha[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_direct_test.append(np.mean(s1))
	E_direct_std_test.append(np.std(s1))
E_direct_test = np.asarray(E_direct_test)
E_direct_std_test = np.asarray(E_direct_std_test)

E_propagation_mean_test = []
E_propagation_std_test = []
for i in range(3):
	e1 = E_propagation[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	E_propagation_mean_test.append(np.mean(s1))
	E_propagation_std_test.append(np.std(s1))
E_propagation_mean_test = np.asarray(E_propagation_mean_test)
E_propagation_std_test = np.asarray(E_propagation_std_test)

gamma_mean_test = []
gamma_std_test = []
for i in range(3):
	e1 = gamma[:, i]
	s1 = []
	for j in range(len(e1)):
		if e1[j] != 0:
			s1.append(e1[j])
	gamma_mean_test.append(np.mean(s1))
	gamma_std_test.append(np.std(s1))
gamma_mean_test = np.asarray(gamma_mean_test)
gamma_std_test = np.asarray(gamma_std_test)


from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = ['Haralick', 'PCA', 'RF']
x1 = np.asarray([1, 2, 3])
w = 0.2
d = w * np.ones(3)
x2 = x1 + d
plt.figure()
plt.bar(x1, E_total_test.ravel(), width=w, align='center', color=colors[0], yerr=E_total_std_test.ravel(), label='Total error')
plt.bar(x2, E_direct_test.ravel(), width=w, align='center', color=colors[1], yerr=E_direct_std_test.ravel(), label='Direct error')
plt.bar(x2, E_propagation_mean_test.ravel(), width=w, bottom=E_direct_test.ravel(), align='center', color=colors[2],
		yerr=E_propagation_std_test.ravel(), label='Propagation error')
plt.title('Error propagation in algorithms for ' + data_name + ' data')
plt.xlabel('Algorithms')
plt.ylabel('Error')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_random_pipeline_algorithms_' + type1 + '_' + data_name + '_test.jpg')
plt.close()

plt.figure()
x1 = np.asarray([1, 2, 3])
plt.bar(x1, gamma_mean_test.ravel(), width=w, color='r', yerr=gamma_std_test.ravel())
plt.title('Propagation factor in algorithms for ' + data_name + ' data')
plt.xlabel('Algorithms')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_algorithms_' + type1 + '_'  + data_name + '_test.jpg')
plt.close()


x = pipeline['all']

_, axs = plt.subplots(nrows=1, ncols=1)
x1 = [1, 2, 3]
axs.errorbar(x1, E_propagation_mean, E_propagation_std, linestyle='None', marker='^', capsize=3, color='b', label='validation')
axs.plot(x1, E_propagation_mean, color='b')
axs.errorbar(x1, E_propagation_mean_test, E_propagation_std_test, linestyle='None', marker='^', capsize=3, color='k', label='testing')
axs.plot(x1, E_propagation_mean_test, color='k')
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
plt.savefig(results_home + 'figures/error_propagation_algorithms_' + data_name + '_val_test.jpg')
plt.close()
