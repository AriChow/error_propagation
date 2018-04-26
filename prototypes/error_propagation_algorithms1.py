'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle

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

	min_fe_naive = {}
	for p in range(len(fe_params)):
		min_fe_naive[p] = 1000000

	min_dr_naive = {}
	for p in range(len(dr_params)):
		min_dr_naive[p] = 1000000

	min_la_naive = {}
	for p in range(len(la_params)):
		min_la_naive[p] = 1000000


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
					if 'naive' not in dr and 'naive' not in la:  # naive -> opt
						if err < naive_opt[0]:
							naive_opt[0] = err
				else:  # opt -> and agnostic ->
					if 'naive' in dr and 'naive' in la:
						if err < opt_naive[0]:  # opt -> naive
							opt_naive[0] = err
						fe_param = pi.haralick_distance
						fe_ind = fe_params.index(fe_param)
						if err < min_fe_naive[fe_ind]:
							min_fe_naive[fe_ind] = err  # agnostic -> naive

		# Step 2: Dimensionality reduction
		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if alg == dr:
				if 'naive' in alg:  # naive ->
					if 'naive' not in fe and 'naive' in la:
						if err < naive_naive[1]:  # naive -> naive
							naive_naive[1] = err
					if 'naive' not in fe and 'naive' not in la:  # naive -> opt
						if err < naive_opt[1]:
							naive_opt[1] = err
				else:  # opt -> and agnostic ->
					if 'naive' not in fe and 'naive' in la:
						if err < opt_naive[1]:  # opt -> naive
							opt_naive[1] = err
						dr_param = pi.pca_whiten
						dr_ind = dr_params.index(dr_param)
						if err < min_dr_naive[dr_ind]:  # agnostic -> naive
							min_dr_naive[dr_ind] = err


		# Step 2: Learning algorithm
		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if alg == la:
				if 'naive' in alg:  # naive ->
					if 'naive' not in fe and 'naive' not in dr:
						if err < naive_naive[2]:  # naive -> naive
							naive_naive[2] = err
					if 'naive' not in fe and 'naive' not in dr:  # naive -> opt
						if err < naive_opt[2]:
							naive_opt[2] = err
				else:  # opt -> and agnostic ->
					if 'naive' not in fe and 'naive' not in dr:
						if err < opt_naive[2]:  # opt -> naive
							opt_naive[2] = err
						la_param = (pi.n_estimators, pi.max_features)
						dists = []
						for k in range(len(la_params)):
							dists.append(np.linalg.norm(np.asarray(la_param) - np.asarray(la_params[k])))
						la_ind = np.argmin(dists)
						if err < min_la_naive[la_ind]:  # agnostic -> naive
							min_la_naive[la_ind] = err


	s_naive = 0
	for k in min_fe_naive.keys():
		if min_fe_naive[k] < 100000:
			agnostic_naive[0] += min_fe_naive[k]
			s_naive += 1
	agnostic_naive[0] /= s_naive

	s_naive = 0
	for k in min_dr_naive.keys():
		if min_dr_naive[k] < 100000:
			agnostic_naive[1] += min_dr_naive[k]
			s_naive += 1
	agnostic_naive[1] /= s_naive

	s_naive = 0
	for k in min_la_naive.keys():
		if min_la_naive[k] < 100000:
			agnostic_naive[2] += min_la_naive[k]
			s_naive += 1
	agnostic_naive[2] /= s_naive
	agnostic_naive[2] /= s_naive

	agnostic_naive = np.asarray(agnostic_naive)
	agnostic_naive = np.expand_dims(agnostic_naive, 0)
	naive_naive = np.asarray(naive_naive)
	naive_naive = np.expand_dims(naive_naive, 0)
	opt_naive = np.asarray(opt_naive)
	opt_naive = np.expand_dims(opt_naive, 0)
	naive_opt = np.asarray(naive_opt)
	naive_opt = np.expand_dims(naive_opt, 0)
	if cnt == 0:
		agnostic_naive_all = agnostic_naive
		naive_naive_all = naive_naive
		opt_naive_all = opt_naive
		naive_opt_all = naive_opt
	else:
		agnostic_naive_all = np.vstack((agnostic_naive_all, agnostic_naive))
		naive_naive_all = np.vstack((naive_naive_all, naive_naive))
		opt_naive_all = np.vstack((opt_naive_all, opt_naive))
		naive_opt_all = np.vstack((naive_opt_all, naive_opt))
	cnt += 1

agnostic_naive = np.mean(agnostic_naive_all, 0)
naive_naive = np.mean(naive_naive_all, 0)
opt_naive = np.mean(opt_naive_all, 0)
naive_opt = np.mean(naive_opt_all, 0)

cnt = 0
opt_opt_all = np.zeros((1, 3))
agnostic_opt_all = np.zeros((1, 3))

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

	min_fe = {}
	for p in range(len(fe_params)):
		min_fe[p] = 1000000

	min_dr = {}
	for p in range(len(dr_params)):
		min_dr[p] = 1000000

	min_la = {}
	for p in range(len(la_params)):
		min_la[p] = 1000000

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
				if err < min_fe[fe_ind]:
					min_fe[fe_ind] = err  # agnostic -> naive

		# Step 2: Dimensionality reduction
		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if alg == dr:
					if err < opt_opt[1]:  # opt -> naive
							opt_opt[1] = err
					if err < min_dr[dr_ind]:  # agnostic -> naive
						min_dr[dr_ind] = err

		# Step 2: Learning algorithm
		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if alg == la:
				if err < opt_opt[2]:  # opt -> naive
					opt_opt[2] = err
				if err < min_la[la_ind]:  # agnostic -> naive
					min_la[la_ind] = err

	s = 0
	for k in min_fe.keys():
		if min_fe[k] < 100000:
			agnostic_opt[0] += min_fe[k]
			s += 1
	agnostic_opt[0] /= s

	s = 0
	for k in min_dr.keys():
		if min_dr[k] < 100000:
			agnostic_opt[1] += min_dr[k]
			s += 1
	agnostic_opt[1] /= s

	s = 0
	for k in min_la.keys():
		if min_la[k] < 100000:
			agnostic_opt[2] += min_la[k]
			s += 1
	agnostic_opt[2] /= s

	opt_opt = np.asarray(opt_opt)
	opt_opt = np.expand_dims(opt_opt, 0)
	agnostic_opt = np.asarray(agnostic_opt)
	agnostic_opt = np.expand_dims(agnostic_opt, 0)
	if cnt == 0:
		opt_opt_all = opt_opt
		agnostic_opt_all = agnostic_opt
	else:
		opt_opt_all = np.vstack((opt_opt_all, opt_opt))
		agnostic_opt_all = np.vstack((agnostic_opt_all, agnostic_opt))
	cnt += 1


agnostic_opt = np.mean(agnostic_opt_all, 0)
opt_opt = np.mean(opt_opt_all, 0)

def func_beta(p, e1, e2, e3, e4, e5, e6):
	alpha1, alpha2, alpha3, gamma1, gamma2, beta = p
	f1 = alpha1 + gamma1 * (alpha1 * beta) - e1
	f2 = alpha2 + gamma1 * (alpha2 + beta) - e2
	f3 = alpha1 + gamma2 * (alpha1 + beta) - e3
	f4 = alpha2 + gamma2 *(alpha2 + beta) - e4
	f5 = alpha3 + gamma2 * (alpha3 + beta) - e5
	f6 = alpha3 + gamma1 * (alpha3 + beta) - e6
	return (f1, f2, f3, f4, f5, f6)

def func_gamma(p, e1, e2, e3, e4, e5, e6):
	alpha1, alpha2, alpha3, gamma, beta1, beta2 = p
	f1 = alpha1 + gamma * (alpha1 * beta1) - e1
	f2 = alpha2 + gamma * (alpha2 + beta1) - e2
	f3 = alpha1 + gamma * (alpha1 + beta2) - e3
	f4 = alpha2 + gamma * (alpha2 + beta2) - e4
	f5 = alpha3 + gamma * (alpha3 + beta2) - e5
	f6 = alpha3 + gamma * (alpha3 + beta1) - e6
	return (f1, f2, f3, f4, f5, f6)



p = np.random.random(6)
from scipy.optimize import fsolve
parameters = np.zeros((3, 6))
error = np.zeros((3, 6))
for i in range(3):
	errs = (agnostic_opt[i], opt_opt[i], agnostic_naive[i], opt_naive[i], naive_naive[i], naive_opt[i])
	params = fsolve(func_beta, p, errs)
	err = func_beta(tuple(params), errs[0], errs[1], errs[2], errs[3], errs[4], errs[5])
	error[i, :] = np.expand_dims(np.asarray(err), 0)
	parameters[i, :] = np.expand_dims(np.asarray(params), 0)

print('Beta assumption:')
print('Parameters:')
print(parameters)

print('Errors:')
print(error)


# p = np.random.random(6)
# from scipy.optimize import fsolve
# parameters = np.zeros((3, 6))
# error = np.zeros((3, 6))
# for i in range(3):
# 	errs = (agnostic_opt[i], opt_opt[i], agnostic_naive[i], opt_naive[i], naive_naive[i], naive_opt[i])
# 	params = fsolve(func_gamma, p, errs)
# 	err = func_beta(tuple(params), errs[0], errs[1], errs[2], errs[3], errs[4], errs[5])
# 	error[i, :] = np.expand_dims(np.asarray(err), 0)
# 	parameters[i, :] = np.expand_dims(np.asarray(params), 0)
#
# print ('\n\n')
# print('Gamma assumption:')
# print('Parameters:')
# print(parameters)
#
# print('Errors:')
# print(error)


# errors = {'opt_opt': opt_opt_all, 'agnostic_opt': agnostic_opt_all, 'agnostic_naive': agnostic_naive_all,
# 		  'naive_naive': naive_naive_all, 'opt_naive': opt_naive_all, 'naive_opt': naive_opt_all}
# pickle.dump(errors, open(results_home + 'intermediate/random_MCMC/error_propagation_steps1.pkl', 'wb'))

print()


