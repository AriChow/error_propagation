'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle

# np.random.seed(42)

home = os.path.expanduser('~')
data_name = 'matsc_dataset2'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
type2 = 'random_MCMC'
alpha1 = np.zeros((stop - 1, 3))
alpha2 = np.zeros((stop - 1, 3))
alpha3 = np.zeros((stop - 1, 3))
beta1 = np.zeros((stop - 1, 3))
beta2 = np.zeros((stop - 1, 3))
gamma1 = np.zeros((stop - 1, 3))
gamma2 = np.zeros((stop - 1, 3))

if data_name == 'breast':
	stop = 3
if data_name == 'brain':
	stop = 5

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
	naive_naive = [1000000] * 3
	naive_opt = [100000] * 3
	opt_naive = [100000] * 3
	agnostic_naive = [0] * 3

	min_fe_naive = {}
	for p in pipeline['feature_extraction']:
		if 'naive' not in p:
			min_fe_naive[p] = 1000000
	min_dr_naive = {}
	for p in pipeline['dimensionality_reduction']:
		if 'naive' not in p:
			min_dr_naive[p] = 1000000
	min_la_naive = {}
	for p in pipeline['learning_algorithm']:
		if 'naive' not in p:
			min_la_naive[p] = 1000000

	# Step 1: Feature extraction

	for i in range(len(pipelines)):
		pi = pipelines[i]
		path = (pi.feature_extraction, pi.dimensionality_reduction, pi.learning_algorithm)
		fe = path[0]
		dr = path[1]
		la = path[2]
		errs = []
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
						if err < min_fe_naive[alg]:
							min_fe_naive[alg] = err  # agnostic -> naive1

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
						if err < min_dr_naive[alg]:  # agnostic -> naive
							min_dr_naive[alg] = err

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
						if err < min_la_naive[alg]:  # agnostic -> naive
							min_la_naive[alg] = err

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
	opt_opt = [1000000] * 3
	agnostic_opt = [0] * 3
	min_fe = {}
	for p in pipeline['feature_extraction']:
		if 'naive' not in p:
			min_fe[p] = 1000000
	min_dr = {}
	for p in pipeline['dimensionality_reduction']:
		if 'naive' not in p:
			min_dr[p] = 1000000
	min_la = {}
	for p in pipeline['learning_algorithm']:
		if 'naive' not in p:
			min_la[p] = 1000000

	for i in range(len(pipelines)):
		pi = pipelines[i]
		fe = pi.feature_extraction.decode('latin1')
		dr = pi.dimensionality_reduction.decode('latin1')
		la = pi.learning_algorithm.decode('latin1')
		path = (fe, dr, la)
		err = pi.get_error()

		#  Step 1: Feature extraction
		p = pipeline['feature_extraction']
		for j in range(len(p)):
			alg = p[j]
			if alg == fe:
				if err < opt_opt[0]:
					opt_opt[0] = err
				if err < min_fe[alg]:  # agnostic -> opt
					min_fe[alg] = err

		# Step 2: Dimensionality reduction
		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if alg == dr:
				if err < opt_opt[1]:
					opt_opt[1] = err
				if err < min_dr[alg]:  # agnostic -> opt
					min_dr[alg] = err

		# Step 2: Learning algorithm
		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if alg == la:
				if err < opt_opt[2]:
					opt_opt[2] = err
				if err < min_la[alg]:  # agnostic -> opt
					min_la[alg] = err

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



p = np.ones(6)
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

E_total = agnostic_opt

E_direct = parameters[:, 0]

gamma_mean = parameters[:, 3]

E_propagation_mean = gamma_mean * (E_direct + parameters[:, 5])
# E_direct[-1] = E_total[-1]
# E_propagation_mean[-1] = 0


from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = ['Feature extraction', 'Feature transformation', 'Learning algorithms']
x1 = np.asarray([1, 2, 3])
w = 0.2
d = w * np.ones(3)
x2 = x1 + d
plt.figure()
plt.bar(x1, E_total.ravel(), width=w, align='center', color=colors[0], label='Total error')
plt.bar(x2, E_direct.ravel(), width=w, align='center', color=colors[1], label='Direct error')
plt.bar(x2, E_propagation_mean.ravel(), width=w, bottom=E_direct.ravel(), align='center', color=colors[2], label='Propagation error')
plt.title('Error propagation in steps for ' + data_name + ' data')
plt.xlabel('Steps')
plt.ylabel('Error')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_new_random_pipeline_steps_' + type1 + '_' + data_name + '.jpg')
plt.close()

plt.figure()
x1 = np.asarray([1, 2, 3])
plt.bar(x1, gamma_mean.ravel(), width=w, color='r')
plt.title('Propagation factor in steps for ' + data_name + ' data')
plt.xlabel('Steps')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.savefig(results_home + 'figures/propagation_factor_new_random_pipeline_steps_' + type1 + '_'  + data_name + '.jpg')
plt.close()

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