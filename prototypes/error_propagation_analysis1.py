'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle

home = os.path.expanduser('~')
data_name = sys.argv[1]
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
E1 = np.zeros((stop - 1, 3))
E2 = np.zeros((stop - 1, 3))
E3 = np.zeros((stop - 1, 3))

for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')

	pipelines = obj.pipelines
	opt_opt = [1000000] * 3
	naive_naive = [1000000] * 3
	naive_opt = [100000] * 3
	opt_naive = [100000] * 3
	agnostic_naive = [0] * 3
	agnostic_opt = [0] * 3

	min_fe = {}
	min_fe_naive = {}
	for p in pipeline['feature_extraction']:
		if 'naive' not in p:
			min_fe[p] = 1000000
			min_fe_naive[p] = 1000000
	min_dr = {}
	min_dr_naive = {}
	for p in pipeline['dimensionality_reduction']:
		if 'naive' not in p:
			min_dr[p] = 1000000
			min_dr_naive[p] = 1000000
	min_la = {}
	min_la_naive = {}
	for p in pipeline['learning_algorithm']:
		if 'naive' not in p:
			min_la[p] = 1000000
			min_la_naive[p] = 1000000

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
							min_fe_naive[alg] = err # agnostic -> naive1
					if 'naive' not in dr and 'naive' not in la:
						if err < opt_opt[0]:
							opt_opt[0] = err
						if err < min_fe[alg]:  # agnostic -> opt
							min_fe[alg] = err

		#  Step 2: Dimensionality reduction
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
					if 'naive' not in fe and 'naive' not in la:
						if err < opt_opt[1]:
							opt_opt[1] = err
						if err < min_dr[alg]:  # agnostic -> opt
							min_dr[alg] = err
					
		#  Step 2: Learning algorithm
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
					if 'naive' not in fe and 'naive' not in dr:
						if err < opt_opt[2]:
							opt_opt[2] = err
						if err < min_la[alg]:  # agnostic -> opt
							min_la[alg] = err

	s = 0
	s_naive = 0
	for k in min_fe.keys():
		if min_fe[k] < 100000:
			agnostic_opt[0] += min_fe[k]
			s += 1
		if min_fe_naive[k] < 100000:
			agnostic_naive[0] += min_fe_naive[k]
			s_naive += 1
	agnostic_opt[0] /= s
	agnostic_naive[0] /= s_naive

	s = 0
	s_naive = 0
	for k in min_dr.keys():
		if min_dr[k] < 100000:
			agnostic_opt[1] += min_dr[k]
			s += 1
		if min_dr_naive[k] < 100000:
			agnostic_naive[1] += min_dr_naive[k]
			s_naive += 1
	agnostic_opt[1] /= s
	agnostic_naive[1] /= s_naive

	s = 0
	s_naive = 0
	for k in min_la.keys():
		if min_la[k] < 100000:
			agnostic_opt[2] += min_la[k]
			s += 1
		if min_la_naive[k] < 100000:
			agnostic_naive[2] += min_la_naive[k]
			s_naive += 1
	agnostic_opt[2] /= s
	agnostic_naive[2] /= s_naive

	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_opt[i] - opt_opt[i]
	E1[run-1, :] = np.asarray(e)


	e = [0] * 3
	for i in range(3):
		e[i] = agnostic_naive[i] - opt_naive[i]
	E2[run - 1, :] = np.asarray(e)


	e = [0] * 3
	for i in range(3):
		e[i] = naive_naive[i] - naive_opt[i]
	E3[run - 1, :] = np.asarray(e)


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
steps = ['FE', 'FT', 'LA']
x1 = np.asarray([1, 2, 3])
w = 0.2
d = w * np.ones(3)
x2 = x1 + d
plt.figure()
plt.bar(x1, E_total.ravel(), width=w, align='center', color=colors[0], yerr=E_total_std.ravel(), label='Total error')
plt.bar(x2, E_direct.ravel(), width=w, align='center', color=colors[1], yerr=E_direct_std.ravel(), label='Direct error')
plt.bar(x2, E_propagation_mean.ravel(), width=w, bottom=E_direct.ravel(), align='center', color=colors[2], yerr=E_propagation_std.ravel(), label='Propagation error')
plt.title('Error quantification and propagation in steps for ' + data_name + ' data')
plt.xlabel('Steps')
plt.ylabel('Error contributions')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_random_pipeline_steps_' + data_name + '.jpg')
plt.close()

plt.figure()
x1 = np.asarray([1, 2, 3])
plt.bar(x1, gamma_mean.ravel(), width=w, color='r', yerr=gamma_std.ravel())
plt.title('Propagation factor in steps for ' + data_name + ' data')
plt.xlabel('Steps')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_steps_' + data_name + '.jpg')
plt.close()
