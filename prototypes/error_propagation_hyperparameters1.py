'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle

home = os.path.expanduser('~')
datasets = ['breast']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["haralick", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'][:-1] + pipeline['dimensionality_reduction'][:-1] + pipeline['learning_algorithm'][:-1]

E_totals = np.zeros((1, 3))
E_directs = np.zeros((1, 3))
E_propagations = np.zeros((1, 3))
gammas = np.zeros((1, 3))

for z in range(len(datasets)):
	data_name = datasets[z]
	# Random1
	start = 1
	stop = 2
	type1 = 'grid_MCMC'
	type2 = 'grid_MCMC'
	E1 = np.zeros((stop - 1, 3))
	E2 = np.zeros((stop - 1, 3))
	E3 = np.zeros((stop - 1, 3))


	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(1) + '.pkl','rb'), encoding='latin1')
	pipelines = obj.pipelines
	path_pipelines = []
	for i in range(len(obj.paths)):
		path = obj.paths[i]
		if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
			path_pipelines = pipelines[i]
			break

	fe_params = set()
	dr_params = set()
	la_params = set()
	for i in range(len(path_pipelines)):
		p = path_pipelines[i]
		fe_params.add(p.haralick_distance)
		dr_params.add(p.n_neighbors)
		la_params.add(p.n_estimators)

	fe_params = list(fe_params)
	dr_params = list(dr_params)
	la_params = list(la_params)

	cnt = 0
	agnostic_naive_all = np.zeros((1, 3))
	naive_naive_all = np.zeros((1, 3))
	opt_naive_all = np.zeros((1, 3))
	naive_opt_all = np.zeros((1, 3))

	for run in range(1, 2):
		obj = pickle.load(
				open(results_home + 'intermediate/' + type2 + '/' + type2 + '_' + data_name + '_run_' +
					 str(run) + '_naive_propagation.pkl', 'rb'), encoding='latin1')
		pipelines = obj.pipelines
		paths = obj.paths
		path_pipelines = []
		for i in range(len(pipelines)):
			pi = pipelines[i]
			path = paths[i]
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
			pi1 = path_pipelines[i][1]
			fe = path[0]
			dr = path[1]
			la = path[2]
			for u in range(len(pi1)):
				pi = pi1[u]
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
								dr_param = pi.n_neighbors
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
								la_param = pi.n_estimators
								la_ind = la_params.index(la_param)
								if err < min_la_naive[la_ind]:  # agnostic -> naive
									min_la_naive[la_ind] = err


		s_naive = 0
		for k in min_fe_naive.keys():
			if min_fe_naive[k] < 100000:
				agnostic_naive[0] += min_fe_naive[k]
				s_naive += 1
		if s_naive == 0:
			agnostic_naive[0] = 0
		else:
			agnostic_naive[0] /= s_naive

		s_naive = 0
		for k in min_dr_naive.keys():
			if min_dr_naive[k] < 100000:
				agnostic_naive[1] += min_dr_naive[k]
				s_naive += 1
		if s_naive == 0:
			agnostic_naive[1] = 0
		else:
			agnostic_naive[1] /= s_naive

		s_naive = 0
		for k in min_la_naive.keys():
			if min_la_naive[k] < 100000:
				agnostic_naive[2] += min_la_naive[k]
				s_naive += 1
		if s_naive == 0:
			agnostic_naive[2] = 0
		else:
			agnostic_naive[2] /= s_naive

		agnostic_naive_all[cnt, :] = agnostic_naive
		naive_naive_all[cnt, :] = naive_naive
		opt_naive_all[cnt, :] = opt_naive
		naive_opt_all[cnt, :] = naive_opt
		cnt += 1

	agnostic_naive = np.mean(agnostic_naive_all, 0)
	naive_naive = np.mean(naive_naive_all, 0)
	opt_naive = np.mean(opt_naive_all, 0)
	naive_opt = np.mean(naive_opt_all, 0)



	for run in range(start, stop):
		obj = pickle.load(
			open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
				 str(run) + '.pkl', 'rb'), encoding='latin1')

		pipelines = obj.pipelines
		path_pipelines = []
		for i in range(len(pipelines)):
			path = obj.paths[i]
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
			pi1 = path_pipelines[i][1]
			fe = path[0]
			dr = path[1]
			la = path[2]

			for u in range(len(pi1)):
				pi = pi1[u]
				# errs = []
				# for j in range(len(pi)):
				# 	errs.append(pi.get_error())
				err = pi.get_error()
				fe_param = pi.haralick_distance
				fe_ind = fe_params.index(fe_param)
				dr_param = pi.n_neighbors
				dr_ind = dr_params.index(dr_param)
				la_param = pi.n_estimators
				la_ind = la_params.index(la_param)

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

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_opt[i] - opt_opt[i]
		E1[run - 1, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_naive[i] - opt_naive[i]
		E2[run - 1, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = naive_naive[i] - naive_opt[i]
		E3[run - 1, :] = np.asarray(e)

	alpha = np.zeros(E1.shape, dtype='float64')
	gamma = np.zeros(E1.shape, dtype='float64')
	E_propagation = np.zeros(E1.shape, dtype='float64')
	for i in range(2):
		E11 = E1[:, i]
		E22 = E2[:, i]
		E33 = E3[:, i]
		for j in range(len(E11)):
			e1 = E11[j]
			e2 = E22[j]
			e3 = E33[j]
			a = e1 * e3 / (e2 + e3 - e1)
			g = (e2 - e1) / e3
			alpha[j, i] = a
			gamma[j, i] = g
			E_propagation[j, i] = a * g

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
	E_direct[2] = E_total[2]
	E_direct_std[2] = E_total_std[2]

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
	E_propagation_mean[2] = 0
	E_propagation_std[2] = 0

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
	gamma_mean[2] = 0
	gamma_std[2] = 0
	gammas[z, :] = gamma_mean
	E_totals[z, :] = E_total
	E_propagations[z, :] = E_propagation_mean
	E_directs[z, :] = E_direct


from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = ['Haralick distance', 'Number of neighbors', 'Number of estimators']
x1 = np.asarray([1, 2, 3])
w = 0.2
d = w * np.ones(3)
x2 = x1 + d
plt.figure()
plt.bar(x1, np.mean(E_totals, 0), width=w, align='center', color=colors[0], yerr=np.std(E_totals, 0), label='Total error')
plt.bar(x2, np.mean(E_directs, 0), width=w, align='center', color=colors[1], yerr=np.std(E_directs, 0), label='Direct error')
plt.bar(x2, np.mean(E_propagations, 0), width=w, bottom=np.mean(E_directs, 0), align='center', color=colors[2],
		yerr=np.std(E_propagations, 0), label='Propagation error')
plt.title('Error propagation in hyper-parameters')
plt.xlabel('Hyper-parameters')
plt.ylabel('Error')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_random_pipeline_hyperparameters_breast.eps')
plt.close()

plt.figure()
x1 = np.asarray([1, 2, 3])
plt.bar(x1, np.mean(gammas, 0), width=w, color='r', yerr=np.std(gammas, 0))
plt.title('Propagation factor in hyper-parameters')
plt.xlabel('Hyper-parameters')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_hyperparameters_breast.eps')
plt.close()
