'''
This file is for quantification of error contributions from
different computational steps of an ML pipeline.
'''

import numpy as np
import os
import sys
import pickle

home = os.path.expanduser('~')
datasets = ['matsc_dataset2']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]
naive_path = ["naive_feature_extraction", "naive_dimensionality_reduction", "naive_learning_algorithm"]

# Grid

E_totals = np.zeros((1, 3))
E_directs = np.zeros((1, 3))
E_propagations = np.zeros((1, 3))
gammas = np.zeros((1, 3))

for z in range(len(datasets)):
	data_name = datasets[z]
	start = 1
	stop = 2
	type2 = 'grid_MCMC'
	E1 = np.zeros((1, 3))
	E2 = np.zeros((1, 3))
	E3 = np.zeros((1, 3))
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



		#  Step 1: Feature extraction

		for i in range(len(pipelines)):
			pi = pipelines[i]
			fe = obj.paths[i][0]
			dr = obj.paths[i][1]
			la = obj.paths[i][2]
			errs = []
			for j in range(len(pi)):
				errs.append(pi[j].get_error())
			err = np.amin(errs)
			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if alg == fe:
					if naive_path[0] in alg:  # naive ->
						if naive_path[1] in dr and naive_path[2] in la:
							if err < naive_naive[0]:  # naive -> naive
								naive_naive[0] = err
						if naive_path[1] not in dr and naive_path[2] not in la:  # naive -> opt
							if err < naive_opt[0]:
								naive_opt[0] = err
					else:  # opt -> and agnostic ->
						if naive_path[1] in dr and naive_path[2] in la:
							if err < opt_naive[0]:  # opt -> naive
								opt_naive[0] = err
							if err < min_fe_naive[alg]:
								min_fe_naive[alg] = err  # agnostic -> naive1

			# Step 2: Dimensionality reduction
			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if alg == dr:
					if naive_path[1] in alg:  # naive ->
						if naive_path[0] not in fe and naive_path[2] in la:
							if err < naive_naive[1]:  # naive -> naive
								naive_naive[1] = err
						if naive_path[0] not in fe and naive_path[2] not in la:  # naive -> opt
							if err < naive_opt[1]:
								naive_opt[1] = err
					else:  # opt -> and agnostic ->
						if naive_path[0] not in fe and naive_path[2] in la:
							if err < opt_naive[1]:  # opt -> naive
								opt_naive[1] = err
							if err < min_dr_naive[alg]:  # agnostic -> naive
								min_dr_naive[alg] = err

			# Step 2: Learning algorithm
			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if alg == la:
					if naive_path[2] in alg:  # naive ->
						if naive_path[0] not in fe and naive_path[1] not in dr:
							if err < naive_naive[2]:  # naive -> naive
								naive_naive[2] = err
						if naive_path[0] not in fe and naive_path[1] not in dr:  # naive -> opt
							if err < naive_opt[2]:
								naive_opt[2] = err
					else:  # opt -> and agnostic ->
						if naive_path[0] not in fe and naive_path[1] not in dr:
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
			open(results_home + 'intermediate/' + type2 + '/' + type2 + '_' + data_name + '_run_' +
				 str(run) + '.pkl', 'rb'), encoding='latin1')

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
			fe = obj.paths[i][0]
			dr = obj.paths[i][1]
			la = obj.paths[i][2]
			errs = []
			for j in range(len(pi)):
				errs.append(pi[j].get_error())
			err = np.amin(errs)

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

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_opt[i] - opt_opt[i]
			if e[i] > 10000:
				e[i] = 0
		E1[run - 1, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_naive[i] - opt_naive[i]
			if e[i] > 10000:
				e[i] = 0
		E2[run - 1, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = naive_naive[i] - naive_opt[i]
			if e[i] > 10000:
				e[i] = 0
		E3[run - 1, :] = np.asarray(e)

	alpha = np.zeros(E1.shape)
	gamma = np.zeros(E1.shape)
	E_propagation = np.zeros(E1.shape)
	# for i in range(3):
	# 	E11 = E1[:, i]
	# 	E22 = E2[:, i]
	# 	E33 = E3[:, i]
	# 	for j in range(len(E11)):
	# 		e1 = E11[j]
	# 		e2 = E22[j]
	# 		e3 = E33[j]
	# 		if e3 == 0:
	# 			g = 0
	# 			a = e1
	# 		else:
	# 			a = e1 * e3 / (e2 + e3 - e1)
	# 			g = (e2 - e1) / e3
	# 		alpha[j, i] = a
	# 		gamma[j, i] = g
	# 		E_propagation[j, i] = a * g

	for i in range(2):
		E11 = E1[:, i]
		E22 = E2[:, i]
		E33 = E3[:, i]
		for j in range(len(E11)):
			e1 = E11[j]
			e2 = E22[j]
			e3 = E33[j]
			a = e1 * e3 / (e3 + e2 - e1)
			g = (e2 - e1) / e3
			alpha[j, i] = a
			gamma[j, i] = g
			E_propagation[j, i] = a * g


	E_total = np.mean(E1, 0)
	E_total_std = np.std(E1, 0)

	E_direct = np.mean(alpha, axis=0)
	E_direct_std = np.std(alpha, axis=0)
	E_direct[2] = E_total[2]
	E_direct_std[2] = E_total_std[2]

	E_propagation_mean = np.mean(E_propagation, axis=0)
	E_propagation_std = np.std(E_propagation, axis=0)
	E_propagation_mean[2] = 0
	E_propagation_std[2] = 0

	gamma_mean = np.mean(gamma, 0)
	gamma_std = np.std(gamma, 0)
	gamma_mean[2] = 0
	gamma_std[2] = 0

	gammas[z, :] = gamma_mean
	E_totals[z, :] = E_total
	E_propagations[z, :] = E_propagation_mean
	E_directs[z, :] = E_direct

E_direct_grid = E_directs
E_propagation_grid = E_propagations
E_total_grid = E_totals
gamma_grid = gammas



E_totals = np.zeros((1, 3))
E_directs = np.zeros((1, 3))
E_propagations = np.zeros((1, 3))
gammas = np.zeros((1, 3))

for z in range(len(datasets)):
	data_name = datasets[z]
	start = 1
	stop = 2
	type2 = 'random_MCMC'
	E1 = np.zeros((5, 3))
	E2 = np.zeros((5, 3))
	E3 = np.zeros((5, 3))
	cnt = 0
	agnostic_naive_all = np.zeros((5, 3))
	naive_naive_all = np.zeros((5, 3))
	opt_naive_all = np.zeros((5, 3))
	naive_opt_all = np.zeros((5, 3))

	for run in range(1, 6):
		obj = pickle.load(
			open(results_home + 'intermediate/' + type2 + '/' + type2 + '_' + data_name + '_run_' +
				 str(run) + '_naive_propagation.pkl', 'rb'), encoding='latin1')
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
		#  Step 1: Feature extraction

		for i in range(len(pipelines)):
			pi = pipelines[i]
			fe = obj.paths[i][0]
			dr = obj.paths[i][1]
			la = obj.paths[i][2]
			errs = []
			for j in range(len(pi)):
				errs.append(pi[j].get_error())
			err = np.amin(errs)
			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if alg == fe:
					if naive_path[0] in alg:  # naive ->
						if naive_path[1] in dr and naive_path[2] in la:
							if err < naive_naive[0]:  # naive -> naive
								naive_naive[0] = err
						if naive_path[1] not in dr and naive_path[2] not in la:  # naive -> opt
							if err < naive_opt[0]:
								naive_opt[0] = err
					else:  # opt -> and agnostic ->
						if naive_path[1] in dr and naive_path[2] in la:
							if err < opt_naive[0]:  # opt -> naive
								opt_naive[0] = err
							if err < min_fe_naive[alg]:
								min_fe_naive[alg] = err  # agnostic -> naive1

			# Step 2: Dimensionality reduction
			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if alg == dr:
					if naive_path[1] in alg:  # naive ->
						if naive_path[0] not in fe and naive_path[2] in la:
							if err < naive_naive[1]:  # naive -> naive
								naive_naive[1] = err
						if naive_path[0] not in fe and naive_path[2] not in la:  # naive -> opt
							if err < naive_opt[1]:
								naive_opt[1] = err
					else:  # opt -> and agnostic ->
						if naive_path[0] not in fe and naive_path[2] in la:
							if err < opt_naive[1]:  # opt -> naive
								opt_naive[1] = err
							if err < min_dr_naive[alg]:  # agnostic -> naive
								min_dr_naive[alg] = err

			# Step 2: Learning algorithm
			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if alg == la:
					if naive_path[2] in alg:  # naive ->
						if naive_path[0] not in fe and naive_path[1] not in dr:
							if err < naive_naive[2]:  # naive -> naive
								naive_naive[2] = err
						if naive_path[0] not in fe and naive_path[1] not in dr:  # naive -> opt
							if err < naive_opt[2]:
								naive_opt[2] = err
					else:  # opt -> and agnostic ->
						if naive_path[0] not in fe and naive_path[1] not in dr:
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

	for run in range(2, 7):
		obj = pickle.load(
			open(results_home + 'intermediate/' + type2 + '/' + type2 + '_' + data_name + '_run_' +
				 str(run) + '_full_final.pkl', 'rb'), encoding='latin1')

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
			# fe = obj.paths[i][0]
			# dr = obj.paths[i][1]
			# la = obj.paths[i][2]
			fe = pi.feature_extraction
			dr = pi.dimensionality_reduction
			la = pi.learning_algorithm
			# errs = []
			# for j in range(len(pi)):
			# 	errs.append(pi[j].get_error())
			# err = np.amin(errs)
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

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_opt[i] - opt_opt[i]
			if e[i] > 10000:
				e[i] = 0
		E1[run - 2, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = agnostic_naive[i] - opt_naive[i]
			if e[i] > 10000:
				e[i] = 0
		E2[run - 2, :] = np.asarray(e)

		e = [0] * 3
		for i in range(3):
			e[i] = naive_naive[i] - naive_opt[i]
			if e[i] > 10000:
				e[i] = 0
		E3[run - 2, :] = np.asarray(e)

	alpha = np.zeros(E1.shape)
	gamma = np.zeros(E1.shape)
	E_propagation = np.zeros(E1.shape)
	# for i in range(3):
	# 	E11 = E1[:, i]
	# 	E22 = E2[:, i]
	# 	E33 = E3[:, i]
	# 	for j in range(len(E11)):
	# 		e1 = E11[j]
	# 		e2 = E22[j]
	# 		e3 = E33[j]
	# 		if e3 == 0:
	# 			g = 0
	# 			a = e1
	# 		else:
	# 			a = e1 * e3 / (e2 + e3 - e1)
	# 			g = (e2 - e1) / e3
	# 		alpha[j, i] = a
	# 		gamma[j, i] = g
	# 		E_propagation[j, i] = a * g

	for i in range(2):
		E11 = E1[:, i]
		E22 = E2[:, i]
		E33 = E3[:, i]
		for j in range(len(E11)):
			e1 = E11[j]
			e2 = E22[j]
			e3 = E33[j]
			a = e1 * e3 / (e3 + e2 - e1)
			g = (e2 - e1) / e3
			alpha[j, i] = a
			gamma[j, i] = g
			E_propagation[j, i] = a * g


	E_total = np.mean(E1, 0)
	E_total_std = np.std(E1, 0)

	E_direct = np.mean(alpha, axis=0)
	E_direct_std = np.std(alpha, axis=0)
	E_direct[2] = E_total[2]
	E_direct_std[2] = E_total_std[2]

	E_propagation_mean = np.mean(E_propagation, axis=0)
	E_propagation_std = np.std(E_propagation, axis=0)
	E_propagation_mean[2] = 0
	E_propagation_std[2] = 0

	gamma_mean = np.mean(gamma, 0)
	gamma_std = np.std(gamma, 0)
	gamma_mean[2] = 0
	gamma_std[2] = 0

	gammas[z, :] = gamma_mean
	E_totals[z, :] = E_total
	E_propagations[z, :] = E_propagation_mean
	E_directs[z, :] = E_direct

E_direct_random = E_directs
E_propagation_random = E_propagations
E_total_random = E_totals
gamma_random = gammas



from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = ['Feature extraction', 'Feature transformation', 'Learning algorithms']
x1 = np.asarray([1, 2, 3])

_, axs = plt.subplots(nrows=1, ncols=1)
axs.errorbar(x1, np.mean(E_propagation_grid, 0), np.std(E_propagation_grid, 0), linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, np.mean(E_propagation_grid, 0), color='r')
axs.errorbar(x1, np.mean(E_propagation_random, 0), np.std(E_propagation_random, 0), linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs.plot(x1, np.mean(E_propagation_random, 0), color='b')
plt.title('Error propagation in steps')
plt.xlabel('Steps')
plt.ylabel('Error')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_steps_comparison_matsc_dataset2.eps')
plt.close()

_, axs = plt.subplots(nrows=1, ncols=1)
axs.errorbar(x1, np.mean(gamma_grid, 0), np.std(gamma_grid, 0), linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, np.mean(gamma_grid, 0), color='r')
axs.errorbar(x1, np.mean(gamma_random, 0), np.std(gamma_random, 0), linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs.plot(x1, np.mean(gamma_random, 0), color='b')
plt.title('Propagation factor in steps')
plt.xlabel('Steps')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/propagation_factor_steps_comparison_matsc_dataset2.eps')
plt.close()
