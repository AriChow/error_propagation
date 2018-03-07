'''
This file is for quantification of error contributions from
different parts of an ML pipeline, namely the computational steps; 
and algorithms.
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
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']


# # Random
# start = 1
# stop = 6
# type1 = 'random_MCMC'
# step_E1_error = np.zeros((stop - 1, 3))
# alg_E1_error = np.zeros((stop - 1, 7))
# alg1_E1_error = np.zeros((stop - 1, 7))
# step_E2_error = np.zeros((stop - 1, 3))
# step_E3_error = np.zeros((stop - 1, 3))
# for run in range(start, stop):
# 	obj = pickle.load(
# 		open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
# 			 str(run) + '_naive.pkl', 'rb'), encoding='latin1')
#
# 	pipelines = obj.pipelines
# 	error_curves = []
# 	for i in range(len(pipelines)):
# 		p = pipelines[i]
# 		objects = []
# 		for j in range(len(p)):
# 			objects.append(p[j].get_error())
# 		error_curves.append(objects)
# 	paths = obj.paths
#
# 	naive_min_err = np.amin(error_curves[-1])
#
# 	all_errs = []
# 	for i in range(len(paths)):
# 		path = paths[i]
# 		if 'naive_feature_extraction' in path or 'naive_dimensionaity_reduction' in path or 'naive_learning_algorithm' in path:
# 			continue
# 		else:
# 			all_errs += error_curves[i]
# 	min_err = np.amin(all_errs)
#
# 	min_fe = [100000] * len(pipeline['feature_extraction'])
# 	min_dr = [100000] * len(pipeline['dimensionality_reduction'])
# 	min_la = [100000] * len(pipeline['learning_algorithm'])
# 	min_all = [100000] * len(pipeline['all'])
# 	min_all1 = [100000] * len(pipeline['all'])
# 	min_alls = [0] * len(pipeline['all'])
# 	naive_opt = [100000] * 3
# 	opt_naive = [100000] * 3
# 	agnostic_naive = [0] * 3
# 	s = [0] * len(pipeline['all'])
# 	s_agnostic = [0] * 3
#
# 	for i in range(len(paths)):
# 		path = paths[i]
# 		err = error_curves[i]
# 		if 'naive_feature_extraction' in path or 'naive_dimensionality_reduction' in path or 'naive_learning_algorithm' in path:
# 			if 'naive' in path[0]:
# 				if np.amin(err) < naive_opt[0]:
# 					naive_opt[0] = np.amin(err)
# 			elif 'naive' in path[1] and 'naive' in path[2]:
# 				if np.amin(err) < opt_naive[0]:
# 					opt_naive[0] = np.amin(err)
# 				agnostic_naive[0] += np.amin(err)
# 				s_agnostic[0] += 1
# 			if 'naive' in path[1]:
# 				if np.amin(err) < naive_opt[1]:
# 					naive_opt[1] = np.amin(err)
# 			elif 'naive' in path[0] and 'naive' in path[2]:
# 				if np.amin(err) < opt_naive[1]:
# 					opt_naive[1] = np.amin(err)
# 				agnostic_naive[1] += np.amin(err)
# 				s_agnostic[1] += 1
# 			if 'naive' in path[2]:
# 				if np.amin(err) < naive_opt[2]:
# 					naive_opt[2] = np.amin(err)
# 			elif 'naive' in path[0] and 'naive' in path[1]:
# 				if np.amin(err) < opt_naive[2]:
# 					opt_naive[2] = np.amin(err)
# 				agnostic_naive[2] += np.amin(err)
# 				s_agnostic[2] += 1
# 		else:
# 			p = pipeline['feature_extraction']
# 			for j in range(len(p)):
# 				alg = p[j]
# 				if path[0] == alg:
# 					if np.amin(err) < min_fe[j]:
# 						min_fe[j] = np.amin(err)
#
# 			p = pipeline['dimensionality_reduction']
# 			for j in range(len(p)):
# 				alg = p[j]
# 				if path[1] == alg:
# 					if np.amin(err) < min_dr[j]:
# 						min_dr[j] = np.amin(err)
#
# 			p = pipeline['learning_algorithm']
# 			for j in range(len(p)):
# 				alg = p[j]
# 				if path[2] == alg:
# 					if np.amin(err) < min_la[j]:
# 						min_la[j] = np.amin(err)
#
# 			p = pipeline['all']
# 			for j in range(len(p)):
# 				alg = p[j]
# 				if alg not in obj.paths[i]:
# 					if np.amin(err) < min_all[j]:
# 						min_all[j] = np.amin(err)
# 				else:
# 					if np.amin(err) < min_all1[j]:
# 						min_all1[j] = np.amin(err)
# 					min_alls[j] += np.amin(err)
# 					s[j] += 1
# 		for j in range(len(s)):
# 			if s[j] == 0:
# 				min_alls[j] = min_all1[j]
# 			else:
# 				min_alls[j] /= s[j]
# 		for j in range(len(s_agnostic)):
# 			if s_agnostic[j] == 0:
# 				agnostic_naive[j] = 0
# 			else:
# 				agnostic_naive[j] /= s_agnostic[j]
#
# 	# E1 (steps and algorithms)
# 	E1 = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]  # stores the E_agnostic_opt for each step of the pipeline
# 	step_E1_error[run-1, :] = np.asarray(E1)
# 	E1 = np.asarray(min_all) - np.asarray([min_err]*3)
# 	alg_E1_error[run-1, :] = E1
# 	E1 = np.asarray(min_alls) - np.asarray(min_all1)
# 	alg1_E1_error[run-1, :] = E1
#
# 	# E2 (only for steps): agnostic_naive - opt_naive
# 	E2 = np.asarray(agnostic_naive) - np.asarray(opt_naive)
# 	step_E2_error[run-1, 0] = E2
#
# 	# E3 (only for algorithms):naive_naive - naive_opt
# 	E2 = np.asarray([naive_min_err] * 3) - np.asarray(naive_opt)
# 	step_E3_error[run - 1, 0] = E2
#
#
# alpha = np.zeros((step_E1_error.shape))
# gamma = np.zeros((step_E1_error.shape))
# E_propagation = np.zeros((step_E1_error.shape))
# for i in range(3):
# 	E1 = step_E1_error[:, i]
# 	E2 = step_E2_error[:, i]
# 	E3 = step_E3_error[:, i]
# 	for j in range(5):
# 		e1 = E1[j]
# 		e2 = E2[j]
# 		e3 = E3[j]
# 		a = e1 * e3 / (e2 + e3 - e1)
# 		g = (e2 - e1) / e3
# 		alpha[i, j] = a
# 		gamma[i, j] = g
# 		E_propagation[i, j] = a * g
#
#
# E_total = np.mean(step_E1_error, 0)
# E_total_std = np.std(step_E1_error, 0)
#
# E_direct = np.mean(alpha, axis=0)
# E_direct_std = np.std(alpha, axis=0)
#
# E_propagation_mean = np.mean(E_propagation, axis=0)
# E_propagation_std = np.std(E_propagation, axis=0)
#
# gamma_mean = np.mean(gamma, 0)
# gamma_std = np.std(gamma, 0)
#
# from matplotlib import pyplot as plt
#
# colors = ['b', 'g', 'y']
# steps = ['FE', 'DR', 'LA']
# x1 = np.asarray([1, 2, 3])
# w = 0.2
# d = w * np.ones(4)
# x2 = x1 + d
# plt.bar(x1, E_total.ravel(), width=w, align='center', color=colors[0], yerr=E_total_std.ravel(), label='Total error')
# plt.bar(x2, E_direct.ravel(), width=w, align='center', color=colors[1], yerr=E_direct_std.ravel(), label='Direct error')
# plt.bar(x2, E_propagation_mean.ravel(), width=w, bottom=E_direct.ravel(), align='center', color=colors[2], yerr=E_propagation_std.ravel(), label='Propagation')
# plt.title('Error quantification and propagation')
# plt.xlabel('Steps')
# plt.ylabel('Error contributions')
# plt.xticks(x1, steps)
# plt.legend()
# plt.autoscale()
# plt.savefig(results_home + 'figures/error_propagation_random_path' + data_name + '.jpg')
# plt.close()
#
# steps = ['FE', 'DR', 'LA']
# x1 = np.asarray([1, 2, 3])
# plt.bar(x1, gamma_mean.ravel(), width=w, align='center', color='r', yerr=gamma_std.ravel())
# plt.title('Propagation factor')
# plt.xlabel('Steps')
# plt.ylabel('Propagation factor')
# plt.xticks(x1, steps)
# plt.legend()
# plt.autoscale()
# plt.savefig(results_home + 'figures/propagation_factor_random_path' + data_name + '.jpg')
# plt.close()



# Random1
start = 1
stop = 2
type1 = 'random_MCMC'
step_E1_error = np.zeros((stop - 1, 3))
alg_E1_error = np.zeros((stop - 1, 7))
alg1_E1_error = np.zeros((stop - 1, 7))
step_E2_error = np.zeros((stop - 1, 3))
step_E3_error = np.zeros((stop - 1, 3))
for run in range(start, stop):
	obj = pickle.load(
		open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
			 str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')

	pipelines = obj.pipelines
	min_err = 1000000
	naive_min_err = [1000000] * 3
	min_fe = [100000] * len(pipeline['feature_extraction'])
	min_dr = [100000] * len(pipeline['dimensionality_reduction'])
	min_la = [100000] * len(pipeline['learning_algorithm'])
	min_all = [100000] * len(pipeline['all'])
	min_all1 = [100000] * len(pipeline['all'])
	min_alls = [0] * len(pipeline['all'])
	naive_opt = [100000] * 3
	opt_naive = [100000] * 3
	agnostic_naive = [0] * 3
	s = [0] * len(pipeline['all'])
	s_agnostic = [0] * 3

	for i in range(len(pipelines)):
		pi = pipelines[i]
		fe = pi.feature_extraction.decode('latin1')
		dr = pi.dimensionality_reduction.decode('latin1')
		la = pi.learning_algorithm.decode('latin1')
		path = [fe, dr, la]
		err = pi.get_error()
		if 'naive' in fe or 'naive' in dr or 'naive' in la:
			if 'naive' in fe and 'naive' in dr and 'naive' in la:
				if err < naive_min_err:
					naive_min_err = err
			if 'naive' in path[0] and 'naive' not in path[1] and 'naive' not in path[2]:
				if err < naive_opt[0]:
					naive_opt[0] = err
			elif 'naive' in path[1] and 'naive' in path[2]:
				if err < opt_naive[0]:
					opt_naive[0] = err
				agnostic_naive[0] += err
				s_agnostic[0] += 1
			if 'naive' in path[1] and 'naive' not in path[0] and 'naive' not in path[2]:
				if err < naive_opt[1]:
					naive_opt[1] = err
			elif 'naive' in path[0] and 'naive' in path[2]:
				if err < opt_naive[1]:
					opt_naive[1] = err
				agnostic_naive[1] += err
				s_agnostic[1] += 1
			if 'naive' in path[2]and 'naive' not in path[0] and 'naive' not in path[1]:
				if err < naive_opt[2]:
					naive_opt[2] = err
			elif 'naive' in path[0] and 'naive' in path[1]:
				if err < opt_naive[2]:
					opt_naive[2] = err
				agnostic_naive[2] += err
				s_agnostic[2] += 1
		else:
			p = pipeline['feature_extraction']
			if err < min_err:
				min_err = err
			for j in range(len(p)):
				alg = p[j]
				if path[0] == alg:
					if err < min_fe[j]:
						min_fe[j] = err

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if path[1] == alg:
					if err < min_dr[j]:
						min_dr[j] = err

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg:
					if err < min_la[j]:
						min_la[j] = err

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if alg not in path:
					if err < min_all[j]:
						min_all[j] = err
				else:
					if err < min_all1[j]:
						min_all1[j] = err
					min_alls[j] += err
					s[j] += 1
	for j in range(len(s)):
		if s[j] == 0:
			min_alls[j] = min_all1[j]
		else:
			min_alls[j] /= s[j]
	for j in range(len(s_agnostic)):
		if s_agnostic[j] == 0:
			agnostic_naive[j] = 0
		else:
			agnostic_naive[j] /= s_agnostic[j]
	# E1 (steps and algorithms)
	E1 = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]  # stores the E_agnostic_opt for each step of the pipeline
	step_E1_error[run-1, :] = np.asarray(E1)
	E1 = np.asarray(min_all) - np.asarray([min_err]*len(pipeline['all']))
	alg_E1_error[run-1, :] = E1
	E1 = np.asarray(min_alls) - np.asarray(min_all1)
	alg1_E1_error[run-1, :] = E1

	# E2 (only for steps): agnostic_naive - opt_naive
	E2 = np.asarray(agnostic_naive) - np.asarray(opt_naive)
	step_E2_error[run-1, :] = E2

	# E3 (only for algorithms):naive_naive - naive_opt
	E2 = np.asarray([naive_min_err] * 3) - np.asarray(naive_opt)
	step_E3_error[run - 1, :] = E2


alpha = np.zeros((step_E1_error.shape))
gamma = np.zeros((step_E1_error.shape))
E_propagation = np.zeros((step_E1_error.shape))
for i in range(3):
	E1 = step_E1_error[:, i]
	E2 = step_E2_error[:, i]
	E3 = step_E3_error[:, i]
	for j in range(len(E1)):
		e1 = E1[j]
		e2 = E2[j]
		e3 = E3[j]
		a = e1 * e3 / (e2 + e3 - e1)
		g = (e2 - e1) / e3
		alpha[j, i] = a
		gamma[j, i] = g
		E_propagation[j, i] = a * g


E_total = np.mean(step_E1_error, 0)
E_total_std = np.std(step_E1_error, 0)

E_direct = np.mean(alpha, axis=0)
E_direct_std = np.std(alpha, axis=0)

E_propagation_mean = np.mean(E_propagation, axis=0)
E_propagation_std = np.std(E_propagation, axis=0)

gamma_mean = np.mean(gamma, 0)
gamma_std = np.std(gamma, 0)

from matplotlib import pyplot as plt

colors = ['b', 'g', 'y']
steps = ['FE', 'DR', 'LA']
x1 = np.asarray([1, 2, 3])
w = 0.2
d = w * np.ones(3)
x2 = x1 + d
plt.bar(x1, E_total.ravel(), width=w, align='center', color=colors[0], yerr=E_total_std.ravel(), label='Total error')
plt.bar(x2, E_direct.ravel(), width=w, align='center', color=colors[1], yerr=E_direct_std.ravel(), label='Direct error')
plt.bar(x2, E_propagation_mean.ravel(), width=w, bottom=E_direct.ravel(), align='center', color=colors[2], yerr=E_propagation_std.ravel(), label='Propagation error')
plt.title('Error quantification and propagation')
plt.xlabel('Steps')
plt.ylabel('Error contributions')
plt.xticks(x1, steps)
plt.legend()
plt.autoscale()
plt.savefig(results_home + 'figures/error_propagation_random_pipeline_' + data_name + '.jpg')
plt.close()

steps = ['FE', 'DR', 'LA']
x1 = np.asarray([1, 2, 3])
plt.bar(x1, gamma_mean.ravel(), width=w, align='center', color='r', yerr=gamma_std.ravel())
plt.title('Propagation factor')
plt.xlabel('Steps')
plt.ylabel('Propagation factor')
plt.xticks(x1, steps)
plt.autoscale()
plt.savefig(results_home + 'figures/propagation_factor_random_pipeline_' + data_name + '.jpg')
plt.close()


