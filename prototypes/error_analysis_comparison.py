import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt


home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']


# Grid
start = 1
stop = 2
type1 = 'grid_MCMC'
step_error = np.zeros((stop-1, 3))
alg_error = np.zeros((stop-1, 7))
alg1_error = np.zeros((stop-1, 7))
for run in range(start, stop):
	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '.pkl', 'rb'), encoding='latin1')

	min_err = 100000000
	min_fe = [1000000] * len(pipeline['feature_extraction'])
	min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
	min_la = [1000000] * len(pipeline['learning_algorithm'])
	min_all = [100000] * len(pipeline['all'])
	min_alls = [0] * len(pipeline['all'])
	s = [0] * len(pipeline['all'])
	pipelines = obj.pipelines
	error_curves = []
	for i in range(len(pipelines)):
		p = pipelines[i]
		objects = []
		for j in range(len(p)):
			objects.append(p[j].get_error())
		error_curves.append(objects)

	for i in range(len(error_curves)):
		err = error_curves[i]
		if np.amin(err) < min_err:
			min_err = np.amin(err)
		p = pipeline[steps[0]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][0] == alg:
				if np.amin(err) < min_fe[j]:
					min_fe[j] = np.amin(err)

		p = pipeline[steps[1]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][1] == alg:
				if np.amin(err) < min_dr[j]:
					min_dr[j] = np.amin(err)

		p = pipeline[steps[2]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][2] == alg:
				if np.amin(err) < min_la[j]:
					min_la[j] = np.amin(err)

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			if alg in obj.paths[i]:
				min_alls[j] += np.amin(err)
				s[j] += 1
			else:
				if np.amin(err) < min_all[j]:
					min_all[j] = np.amin(err)

	for j in range(len(s)):
		min_alls[j] /= s[j]

	errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
	errors = np.asarray(errors)
	step_error[run - 1, :] = errors
	min_alls = np.asarray(min_alls)
	min_all = np.asarray(min_all)
	alg1_error[run - 1, :] = min_all - np.asarray([min_err] * 7)
	alg_error[run - 1, :] = min_alls - np.asarray([min_err] * 7)

std_error = np.std(step_error, 0)
step_error = np.mean(step_error, 0)
grid_step_error = step_error.astype('float32')
grid_std_error = std_error.astype('float32')
std_error = np.std(alg_error, 0)
step_error = np.mean(alg_error, 0)
grid_alg_error = step_error.astype('float32')
grid_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
grid_alg1_error = step_error.astype('float32')
grid_alg1_std_error = std_error.astype('float32')


ids = np.argsort(grid_alg_error)
grid_alg_error = grid_alg_error[ids]
grid_alg_std_error = grid_alg_std_error[ids]
steps1 = []
for i in range(len(ids)):
	steps1.append(pipeline['all'][ids[i]])
pipeline['all'] = copy.deepcopy(steps1)

ids = np.argsort(grid_alg1_error)
grid_alg1_error = grid_alg1_error[ids]
grid_alg1_std_error = grid_alg1_std_error[ids]
steps1 = []
for i in range(len(ids)):
	steps1.append(pipeline['all'][ids[i]])
pipeline_all = copy.deepcopy(steps1)

ids = np.argsort(grid_step_error)
grid_step_error = grid_step_error[ids]
grid_std_error = grid_std_error[ids]
steps1 = []
for i in range(len(ids)):
	steps1.append(steps[ids[i]])
steps = copy.deepcopy(steps1)

# Random
start = 1
stop = 6
type1 = 'random_MCMC'
step_error = np.zeros((stop-1, 3))
alg_error = np.zeros((stop-1, 7))
alg1_error = np.zeros((stop-1, 7))
for run in range(start, stop):
	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '.pkl', 'rb'), encoding='latin1')

	min_err = 100000000
	min_fe = [1000000] * len(pipeline[steps[0]])
	min_dr = [1000000] * len(pipeline[steps[1]])
	min_la = [1000000] * len(pipeline[steps[2]])
	min_all = [1000000] * len(pipeline_all)
	min_alls = [0] * len(pipeline['all'])
	s = [0] * len(pipeline['all'])
	pipelines = obj.pipelines
	error_curves = []
	for i in range(len(pipelines)):
		p = pipelines[i]
		objects = []
		for j in range(len(p)):
			objects.append(p[j].get_error())
		error_curves.append(objects)

	for i in range(len(error_curves)):
		err = error_curves[i]
		if np.amin(err) < min_err:
			min_err = np.amin(err)
		p = pipeline[steps[0]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][ids[0]] == alg:
				if np.amin(err) < min_fe[j]:
					min_fe[j] = np.amin(err)

		p = pipeline[steps[1]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][ids[1]] == alg:
				if np.amin(err) < min_dr[j]:
					min_dr[j] = np.amin(err)

		p = pipeline[steps[2]]
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][ids[2]] == alg:
				if np.amin(err) < min_la[j]:
					min_la[j] = np.amin(err)

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			if alg in obj.paths[i]:
				min_alls[j] += np.amin(err)
				s[j] += 1

		p = pipeline_all
		for j in range(len(p)):
			alg = p[j]
			if alg not in obj.paths[i]:
				if np.amin(err) < min_all[j]:
					min_all[j] = np.amin(err)
	for j in range(len(s)):
		min_alls[j] /= s[j]

	errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
	errors = np.asarray(errors)
	step_error[run - 1, :] = errors
	min_alls = np.asarray(min_alls)
	min_all = np.asarray(min_all)
	alg_error[run - 1, :] = min_alls - np.asarray([min_err] * 7)
	alg1_error[run - 1, :] = min_all - np.asarray([min_err] * 7)

std_error = np.std(step_error, 0)
step_error = np.mean(step_error, 0)
random_step_error = step_error.astype('float32')
random_std_error = std_error.astype('float32')
std_error = np.std(alg_error, 0)
step_error = np.mean(alg_error, 0)
random_alg_error = step_error.astype('float32')
random_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
random_alg1_error = step_error.astype('float32')
random_alg1_std_error = std_error.astype('float32')

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
step_error = np.zeros((stop-1, 3))
alg_error = np.zeros((stop-1, 7))
alg_error = np.zeros((stop-1, 7))
for run in range(start, stop):
	obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'), encoding='latin1')

	min_err = 100000000
	min_fe = [1000000] * len(pipeline[steps[0]])
	min_dr = [1000000] * len(pipeline[steps[1]])
	min_la = [1000000] * len(pipeline[steps[2]])
	min_all = [1000000] * len(pipeline_all)
	min_alls = [[]] * len(pipeline['all'])
	s = [0] * len(pipeline['all'])
	pipelines = obj.pipelines

	for i in range(len(pipelines)):
		pi = pipelines[i]
		pie = pi.get_error()
		if pie < min_err:
			min_err = pie
		p = pipeline[steps[0]]
		for j in range(len(p)):
			alg = p[j]
			p1 = ''
			if steps[0] == 'feature_extraction':
				p1 = pi.feature_extraction.decode('latin1')
			elif steps[0] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction.decode('latin1')
			elif steps[0] == 'learning_algorithm':
				p1 = pi.learning_algorithm.decode('latin1')
			if p1 == alg:
				if pie < min_fe[j]:
					min_fe[j] = pie
		p = pipeline[steps[1]]
		for j in range(len(p)):
			alg = p[j]
			p1 = ''
			if steps[1] == 'feature_extraction':
				p1 = pi.feature_extraction.decode('latin1')
			elif steps[1] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction.decode('latin1')
			elif steps[1] == 'learning_algorithm':
				p1 = pi.learning_algorithm.decode('latin1')
			if p1 == alg:
				if pie < min_dr[j]:
					min_dr[j] = pie
		p = pipeline[steps[2]]
		for j in range(len(p)):
			alg = p[j]
			p1 = ''
			if steps[2] == 'feature_extraction':
				p1 = pi.feature_extraction.decode('latin1')
			elif steps[2] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction.decode('latin1')
			elif steps[2] == 'learning_algorithm':
				p1 = pi.learning_algorithm.decode('latin1')
			if p1 == alg:
				if pie < min_la[j]:
					min_la[j] = pie

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			m = copy.deepcopy(min_alls[j])
			if pi.learning_algorithm.decode('latin1') == alg or pi.feature_extraction.decode('latin1') == alg or pi.dimensionality_reduction.decode('latin1') == alg:
				m.append((pie, pi))
			min_alls[j] = copy.deepcopy(m)

		p = pipeline_all
		for j in range(len(p)):
			alg = p[j]
			if pi.learning_algorithm.decode('latin1') == alg or pi.feature_extraction.decode('latin1') == alg or pi.dimensionality_reduction.decode('latin1') == alg:
				continue
			else:
				if pie < min_all[j]:
					min_all[j] = pie
	min_alls1 = [[]] * len(pipeline['all'])
	for j in range(len(min_alls)):
		algs = min_alls[j]
		flag = [0] * len(algs)
		for k in range(len(algs)-1):
			if flag[k] == 1:
				continue
			else:
				m1 = copy.deepcopy(min_alls1[j])
				m = [algs[k][0]]
				flag[k] = 1
				pk = algs[k][1]
				for u in range(k+1, len(algs)):
					if flag[u] == 1:
						continue
					else:
						pu = algs[u][1]
						if pk.feature_extraction.decode('latin1') == pu.feature_extraction.decode('latin1') and \
										pk.dimensionality_reduction.decode('latin1') == \
										pu.dimensionality_reduction.decode('latin1') and \
										pk.learning_algorithm.decode('latin1') == pu.learning_algorithm.decode('latin1'):
							m.append(algs[u][0])
							flag[u] = 1
				m1.append(np.amin(m))
				min_alls1[j] = copy.deepcopy(m1)
	min_alls = [0] * len(pipeline['all'])
	for i in range(len(min_alls1)):
		min_alls[i] = np.mean(min_alls1[i])
	errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
	errors = np.asarray(errors)
	step_error[run - 1, :] = errors
	min_alls = np.asarray(min_alls)
	min_all = np.asarray(min_all)
	alg_error[run - 1, :] = min_alls - np.asarray([min_err] * 7)
	alg1_error[run - 1, :] = min_all - np.asarray([min_err] * 7)

std_error = np.std(step_error, 0)
step_error = np.mean(step_error, 0)
random1_step_error = step_error.astype('float32')
random1_std_error = std_error.astype('float32')
std_error = np.std(alg_error, 0)
step_error = np.mean(alg_error, 0)
random1_alg_error = step_error.astype('float32')
random1_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
random1_alg1_error = step_error.astype('float32')
random1_alg1_std_error = std_error.astype('float32')

x = steps
plt.figure()
x1 = range(1, 4)
plt.errorbar(x1, grid_step_error, grid_std_error, linestyle='None', marker='o', color='r', capsize=3)
plt.plot(x1, grid_step_error, 'ro-', label='grid search')
plt.errorbar(x1, random_step_error, random_std_error, linestyle='None', marker='^', color='b', capsize=3)
plt.plot(x1, random_step_error, 'b^-', label='random search(type1)')
plt.errorbar(x1, random1_step_error, random1_std_error, linestyle='None', marker='*', color='g', capsize=3)
plt.plot(x1, random1_step_error, 'g*-', label='random search(type2)')
plt.legend()
plt.title('Step errors')
plt.xlabel('Agnostic step')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_step_' + data_name + '.jpg')
plt.close()

x = pipeline['all']
plt.figure()
x1 = range(1, 8)
plt.errorbar(x1, grid_alg_error, grid_alg_std_error, linestyle='None', marker='o', color='r', capsize=3)
plt.plot(x1, grid_alg_error, 'ro-', label='grid search')
plt.errorbar(x1, random_alg_error, random_alg_std_error, linestyle='None', marker='^', color='b', capsize=3)
plt.plot(x1, random_alg_error, 'b^-', label='random search(type1)')
plt.errorbar(x1, random1_alg_error, random1_alg_std_error, linestyle='None', marker='*', color='g', capsize=3)
plt.plot(x1, random1_alg_error, 'g*-', label='random search(type2)')
plt.legend()
plt.title('Algorithm error contributions')
plt.xlabel('Agnostic algorithm')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_algorithm_type2_' + data_name + '.jpg')
plt.close()

x = pipeline_all
plt.figure()
x1 = range(1, 8)
plt.errorbar(x1, grid_alg1_error, grid_alg1_std_error, linestyle='None', marker='o', color='r', capsize=3)
plt.plot(x1, grid_alg1_error, 'ro-', label='grid search')
plt.errorbar(x1, random_alg1_error, random_alg1_std_error, linestyle='None', marker='^', color='b', capsize=3)
plt.plot(x1, random_alg1_error, 'b^-', label='random search(type1)')
plt.errorbar(x1, random1_alg1_error, random1_alg1_std_error, linestyle='None', marker='*', color='g', capsize=3)
plt.plot(x1, random1_alg1_error, 'g*-', label='random search(type2)')
plt.legend()
plt.title('Algorithm error contributions')
plt.xlabel('Agnostic algorithm')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_algorithm_type1_' + data_name + '.jpg')
plt.close()