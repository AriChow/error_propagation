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
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
step_error = np.zeros((stop-1, 3))
alg_error = np.zeros((stop-1, 10))
alg1_error = np.zeros((stop-1, 10))
for run in range(start, stop):
	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full_naive.pkl', 'rb'), encoding='latin1')

	min_err = 100000000
	min_fe = [1000000] * len(pipeline[steps[0]])
	min_dr = [1000000] * len(pipeline[steps[1]])
	min_la = [1000000] * len(pipeline[steps[2]])
	min_all = [1000000] * len(pipeline['all'])
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
				p1 = pi.feature_extraction
			elif steps[0] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction
			elif steps[0] == 'learning_algorithm':
				p1 = pi.learning_algorithm
			if p1 == alg:
				if pie < min_fe[j]:
					min_fe[j] = pie
		p = pipeline[steps[1]]
		for j in range(len(p)):
			alg = p[j]
			p1 = ''
			if steps[1] == 'feature_extraction':
				p1 = pi.feature_extraction
			elif steps[1] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction
			elif steps[1] == 'learning_algorithm':
				p1 = pi.learning_algorithm
			if p1 == alg:
				if pie < min_dr[j]:
					min_dr[j] = pie
		p = pipeline[steps[2]]
		for j in range(len(p)):
			alg = p[j]
			p1 = ''
			if steps[2] == 'feature_extraction':
				p1 = pi.feature_extraction
			elif steps[2] == 'dimensionality_reduction':
				p1 = pi.dimensionality_reduction
			elif steps[2] == 'learning_algorithm':
				p1 = pi.learning_algorithm
			if p1 == alg:
				if pie < min_la[j]:
					min_la[j] = pie

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			m = copy.deepcopy(min_alls[j])
			if pi.learning_algorithm == alg or pi.feature_extraction == alg or pi.dimensionality_reduction == alg:
				m.append((pie, pi))
			min_alls[j] = copy.deepcopy(m)

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			if pi.learning_algorithm == alg or pi.feature_extraction == alg or pi.dimensionality_reduction == alg:
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
						if pk.feature_extraction == pu.feature_extraction and \
										pk.dimensionality_reduction == \
										pu.dimensionality_reduction and \
										pk.learning_algorithm == pu.learning_algorithm:
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
	alg_error[run - 1, :] = min_alls - np.asarray([min_err] * 10)
	alg1_error[run - 1, :] = min_all - np.asarray([min_err] * 10)

naive_error = alg_error[:, [3, 6, 9]]
step_propagation_error = step_error - naive_error
alg_propagation_error = np.zeros((5, 10))
for i in range(4):
	alg_propagation_error[:, i] = alg_error[:, i] - naive_error[:, 0]
for i in range(4, 7):
	alg_propagation_error[:, i] = alg_error[:, i] - naive_error[:, 1]
for i in range(7, 10):
	alg_propagation_error[:, i] = alg_error[:, i] - naive_error[:, 2]

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

std_error = np.std(step_propagation_error, 0)
step_error = np.mean(step_propagation_error, 0)
random1_step_propagation_error = step_error.astype('float32')
random1_step_propagation_std_error = std_error.astype('float32')

std_error = np.std(alg_propagation_error, 0)
step_error = np.mean(alg_propagation_error, 0)
random1_alg1_propagation_error = step_error.astype('float32')
random1_alg1_propagation_std_error = std_error.astype('float32')

ids = np.argsort(random1_alg_error)
random1_alg_error = random1_alg_error[ids]
random1_alg_std_error = random1_alg_std_error[ids]
steps1 = []
for i in range(len(ids)):
	steps1.append(pipeline['all'][ids[i]])
pipeline['all'] = copy.deepcopy(steps1)

ids = np.argsort(random1_alg1_error)
random1_alg1_error = random1_alg1_error[ids]
random1_alg1_std_error = random1_alg1_std_error[ids]
random1_alg1_propagation_error = random1_alg1_propagation_error[ids]
random1_alg1_propagation_std_error = random1_alg1_propagation_std_error[ids]
# steps1 = []
# for i in range(len(ids)):
# 	steps1.append(pipeline['all'][ids[i]])
# pipeline_all = copy.deepcopy(steps1)
#
# ids = np.argsort(random1_step_error)
# random1_step_error = random1_step_error[ids]
# random1_std_error = random1_std_error[ids]
# random1_step_propagation_error = random1_step_propagation_error[ids]
# random1_step_propagation_std_error = random1_step_propagation_std_error[ids]
# steps1 = []
# for i in range(len(ids)):
# 	steps1.append(steps[ids[i]])
# steps = copy.deepcopy(steps1)

x = steps
plt.figure()
x1 = range(1, 4)
plt.errorbar(x1, random1_step_error, random1_std_error, linestyle='None', marker='^', color='b', capsize=3)
plt.plot(x1, random1_step_error, 'b^-', label='random search(total error)')
plt.errorbar(x1, random1_step_propagation_error, random1_step_propagation_std_error, linestyle='None', marker='*', color='g', capsize=3)
plt.plot(x1, random1_step_propagation_error, 'g*-', label='random search(actual error)')
plt.legend()
plt.title('Step errors')
plt.xlabel('Agnostic step')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_propagation_step_' + data_name + '.jpg')
plt.close()

x = pipeline['all']
plt.figure(figsize=[20, 10])
x1 = range(1, 11)
plt.errorbar(x1, random1_alg_error, random1_alg_std_error, linestyle='None', marker='^', color='b', capsize=3)
plt.plot(x1, random1_alg_error, 'b^-', label='random search(total error)')
plt.errorbar(x1, random1_alg1_propagation_error, random1_alg1_propagation_std_error, linestyle='None', marker='*', color='g', capsize=3)
plt.plot(x1, random1_alg1_propagation_error, 'g*-', label='random search(actual error)')
plt.legend()
plt.title('Algorithm error contributions')
plt.xlabel('Agnostic algorithm')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_propagation_algorithm_type1_' + data_name + '.jpg')
plt.close()
