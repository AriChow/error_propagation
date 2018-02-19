import pickle
import os
import numpy as np
import sys
import copy
from matplotlib import pyplot as plt


home = os.path.expanduser('~')
datasets = ['breast', 'matsc_dataset1', 'matsc_dataset2', 'brain', 'bone']
colors = ['b', 'k', 'y', 'r', 'g']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']





# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
step = []
algt1 = []
alg1 = []
step_std = []
alg_std = []
alg1_std = []
for id, data_name in enumerate(datasets):
	step_error = np.zeros((stop-1, 3))
	alg_error = np.zeros((stop-1, 7))
	alg1_error = np.zeros((stop-1, 7))
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

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

			p = pipeline['all']
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
	random1_step_error = np.expand_dims(step_error.astype('float32'), 0)
	random1_std_error = np.expand_dims(std_error.astype('float32'), 0)
	std_error = np.std(alg_error, 0)
	step_error = np.mean(alg_error, 0)
	random1_alg_error = np.expand_dims(step_error.astype('float32'), 0)
	random1_alg_std_error = np.expand_dims(std_error.astype('float32'), 0)
	std_error = np.std(alg1_error, 0)
	step_error = np.mean(alg1_error, 0)
	random1_alg1_error = np.expand_dims(step_error.astype('float32'), 0)
	random1_alg1_std_error = np.expand_dims(std_error.astype('float32'), 0)

	if id == 0:
		step = random1_step_error
		step_std = random1_std_error
		algt1 = random1_alg_error
		alg_std = random1_alg_std_error
		alg1 = random1_alg1_error
		alg1_std = random1_alg1_std_error
	else:
		step = np.vstack((step, random1_step_error))
		step_std = np.vstack((step_std, random1_std_error))
		algt1 = np.vstack((algt1, random1_alg_error))
		alg_std = np.vstack((alg_std, random1_alg_std_error))
		alg1 = np.vstack((alg1, random1_alg1_error))
		alg1_std = np.vstack((alg1_std, random1_alg1_std_error))

x = steps
plt.figure()
x1 = range(1, 4)
for i, data_name in enumerate(datasets):
	plt.errorbar(x1, step[i, :], step_std[i, :], linestyle='None', marker='*', color=colors[i], capsize=3)
	plt.plot(x1, step[i, :], colors[i] + '*-', label=data_name)
plt.legend()
plt.title('Step errors')
plt.xlabel('Agnostic step')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_step_datasets.jpg')
plt.close()

x = pipeline['all']
plt.figure()
x1 = range(1, 8)
for i, data_name in enumerate(datasets):
	plt.errorbar(x1, algt1[i, :], alg_std[i, :], linestyle='None', marker='*', color=colors[i], capsize=3)
	plt.plot(x1, algt1[i, :], colors[i] + '*-', label=data_name)
plt.legend()
plt.title('Algorithm error contributions (method 2)')
plt.xlabel('Agnostic algorithms')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_algorithm_type1_datasets.jpg')
plt.close()

x = pipeline['all']
plt.figure()
x1 = range(1, 8)
for i, data_name in enumerate(datasets):
	plt.errorbar(x1, alg1[i, :], alg1_std[i, :], linestyle='None', marker='*', color=colors[i], capsize=3)
	plt.plot(x1, alg1[i, :], colors[i] + '*-', label=data_name)
plt.legend()
plt.title('Algorithm error contributions (method 1)')
plt.xlabel('Agnostic algorithm')
plt.ylabel('Errors')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/agnostic_errors_algorithm_type2_datasets.jpg')
plt.close()



