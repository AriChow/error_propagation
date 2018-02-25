import pickle
import os
import numpy as np
import copy
from matplotlib import pyplot as plt


home = os.path.expanduser('~')
datasets = ['breast', 'matsc_dataset1', 'matsc_dataset2', 'brain']
colors = ['b', 'k', 'y', 'r', 'g']
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
steps = ['feature_extraction', 'dimensionality_reduction', 'learning_algorithm']
pipeline['feature_extraction'] = ["haralick", "VGG", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']

# Grid search
start = 1
stop = 2
type1 = 'grid_MCMC'
grid_times = []
for data_name in datasets:
	times = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

		t = 0
		if type(obj.times) == list:
			t = sum(obj.times)
		else:
			for k in obj.times.keys():
				t += sum(obj.times[k])
		times.append(t)
	grid_times.append(times)

# Random
start = 1
stop = 6
type1 = 'random_MCMC'
random_times = []
for data_name in datasets:
	times = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

		times.append(sum(obj.times))
	random_times.append(times)

# Random1
start = 1
stop = 6
type1 = 'random_MCMC'
random1_times = []
for data_name in datasets:
	times = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		times.append(obj.times[-1])
	random1_times.append(times)

# Bayesian
start = 1
stop = 6
type1 = 'bayesian_MCMC'
bayesian_times = []
for data_name in datasets:
	times = []
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_parallel.pkl', 'rb'), encoding='latin1')

		times.append(sum(obj.times))
	bayesian_times.append(times)

# Bayesian1
start = 1
stop = 6
type1 = 'bayesian_MCMC'
bayesian1_times = []
for data_name in datasets:
	times = []
	if data_name == 'bone':
		stop = 3
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl', 'rb'), encoding='latin1')

		times.append(obj.times[-1])
	bayesian1_times.append(times)

step = np.zeros((len(datasets), 5))
step_std = np.zeros((len(datasets), 5))

for i in range(len(datasets)):
	step[i, :] = np.asarray([np.mean(grid_times[i]), np.mean(random_times[i]), np.mean(random1_times[i]), \
							 np.mean(bayesian_times[i]), np.mean(bayesian1_times[i])])
	step_std[i, :] = np.asarray([np.std(grid_times[i]), np.std(random_times[i]), np.std(random1_times[i]), \
							 np.std(bayesian_times[i]), np.std(bayesian1_times[i])])

x = ['Grid', 'Random(path)', 'Random(pipeline)']
plt.figure(figsize=(10, 5))
x1 = range(1, 4)
for i, data_name in enumerate(datasets):
	plt.errorbar(x1, step[i, :3], step_std[i, :3], linestyle='None', marker='*', color=colors[i], capsize=3)
	plt.plot(x1, step[i, :3], colors[i] + '*-', label=data_name)
plt.legend()
plt.title('Time comparison of algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Time(s)')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/times_datasets_no_bayesian.jpg')
plt.close()

x = ['Grid', 'Random(path)', 'Random(pipeline)']
plt.figure(figsize=(10, 5))
x1 = range(1, 4)
plt.errorbar(x1, np.mean(step[:, :3], axis=0), np.std(step[:, :3], axis=0), linestyle='None', marker='*', capsize=3)
plt.plot(x1, np.mean(step[:, :3], axis=0))
plt.title('Time comparison of algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Time(s)')
plt.xticks(x1, x)
plt.savefig(results_home + 'figures/times_algorithms_no_bayesian.jpg')
plt.close()
