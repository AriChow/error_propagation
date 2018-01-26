import pickle
import os
import numpy as np
import matplotlib
import sys
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']
start = int(sys.argv[3])
stop = int(sys.argv[4])

# Bayesian
types = [sys.argv[2]]
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		if type1 == 'bayesian_MCMC':
			error_curves = obj.error_curves
		else:
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
			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if obj.paths[i][0] == alg:
					if np.amin(err) < min_fe[j]:
						min_fe[j] = np.amin(err)

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if obj.paths[i][1] == alg:
					if np.amin(err) < min_dr[j]:
						min_dr[j] = np.amin(err)

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if obj.paths[i][2] == alg:
					if np.amin(err) < min_la[j]:
						min_la[j] = np.amin(err)

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if alg not in obj.paths[i]:
					if np.amin(err) < min_all[j]:
						min_all[j] = np.amin(err)

		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
		errors = np.asarray(errors)
		step_error[run-1, z, :] = errors
		min_all = np.asarray(min_all)
		alg_error[run - 1, z, :] = min_all - np.asarray([min_err] * 7)
std_error = np.std(step_error, 0)
error = np.mean(step_error, 0)
step_error = step_error.astype('float32')
std_error = std_error.astype('float32')
x = ['Feature extraction', 'Dimensionality reduction', 'Learning algorithm']
plt.figure(1, figsize=(12, 4))

# plt.subplot(131)
# plt.errorbar(range(1, 4), error[0, :], std_error[:, :], linestyle='None', marker='^', capsize=3)
# plt.title('Random')
# plt.xlabel('Agnostic Step')
# plt.ylabel('Errors')
# plt.xticks(range(1, 4), x)

plt.subplot(121)
plt.errorbar(range(1, 4), error[0, :], std_error[0, :], linestyle='None', marker='^', capsize=3)
plt.axhline(y=0)
plt.title('MCMC_step_errors')
plt.xlabel('Agnostic step')
plt.ylabel('Errors')
plt.xticks(range(1, 4), x)

std_error = np.std(alg_error, 0)
error = np.mean(alg_error, 0)
step_error = step_error.astype('float32')
std_error = std_error.astype('float32')
x = pipeline['all']
plt.subplot(122)
plt.errorbar(range(1, 8), error[0, :], std_error[0, :], linestyle='None', marker='^', capsize=3)
plt.axhline(y=0)
plt.title('MCMC_algorithm_errors')
plt.xlabel('Agnostic algorithm')
plt.ylabel('Errors')
plt.xticks(range(1, 8), x)
# plt.subplot(133)
# plt.errorbar(range(1, 4), error[2, :], std_error[2, :], linestyle='None', marker='^', capsize=3)
# plt.title('MCMC')
# plt.xlabel('Agnostic Step')
# plt.ylabel('Errors')
# plt.xticks(range(1, 4), x)

plt.savefig(results_home + 'figures/agnostic_errors_' + types[0] + '_' + data_name + '.jpg')
plt.close()



