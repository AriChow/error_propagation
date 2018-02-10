import pickle
import os
import numpy as np
import matplotlib
import sys
# matplotlib.use('TkAgg')
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


# Bayesian
types = ['bayesian_MCMC']
start = 1
stop = 6
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
alg1_error = np.zeros((stop-1, 1, 7))
for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_parallel.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		min_all1 = [100000] * len(pipeline['all'])
		min_alls = [0] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])
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
				else:
					if np.amin(err) < min_all1[j]:
						min_all1[j] = np.amin(err)
					min_alls[j] += np.amin(err)
					s[j] += 1

		for j in range(len(s)):
			min_alls[j] /= s[j]
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
		errors = np.asarray(errors)
		step_error[run-1, z, :] = errors
		min_all = np.asarray(min_all)
		min_all1 = np.asarray(min_all1)
		min_alls = np.asarray(min_alls)
		alg_error[run - 1, z, :] = min_all - np.asarray([min_err] * 7)
		alg1_error[run - 1, z, :] = min_alls - min_all1
std_error = np.std(step_error, 0)
step_error = np.mean(step_error, 0)
bayesian_step_error = step_error.astype('float32')
bayesian_std_error = std_error.astype('float32')
std_error = np.std(alg_error, 0)
step_error = np.mean(alg_error, 0)
bayesian_alg_error = step_error.astype('float32')
bayesian_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
bayesian_alg1_error = step_error.astype('float32')
bayesian_alg1_std_error = std_error.astype('float32')


# Grid
start = 1
stop = 2
types = ['grid_MCMC']
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
alg1_error = np.zeros((stop-1, 1, 7))
for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		min_all1 = [100000] * len(pipeline['all'])
		min_alls = [0] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])
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
				else:
					if np.amin(err) < min_all1[j]:
						min_all1[j] = np.amin(err)
					min_alls[j] += np.amin(err)
					s[j] += 1

		for j in range(len(s)):
			min_alls[j] /= s[j]
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
		errors = np.asarray(errors)
		step_error[run-1, z, :] = errors
		min_all = np.asarray(min_all)
		min_all1 = np.asarray(min_all1)
		min_alls = np.asarray(min_alls)
		alg_error[run - 1, z, :] = min_all - np.asarray([min_err] * 7)
		alg1_error[run - 1, z, :] = min_alls - min_all1
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


# Random
start = 1
stop = 6
types = ['random_MCMC']
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
alg1_error = np.zeros((stop-1, 1, 7))
for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		min_all1 = [100000] * len(pipeline['all'])
		min_alls = [0] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])
		pipelines = []
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
				else:
					if np.amin(err) < min_all1[j]:
						min_all1[j] = np.amin(err)
					min_alls[j] += np.amin(err)
					s[j] += 1

		for j in range(len(s)):
			min_alls[j] /= s[j]
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
		errors = np.asarray(errors)
		step_error[run-1, z, :] = errors
		min_all = np.asarray(min_all)
		min_all1 = np.asarray(min_all1)
		min_alls = np.asarray(min_alls)
		alg_error[run - 1, z, :] = min_all - np.asarray([min_err] * 7)
		alg1_error[run - 1, z, :] = min_alls - min_all1
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


x = ['Feature extraction', 'Dimensionality reduction', 'Learning algorithm']

x1 = [1, 2, 3]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axs[0].errorbar(x1, grid_step_error[0], grid_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs[0].errorbar(x1, random_step_error[0], random_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs[0].errorbar(x1, bayesian_step_error[0], bayesian_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='bayesian optimization (SMAC)')
axs[0].legend()
axs[0].axhline(y=0)
axs[0].set_title('Step error contributions')
axs[0].set_xlabel('Agnostic step')
axs[0].set_ylabel('Errors')
labels = []
cnt = 1
for item in axs[0].get_xticklabels():
	if len(item.get_text()) == 0:
		labels.append('')
	elif int(float(item.get_text())) == cnt:
		labels.append(x[cnt-1])
		cnt += 1
	else:
		labels.append('')

axs[0].set_xticklabels(labels)


x = pipeline['all']
x1 = [1, 2, 3, 4, 5, 6, 7]
axs[1].errorbar(x1, grid_alg_error[0], grid_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs[1].errorbar(x1, random_alg_error[0], random_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs[1].errorbar(x1, bayesian_alg_error[0], bayesian_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='bayesian optimization (SMAC)')
axs[1].legend()
axs[1].axhline(y=0)
axs[1].set_title('Algorithm error contributions (type 1)')
axs[1].set_xlabel('Agnostic algorithm')
axs[1].set_ylabel('Errors')
labels = []
cnt = 1
for item in axs[1].get_xticklabels():
	if len(item.get_text()) == 0:
		labels.append('')
	elif int(float(item.get_text())) == cnt:
		labels.append(x[cnt-1])
		cnt += 1
	else:
		labels.append('')

axs[1].set_xticklabels(labels)

x = pipeline['all']
axs[2].errorbar(range(1, 8), grid_alg1_error[0], grid_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs[2].errorbar(range(1, 8), random_alg1_error[0], random_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs[2].errorbar(range(1, 8), bayesian_alg1_error[0], bayesian_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='bayesian optimization (SMAC)')
axs[2].legend()
axs[2].axhline(y=0)
axs[2].set_title('Algorithm error contributions (type 2)')
axs[2].set_xlabel('Agnostic algorithm')
axs[2].set_ylabel('Errors')
axs[2].set_xticklabels(labels)
plt.savefig(results_home + 'figures/agnostic_error_' + data_name + '.jpg')
plt.close()



