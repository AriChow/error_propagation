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

# Grid search
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
			if s[j] == 0:
				min_alls[j] = min_all1[j]
			else:
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
grid_alg_error = step_error.astype('float32')  # Method 1
grid_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
grid_alg1_error = step_error.astype('float32')  # Method 2
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
			if s[j] == 0:
				min_alls[j] = min_all1[j]
			else:
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


# Random1
start = 1
stop = 6
types = ['random_MCMC']
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
alg1_error = np.zeros((stop-1, 1, 7))
for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		min_all1 = [100000] * len(pipeline['all'])
		min_alls = [0] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])
		pipelines = obj.pipelines

		for i in range(len(pipelines)):
			pi = pipelines[i]
			pie = pi.get_error()
			if pie < min_err:
				min_err = pie
			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if pi.feature_extraction.decode('latin1') == alg:
					if pie < min_fe[j]:
						min_fe[j] = pie
			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if pi.dimensionality_reduction.decode('latin1') == alg:
					if pie < min_dr[j]:
						min_dr[j] = pie
			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if pi.learning_algorithm.decode('latin1') == alg:
					if pie < min_la[j]:
						min_la[j] = pie

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if pi.learning_algorithm.decode('latin1') == alg or pi.feature_extraction.decode('latin1') == alg or pi.dimensionality_reduction.decode('latin1') == alg:
					if pie < min_all1[j]:
						min_all1[j] = pie
					min_alls[j] += pie
					s[j] += 1
				else:
					if pie < min_all[j]:
						min_all[j] = pie
		for j in range(len(s)):
			if s[j] == 0:
				min_alls[j] = min_all1[j]
			else:
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
			if s[j] == 0:
				min_alls[j] = min_all1[j]
			else:
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


# Bayesian1
start = 1
stop = 6
types = ['bayesian_MCMC']
step_error = np.zeros((stop-1, 1, 3))
alg_error = np.zeros((stop-1, 1, 7))
alg1_error = np.zeros((stop-1, 1, 7))
hypers = ['haralick_distance', 'pca_whiten', 'n_components', 'n_neighbors', 'n_estimators', 'max_features', 'svm_gamma', 'svm_C']

for run in range(start, stop):
	for z, type1 in enumerate(types):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_full.pkl','rb'), encoding='latin1')

		min_err = 100000000
		min_fe = [1000000] * len(pipeline['feature_extraction'])
		min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
		min_la = [1000000] * len(pipeline['learning_algorithm'])
		min_all = [100000] * len(pipeline['all'])
		min_all1 = [100000] * len(pipeline['all'])
		min_alls = [0] * len(pipeline['all'])
		s = [0] * len(pipeline['all'])

		pipelines = obj.all_incumbents
		pipeline_errors = obj.error_curves[0]
		for i in range(len(pipelines)):
			pi = pipelines[i]
			pipeline1 = pi._values
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			# g = image_classification_pipeline(hyper, ml_type='validation', data_name=data_name,
			# 								  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
			# 								  dr=pipeline1['dimensionality_reduction'],
			# 								  la=pipeline1['learning_algorithm'],
			# 								  val_splits=3, test_size=0.2)
			# g.run()
			pie = pipeline_errors[i]

			if pie < min_err:
				min_err = pie
			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if pipeline1['feature_extraction'] == alg:
					if pie < min_fe[j]:
						min_fe[j] = pie
			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if pipeline1['dimensionality_reduction'] == alg:
					if pie < min_dr[j]:
						min_dr[j] = pie
			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if pipeline1['learning_algorithm'] == alg:
					if pie < min_la[j]:
						min_la[j] = pie

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if pipeline1['feature_extraction'] == alg or pipeline1['dimensionality_reduction'] == alg or pipeline1['learning_algorithm'] == alg:
					if pie < min_all1[j]:
						min_all1[j] = pie
					min_alls[j] += pie
					s[j] += 1
				else:
					if pie < min_all[j]:
						min_all[j] = pie
		for j in range(len(min_all1)):
			if min_all1[j] == 100000:
				min_all1[j] = 0

		for j in range(len(min_all)):
			if min_all[j] == 100000:
				min_all[j] = 0

		for j in range(len(s)):
			if s[j] == 0:
				min_alls[j] = min_all1[j]
			else:
				min_alls[j] /= s[j]
		min_fe1 = []
		for f in min_fe:
			if f < 2:
				min_fe1.append(f)
		min_fe = min_fe1

		min_fe1 = []
		for f in min_dr:
			if f < 2:
				min_fe1.append(f)
		min_dr = min_fe1

		min_fe1 = []
		for f in min_la:
			if f < 2:
				min_fe1.append(f)
		min_la = min_fe1
			
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
		errors = np.asarray(errors)
		step_error[run - 1, z, :] = errors
		min_all = np.asarray(min_all)
		min_all1 = np.asarray(min_all1)
		min_alls = np.asarray(min_alls)
		alg_error[run - 1, z, :] = min_all - np.asarray([min_err] * 7)
		alg1_error[run - 1, z, :] = min_alls - min_all1
std_error = np.std(step_error, 0)
step_error = np.mean(step_error, 0)
bayesian1_step_error = step_error.astype('float32')
bayesian1_std_error = std_error.astype('float32')
std_error = np.std(alg_error, 0)
step_error = np.mean(alg_error, 0)
bayesian1_alg_error = step_error.astype('float32')
bayesian1_alg_std_error = std_error.astype('float32')
std_error = np.std(alg1_error, 0)
step_error = np.mean(alg1_error, 0)
bayesian1_alg1_error = step_error.astype('float32')
bayesian1_alg1_std_error = std_error.astype('float32')





import matplotlib.pyplot as plt
x = ['Feature extraction', 'Feature transformation', 'Learning algorithm']

x1 = [1, 2, 3]
_, axs = plt.subplots(nrows=1, ncols=1)
axs.errorbar(x1, grid_step_error[0], grid_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, grid_step_error[0], color='r')
axs.errorbar(x1, random_step_error[0], random_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search (HPO)')
axs.plot(x1, random_step_error[0], color='b')
axs.errorbar(x1, random1_step_error[0], random1_std_error[0], linestyle='None', marker='^', capsize=3, color='k', label='random search(CASH)')
axs.plot(x1, random1_step_error[0], color='k')
axs.errorbar(x1, bayesian_step_error[0], bayesian_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='SMAC (HPO)')
axs.plot(x1, bayesian_step_error[0], color='g')
axs.errorbar(x1, bayesian1_step_error[0], bayesian1_std_error[0], linestyle='None', marker='^', capsize=3, color='y', label='SMAC (CASH)')
axs.plot(x1, bayesian1_step_error[0], color='y')
box = axs.get_position()
axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
axs.axhline(y=0)
axs.set_title('Step error contributions')
axs.set_xlabel('Agnostic step')
axs.set_ylabel('Error contributions')
labels = []
cnt = 1
for item in axs.get_xticklabels():
	if len(item.get_text()) == 0:
		labels.append('')
	elif int(float(item.get_text())) == cnt:
		labels.append(x[cnt-1])
		cnt += 1
	else:
		labels.append('')

axs.set_xticklabels(labels)
plt.savefig(results_home + 'figures/agnostic_error_step_' + data_name + '.jpg')
plt.close()

x = pipeline['all']
_, axs = plt.subplots(nrows=1, ncols=1)
x1 = [1, 2, 3, 4, 5, 6, 7]
axs.errorbar(x1, grid_alg_error[0], grid_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, grid_alg_error[0], color='r')
axs.errorbar(x1, random_alg_error[0], random_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search (HPO)')
axs.plot(x1, random_alg_error[0], color='b')
axs.errorbar(x1, random1_alg_error[0], random1_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='k', label='random search(CASH)')
axs.plot(x1, random1_alg_error[0], color='k')
axs.errorbar(x1, bayesian_alg_error[0], bayesian_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='SMAC (HPO)')
axs.plot(x1, bayesian_alg_error[0], color='g')
axs.errorbar(x1, bayesian1_alg_error[0], bayesian1_alg_std_error[0], linestyle='None', marker='^', capsize=3, color='y', label='SMAC (CASH)')
axs.plot(x1, bayesian1_alg_error[0], color='y')
box = axs.get_position()
axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
axs.axhline(y=0)
axs.set_title('Algorithm error contributions (method 1)')
axs.set_xlabel('Agnostic algorithm')
axs.set_ylabel('Error contributions')
labels = []
cnt = 1
for item in axs.get_xticklabels():
	if len(item.get_text()) == 0:
		labels.append('')
	elif int(float(item.get_text())) == cnt:
		labels.append(x[cnt-1])
		cnt += 1
	else:
		labels.append('')

axs.set_xticklabels(labels)
plt.savefig(results_home + 'figures/agnostic_error_alg_1_' + data_name + '.jpg')
plt.close()


_, axs = plt.subplots(nrows=1, ncols=1)
x = pipeline['all']
axs.errorbar(x1, grid_alg1_error[0], grid_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, grid_alg1_error[0], color='r')
axs.errorbar(x1, random_alg1_error[0], random_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='b', label='random search (HPO)')
axs.plot(x1, random_alg1_error[0], color='b')
axs.errorbar(x1, random1_alg1_error[0], random1_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='k', label='random search(CASH)')
axs.plot(x1, random1_alg1_error[0], color='k')
axs.errorbar(x1, bayesian_alg1_error[0], bayesian_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='g', label='SMAC (HPO)')
axs.plot(x1, bayesian_alg1_error[0], color='g')
axs.errorbar(x1, bayesian1_alg1_error[0], bayesian1_alg1_std_error[0], linestyle='None', marker='^', capsize=3, color='y', label='SMAC (CASH)')
axs.plot(x1, bayesian1_alg1_error[0], color='y')
box = axs.get_position()
axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
axs.axhline(y=0)
axs.set_title('Algorithm error contributions (method 2)')
axs.set_xlabel('Agnostic algorithm')
axs.set_ylabel('Error contributions')
axs.set_xticklabels(labels)
plt.savefig(results_home + 'figures/agnostic_error_alg_2_' + data_name + '.jpg')
plt.close()

