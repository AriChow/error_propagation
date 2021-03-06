import numpy as np
import os
import pickle

home = os.path.expanduser('~')
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
datasets = ['matsc_dataset2']
pipeline['feature_extraction'] = ["haralick"]
pipeline['dimensionality_reduction'] = ["ISOMAP"]
pipeline['learning_algorithm'] = ["RF"]

pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']
grid_error = np.zeros((1, 3))
random_error = np.zeros((1, 3))
bayesian_error = np.zeros((1, 3))
for z in range(len(datasets)):
	data_name = datasets[z]
	# Bayesian optimization
	start = 1
	stop = 5
	if data_name == 'breast':
		stop = 5

	type1 = 'bayesian_MCMC'
	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_breast_run_' +
							   str(1) + '_final1.pkl','rb'), encoding='latin1')
	path_pipelines = obj.all_incumbents

	fe_params = set()
	dr_params = set()
	la_params = set()
	for i in range(len(path_pipelines)):
		p = path_pipelines[i]
		fe_params.add(p._values['haralick_distance'])
		dr_params.add((p._values['isomap_n_neighbors'], p._values['isomap_n_components']))
		la_params.add((p._values['rf_n_estimators'], p._values['rf_max_features']))

	fe_params = list(fe_params)
	dr_params = list(dr_params)
	la_params = list(la_params)

	alg_error = np.zeros((stop-1, 3))
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_final1.pkl','rb'), encoding='latin1')
		path_pipelines = obj.all_incumbents

		min_err = 1000000
		min_fe = [1000000] * len(fe_params)
		min_dr = [1000000] * len(dr_params)
		min_la = [1000000] * len(la_params)
		for i in range(len(path_pipelines)):
			p = path_pipelines[i]
			err = obj.error_curves[0][i]
			if err < min_err:
				min_err = err
			for j in range(len(fe_params)):
				if p._values['haralick_distance'] == fe_params[j]:
					if min_fe[j] > err:
						min_fe[j] = err

			for j in range(len(dr_params)):
				if (p._values['isomap_n_neighbors'], p._values['isomap_n_components']) == dr_params[j]:
					if min_dr[j] > err:
						min_dr[j] = err

			for j in range(len(la_params)):
				if (p._values['rf_n_estimators'], p._values['rf_max_features']) == la_params[j]:
					if min_la[j] > err:
						min_la[j] = err

		fes = 0
		s = 0
		for f in min_fe:
			if f < 100:
				fes += f
				s += 1
		if s != 0:
			fes /= s
		else:
			fes = 100000

		drs = 0
		s = 0
		for f in min_dr:
			if f < 100:
				drs += f
				s += 1
		if s != 0:
			drs /= s
		else:
			drs = 100000

		las = 0
		s = 0
		for f in min_la:
			if f < 100:
				las += f
				s += 1
		if s != 0:
			las /= s
		else:
			las = 100000

		errors = [fes - min_err, drs - min_err, las - min_err]
		for i in range(len(errors)):
			if np.abs(errors[i]) > 100:
				errors[i] = 0
		errors = np.asarray(errors)
		alg_error[run - 1, :] = errors
	std_error = np.std(alg_error, 0)
	step_error = np.mean(alg_error, 0)
	bayesian_alg_error = step_error.astype('float32')
	bayesian_alg_std_error = std_error.astype('float32')


	# Grid search
	start = 1
	stop = 2
	type1 = 'grid_MCMC'
	alg_error = np.zeros((stop-1, 3))
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl','rb'), encoding='latin1')
		pipelines = obj.pipelines
		paths = obj.paths
		path_pipelines = []
		for i in range(len(pipelines)):
			path = paths[i]
			if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
				path_pipelines = pipelines[i]
				break

		fe_params = set()
		dr_params = set()
		la_params = set()
		for i in range(len(path_pipelines)):
			p = path_pipelines[i]
			fe_params.add(p.haralick_distance)
			dr_params.add((p.n_neighbors, p.n_components))
			la_params.add((p.n_estimators, p.max_features))

		fe_params = list(fe_params)
		dr_params = list(dr_params)
		la_params = list(la_params)
		min_err = 1000000
		min_fe = [1000000] * len(fe_params)
		min_dr = [1000000] * len(dr_params)
		min_la = [1000000] * len(la_params)
		for i in range(len(path_pipelines)):
			p = path_pipelines[i]
			err = p.get_error()
			if err < min_err:
				min_err = err
			for j in range(len(fe_params)):
				if p.haralick_distance == fe_params[j]:
					if min_fe[j] > err:
						min_fe[j] = err

			for j in range(len(dr_params)):
				if (p.n_neighbors, p.n_components) == dr_params[j]:
					if min_dr[j] > err:
						min_dr[j] = err

			for j in range(len(la_params)):
				if (p.n_estimators, p.max_features) == la_params[j]:
					if min_la[j] > err:
						min_la[j] = err
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]

		errors = np.asarray(errors)
		alg_error[run - 1, :] = errors
	std_error = np.std(alg_error, 0)
	step_error = np.mean(alg_error, 0)
	grid_alg_error = step_error.astype('float32')
	grid_alg_std_error = std_error.astype('float32')


	# Random search
	start = 1
	stop = 6
	type1 = 'random_MCMC'
	alg_error = np.zeros((stop-1, 3))
	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_final1.pkl','rb'), encoding='latin1')
		pipelines = obj.pipelines
		paths = obj.paths
		path_pipelines = []
		for i in range(len(pipelines)):
			path = paths[i]
			if path[0] == pipeline['feature_extraction'][0] and path[1] == pipeline['dimensionality_reduction'][0] and path[2] == pipeline['learning_algorithm'][0]:
				path_pipelines = pipelines[i]
				break


		fe_params = set()
		dr_params = set()
		la_params = set()
		for i in range(len(path_pipelines)):
			p = path_pipelines[i]
			fe_params.add(p.haralick_distance)
			dr_params.add((p.n_neighbors, p.n_components))
			la_params.add((p.n_estimators, p.max_features))

		min_err = 1000000
		fe_params = list(fe_params)
		dr_params = list(dr_params)
		la_params = list(la_params)
		min_fe = [1000000] * len(fe_params)
		min_dr = [1000000] * len(dr_params)
		min_la = [1000000] * len(la_params)
		for i in range(len(path_pipelines)):
			p = path_pipelines[i]
			err = p.get_error()
			if err < min_err:
				min_err = err
			for j in range(len(fe_params)):
				if p.haralick_distance == fe_params[j]:
					if min_fe[j] > err:
						min_fe[j] = err

			for j in range(len(dr_params)):
				if (p.n_neighbors, p.n_components) == dr_params[j]:
					if min_dr[j] > err:
						min_dr[j] = err

			for j in range(len(la_params)):
				if (p.n_estimators, p.max_features) == la_params[j]:
					if min_la[j] > err:
						min_la[j] = err
		errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]

		errors = np.asarray(errors)
		alg_error[run - 1, :] = errors
	std_error = np.std(alg_error, 0)
	step_error = np.mean(alg_error, 0)
	random_alg_error = step_error.astype('float32')
	random_alg_std_error = std_error.astype('float32')
	grid_error[z, :] = grid_alg_error
	random_error[z, :] = random_alg_error
	bayesian_error[z, :] = bayesian_alg_error

import matplotlib.pyplot as plt

x = pipeline['all']

_, axs = plt.subplots(nrows=1, ncols=1)
x1 = [1, 2, 3]
axs.errorbar(x1, np.mean(grid_error, 0), np.std(grid_error, 0), linestyle='None', marker='^', capsize=3, color='r', label='grid search')
axs.plot(x1, np.mean(grid_error, 0), color='r')
axs.errorbar(x1, np.mean(random_error, 0), np.std(random_error, 0), linestyle='None', marker='^', capsize=3, color='b', label='random search')
axs.plot(x1, np.mean(random_error, 0), color='b')
axs.errorbar(x1, np.mean(bayesian_error, 0), np.std(bayesian_error, 0), linestyle='None', marker='^', capsize=3, color='y', label='bayesian optimization')
axs.plot(x1, np.mean(bayesian_error, 0), color='y')
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

box = axs.get_position()
axs.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
axs.axhline(y=0)
axs.set_title('Algorithm error contributions')
axs.set_xlabel('Agnostic algorithm')
axs.set_ylabel('Error contributions')
axs.set_xticklabels(labels)
plt.savefig(results_home + 'figures/agnostic_error_alg_alternative_matsc_dataset2.eps')
plt.close()





