from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time

class RL_MCMC():
	def __init__(self, data_name=None, data_loc=None, results_loc=None, run=None, type1=None, pipeline=None, path_resources=None, hyper_resources=None, iters=None):
		self.pipeline = pipeline
		self.paths = []
		self.pipelines = []
		self.times = []
		self.run = run
		self.path_resources = path_resources
		self.hyper_resources = hyper_resources
		self.potential = []
		self.best_pipelines = []
		self.data_name = data_name
		self.data_loc = data_loc
		self.iters = iters
		self.results_loc = results_loc
		self.type1 = type1
		self.error_curve = []

	def populate_paths(self):
		pipeline = self.pipeline
		paths = []
		for i in pipeline['feature_extraction']:
			path = [i]
			for j in pipeline['dimensionality_reduction']:
				path1 = copy.deepcopy(path)
				path1.append(j)
				for k in pipeline['learning_algorithm']:
					path2 = copy.deepcopy(path1)
					path2.append(k)
					paths.append(path2)
		self.paths = paths


	def pick_hyper(self, pipeline, eps, t):
		errs = []
		hypers = []
		paths = []
		for p1 in pipeline:
			errs.append(1.0 / np.sqrt(p1[0].get_error()))
			p = p1[0].kwargs
			paths.append(p1[1])
			hypers.append(p)
		errs = np.asarray(errs)
		errs /= np.sum(errs)
		hyper = {}
		discrete = ['haralick_distance', 'pca_whiten', 'n_neighbors', 'n_estimators', 'n_components']

		p = eps * 1.0 / (t ** (1 / 8))  # probability of exploitation vs exploration
		r = np.random.uniform(0, 1, 1)
		pipeline = self.pipeline
		if r[0] < p:
			# Pick hyper-parameters randomly (Exploration)
			r1 = np.random.choice(len(self.paths), 1)
			path = self.paths[r1[0]]
			if path[0] == 'haralick':
				r = np.random.choice(pipeline['haralick_distance'], 1)
				hyper['haralick_distance'] = r[0]
			if path[1] == 'PCA':
				r = np.random.choice(pipeline['pca_whiten'], 1)
				hyper['pca_whiten'] = r[0]
			elif path[1] == 'ISOMAP':
				r = np.random.choice(pipeline['n_neighbors'], 1)
				hyper['n_neighbors'] = r[0]
				r = np.random.choice(pipeline['n_components'], 1)
				hyper['n_components'] = r[0]
			if path[2] == 'RF':
				r = np.random.choice(pipeline['n_estimators'], 1)
				hyper['n_estimators'] = r[0]
				r = np.random.uniform(pipeline['max_features'], 1)
				hyper['max_features'] = r[0]
			elif path[2] == 'SVM':
				r = np.random.uniform(pipeline['svm_C'][0], pipeline['svm_C'][-1], 1)
				hyper['svm_C'] = r[0]
				r = np.random.uniform(pipeline['svm_gamma'][0], pipeline['svm_gamma'][-1], 1)
				hyper['svm_gamma'] = r[0]
		else:
			# Pick the hyper-parameters using probabilistic greedy search (Exploitation)
			# Pick the parameter point
			h_vals = []  # For checking whether the new perturbed hyper-parameter is already explored
			for h in hypers:
				hh = []
				for key in h.keys():
					hh.append(h[key])
				h_vals.append(hh)

			# Choosing a random hyper-parameter that is already explored perturbing it
			r1 = np.random.choice(range(len(hypers)), size=1, p=errs)
			path = paths[r1[0]]
			h1 = hypers[r1[0]]
			# Batch the discrete and continuous hyperparameters together and perturb
			hyper_names = h1.keys()
			hyper_discrete = []
			hyper_continuous = []
			for h in hyper_names:
				if h in discrete:
					hyper_discrete.append(h)
				else:
					hyper_continuous.append(h)

			# Sample new discrete hyperparameters
			hyper_discrete_value = []
			for h in hyper_discrete:
				hyper_discrete_value.append(h1[h])

			hd = []
			for i, h in enumerate(hyper_discrete):
				pipeline_values = self.pipeline[h]
				lenh = len(pipeline_values)
				sample_space = 5
				if lenh < 5:
					sample_space = lenh
				ind = pipeline_values.index(hyper_discrete_value[i])
				possible_values = []
				for i1 in range(ind, -1, -1):
					if len(possible_values) > 2:
						break
					possible_values.append(pipeline_values[i1])
				if ind < lenh - 1:
					for i1 in range(ind + 1, lenh):
						if len(possible_values) >= sample_space:
							break
						possible_values.append(pipeline_values[i1])
				hd.append(possible_values)
			import itertools
			hyper_discrete_values = list(itertools.product(*hd))
			for i, h in enumerate(hyper_discrete_values):
				hyper_discrete_values[i] = list(h)
			r1 = np.random.choice(range(len(hyper_discrete_values)), 1)
			hd_new = hyper_discrete_values[r1[0]]

			# Sample new continuous hyperparameters
			hyper_continuous_value = []
			d = len(hyper_continuous)
			for h in hyper_continuous:
				hyper_continuous_value.append(h1[h])
			hyper_continuous_values = []
			for h1 in hyper_continuous:
				hc = []
				for h in hypers:
					if h1 in h.keys():
						hc.append(h[h1])
				hyper_continuous_values.append(hc)
			d1 = []
			for i in hyper_continuous_values:
				d1.append(np.linalg.norm(i))
			std = 3 * np.sqrt(d * np.var(d1)) / len(hyper_continuous_values)

			# cov = np.diag(hyper_continuous_values)
			# hc_new = np.random.multivariate_normal(hyper_continuous_value, cov, 1)
			hyper = {}
			for i, h in enumerate(hyper_discrete):
				hyper[h] = hd_new[i]
			for i, h in enumerate(hyper_continuous):
				#hn = hc_new[0][i]
				# if hn < self.pipeline[h][0]:
				# 	hn = self.pipeline[h][0]
				# elif hn > self.pipeline[h][-1]:
				# 	hn = self.pipeline[h][-1]
				h_low = h1[h] - std
				h_high = h1[h] + std
				if h_low < 0:
					h_low = self.pipeline[h][0]
				if h_high > self.pipeline[h][-1]:
					h_high = self.pipeline[h][-1]
				r = np.random.uniform(h_low, h_high, 1)
				hyper[h] = r[0]
		return hyper, path

	def rlMcmc(self):
		eps = 2
		paths = self.paths
		pipeline = self.pipeline
		# # Obtain coarse potentials
		# pipelines = []
		# cnt = 0
		# while True:
		# 	id = np.random.choice(range(len(paths)), 1)
		# 	path = paths[id[0]]
		# 	hyper = {}
		# 	if path[0] == 'haralick':
		# 		r = np.random.choice(pipeline['haralick_distance'], 1)
		# 		hyper['haralick_distance'] = r[0]
		# 	if path[1] == 'PCA':
		# 		r = np.random.choice(pipeline['pca_whiten'], 1)
		# 		hyper['pca_whiten'] = r[0]
		# 	elif path[1] == 'ISOMAP':
		# 		r = np.random.choice(pipeline['n_neighbors'], 1)
		# 		hyper['n_neighbors'] = r[0]
		# 		r = np.random.choice(pipeline['n_components'], 1)
		# 		hyper['n_components'] = r[0]
		# 	if path[2] == 'RF':
		# 		r = np.random.choice(pipeline['n_estimators'], 1)
		# 		hyper['n_estimators'] = r[0]
		# 		r = np.random.uniform(pipeline['max_features'], 1)
		# 		hyper['max_features'] = r[0]
		# 	elif path[2] == 'SVM':
		# 		r = np.random.uniform(pipeline['svm_C'][0], pipeline['svm_C'][-1], 1)
		# 		hyper['svm_C'] = r[0]
		# 		r = np.random.uniform(pipeline['svm_gamma'][0], pipeline['svm_gamma'][-1], 1)
		# 		hyper['svm_gamma'] = r[0]
		# 	g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
		# 									  data_loc=self.data_loc, type1='RL1', fe=path[0], dr=path[1], la=path[2],
		# 									  val_splits=3, test_size=0.2)
		# 	g.run()
		#
		# 	cnt += 1
		# 	if cnt >= self.hyper_resources:
		# 		break
		# 	pipelines.append((g, path))
		#
		# pickle.dump(pipelines, open(self.results_loc + 'intermediate/RL_MCMC/rl_mcmc_initial_pipeline_full_multivariate_' +
		# 							self.data_name + '_' + str(self.run) + '.pkl', 'wb'))
		pipelines = pickle.load(open(self.results_loc + 'intermediate/RL_MCMC/rl_mcmc_initial_pipeline_full_multivariate_' +
									self.data_name + '_' + str(self.run) + '.pkl', 'rb'))
		times = []
		best_pipelines = []
		t0 = time.time()
		best_error1 = 100000
		t = 0
		cnt = 0
		while(True):
			t += 1
			self.pipelines = copy.deepcopy(pipelines)
			hyper, path = self.pick_hyper(pipelines, eps, t)
			g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
											  data_loc=self.data_loc, type1='RL1', fe=path[0], dr=path[1], la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			pipelines.append((g, path))
			p = pipelines
			err = []
			for j in range(len(p)):
				err.append(p[j][0].get_error())
			best_error = np.amin(err)
			if best_error >= best_error1:
				cnt += 1
			else:
				cnt = 0
			if best_error1 > best_error:
				best_error1 = best_error
				best_pipelines.append(g)
			self.error_curve.append(best_error1)
			if cnt >= self.iters or t > 10000:
				break
			t1 = time.time()
			times.append(t1-t0)
		self.pipelines = copy.deepcopy(pipelines)
		self.times = times
		self.best_pipelines = best_pipelines
		pickle.dump(self, open(
			self.results_loc + 'intermediate/RL_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(self.run) + '_full_multivariate_naive.pkl',
			'wb'))
