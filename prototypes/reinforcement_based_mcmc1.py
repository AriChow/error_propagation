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

	def pick_path(self, pipelines, eps, t):
		self.potential = []
		for i in range(len(pipelines)):
			p = pipelines[i]
			err = []
			for j in range(len(p)):
				err.append(p[j].get_error())
			self.potential.append(np.sum(err) / len(err))
		potentials = self.potential / np.sum(self.potential)
		p = eps * 1.0 / (t ** (1/8))
		r = np.random.uniform(0, 1, 1)
		if r[0] < p:  # pick path randomly
			r1 = np.random.choice(range(len(pipelines)), 1)
			path = pipelines[r1[0]]
			ind = r1[0]
		else:  # pick path based on potentials
			r1 = np.random.choice(range(len(pipelines)), p=potentials, size=1)
			path = pipelines[r1[0]]
			ind = r1[0]
		p = [path[0].feature_extraction, path[0].dimensionality_reduction, path[0].learning_algorithm]
		return p, ind


	def pick_hyper(self, pipeline, eps, t):
		errs = []
		hypers = []
		for p1 in pipeline:
			errs.append(p1.get_error())
			hypers.append(p1.kwargs)
		# errs /= np.sum(errs)
		hyper = {}
		p = eps * 1.0 / (t ** (1/8))
		p1 = pipeline[0]
		hyper1 = p1.kwargs
		discrete = ['haralick_distance', 'pca_whiten', 'n_neighbors', 'n_estimators', 'n_components']
		for h in hyper1.keys():
			r = np.random.uniform(0, 1, 1)
			if r[0] < p:  # pick hyper-parameters randomly
				if h in discrete:
					r1 = np.random.choice(self.pipeline[h], 1)
					hyper[h] = r1[0]
				else:
					r1 = np.random.uniform(self.pipeline[h][0], self.pipeline[h][-1], 1)
					hyper[h] = r1[0]
			else:  # pick hyper-parameters based on potentials
				H = []
				for i in range(len(hypers)):
					H.append(hypers[i][h])
				if h in discrete:
					err = {}
					for j in self.pipeline[h]:
						err[j] = 0
					for j in range(len(H)):
						err[H[j]] += errs[j]
					errs1 = []
					h_choice = []
					for key, val in err.items():
						errs1.append(val)
						h_choice.append(key)
					errs1 /= np.sum(errs1)
					r1 = np.random.choice(h_choice, size=1, p=errs1)
					hyper[h] = r1[0]
				else:
					mu = np.mean(H)
					std = np.std(H)
					r1 = np.random.normal(mu, std, 1)
					if r1[0] <= self.pipeline[h][0]:
						r1[0] = self.pipeline[h][0]
					if r1[0] >= self.pipeline[h][-1]:
						r1[0] = self.pipeline[h][-1]
					hyper[h] = r1[0]
		return hyper

	def rlMcmc(self):
		eps = 1
		paths = self.paths
		pipeline = self.pipeline
		# Obtain coarse potentials
		pipelines = []
		for path in paths:
			objects = []
			cnt = 0
			while True:
				hyper = {}
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
				g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
												  data_loc=self.data_loc, type1='RL1', fe=path[0], dr=path[1], la=path[2],
												  val_splits=3, test_size=0.2)
				g.run()

				cnt += 1
				if cnt >= self.hyper_resources:
					break
				objects.append(g)
			pipelines.append(objects)

		# pickle.dump(pipelines, open(self.results_loc + 'intermediate/RL_MCMC/rl_mcmc_initial_pipeline.pkl', 'wb'))
		# pipelines = pickle.load(open(self.results_loc + 'intermediate/RL_MCMC/rl_mcmc_initial_pipeline.pkl', 'rb'))

		times = []
		best_pipelines = []
		t0 = time.time()
		best_error1 = 100000
		t = 0
		cnt = 0
		while(True):
			t += 1
			self.pipelines = pipelines
			path, ind = self.pick_path(pipelines, eps, t)
			hyper = self.pick_hyper(pipelines[ind], eps, t)
			g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
											  data_loc=self.data_loc, type1='RL1', fe=path[0], dr=path[1], la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			pipelines[ind].append(g)
			p = pipelines[ind]
			err = []
			for j in range(len(p)):
				err.append(p[j].get_error())
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
		self.pipelines = pipelines
		self.times = times
		self.best_pipelines = best_pipelines
		pickle.dump(self, open(
			self.results_loc + 'intermediate/RL_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(self.run) + '_full.pkl',
			'wb'))