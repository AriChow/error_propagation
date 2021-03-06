from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
import multiprocessing as mp
class RL_MCMC():
	def __init__(self, data_name=None, data_loc=None, results_loc=None, run=None, type1=None, pipeline=None, path_resources=None, hyper_resources=None, iters=None):
		self.pipeline = pipeline
		self.paths = []
		self.pipelines = []
		self.times = []
		self.run = run
		self.path_resources = path_resources
		self.hyper_resources = hyper_resources
		self.data_name = data_name
		self.data_loc = data_loc
		self.best_pipelines = []
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

	def MCMC_parallel(self, ind, pipelines, output):
		best_error1 = 100000
		t = 0
		cnt = 0
		error_curves = []
		eps = 1
		t0 = time.time()
		best_pipelines = []
		path = [pipelines[ind][0].feature_extraction, pipelines[ind][0].dimensionality_reduction,
				pipelines[ind][0].learning_algorithm]
		while (True):
			t += 1
			hyper = self.pick_hyper(pipelines[ind], eps, t)
			g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
											  data_loc=self.data_loc, type1='RL_parallel', fe=path[0], dr=path[1], la=path[2],
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
			error_curves.append(best_error1)
			if cnt >= self.iters or t > 10000:
				break
		t1 = time.time()
		self.pipelines = pipelines
		times = t1 - t0
		output.put((ind, pipelines, best_pipelines, error_curves, times))

	def  rlMcmc(self):
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
												  data_loc=self.data_loc, type1='RL_parallel', fe=path[0], dr=path[1], la=path[2],
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
		results = []
		error_curve = []
		best_pipelines = []
		self.pipelines = pipelines
		output = mp.Queue()
		processes = [mp.Process(target=self.MCMC_parallel, args=(i, pipelines, output)) for i in range(len(self.paths))]

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		results += [output.get() for p in processes]

		results.sort()
		for r in results:
			pipelines.append(r[1])
			best_pipelines.append(r[2])
			error_curve.append(r[3])
			times.append(r[4])
		self.pipelines = pipelines
		self.best_pipelines = best_pipelines
		self.times = times
		self.error_curve = error_curve
		pickle.dump(self, open(
			self.results_loc + 'intermediate/RL_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(self.run)
			+ '_parallel.pkl', 'wb'))

