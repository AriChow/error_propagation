from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
import multiprocessing as mp
class random_MCMC():
	def __init__(self, data_name=None, data_loc=None, results_loc=None, run=None, type1=None, pipeline=None, iters=None):
		self.pipeline = pipeline
		self.paths = []
		self.pipelines = []
		self.times = []
		self.run = run
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

	def random_parallel(self, i, output):
		t0 = time.time()
		cnt = 0
		last_error = 1000000
		objects = []
		best_pipelines = []
		error_curves = []
		t = 0
		pipeline = self.pipeline
		pipelines = []
		path = self.paths[i]
		while True:
			t += 1
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
											  data_loc=self.data_loc, type1='random_parallel', fe=path[0], dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			err = g.get_error()
			if err >= last_error:
				cnt += 1
			else:
				cnt = 0
			if err < last_error:
				last_error = err
				best_pipelines.append(g)
			objects.append(g)
			error_curves.append(last_error)
			if cnt > self.iters or t > 10000:
				break
		t1 = time.time()
		times = t1 - t0
		pipelines.append(objects)
		output.put((i, pipeline, error_curves, times, best_pipelines))

	def randomMcmc(self):
		results = []
		times = []
		error_curve = []
		pipelines = []
		best_pipelines = []
		# for z in range(4):
		output = mp.Queue()
		processes = [mp.Process(target=self.random_parallel, args=(i, output)) for i in range(len(self.paths))]

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		results += [output.get() for p in processes]

		results.sort()
		for r in results:
			pipelines.append(r[1])
			error_curve.append(r[2])
			times.append(r[3])
			best_pipelines.append(r[4])
		self.pipelines = pipelines
		self.times = times
		self.error_curve = error_curve
		self.best_pipelines = best_pipelines
		pickle.dump(self, open(
			self.results_loc + 'intermediate/random_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(self.run)
			+ '_parallel.pkl', 'wb'))
