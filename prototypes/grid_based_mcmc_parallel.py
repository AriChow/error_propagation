from prototypes.data_analytic_pipeline import image_classification_pipeline
import copy
import pickle
import time
import multiprocessing as mp

class grid_MCMC():
	def __init__(self, data_name=None, data_loc=None, results_loc=None, run=None, type1=None, pipeline=None):
		self.pipeline = pipeline
		self.paths = []
		self.pipelines = []
		self.times = []
		self.run = run
		self.data_name = data_name
		self.data_loc = data_loc
		self.results_loc = results_loc
		self.type1 = type1

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

	def populate_path(self, path):
		hypers = []
		if path[0] == 'haralick':
			h = self.pipeline['haralick_distance']
			hyper = {}
			for i in range(len(h)):
				hyper['haralick_distance'] = h[i]
				hypers.append(copy.deepcopy(hyper))
		hypers1 = []
		if path[1] == 'PCA':
			h = self.pipeline['pca_whiten']
			if len(hypers) > 0:
				for i in range(len(hypers)):
					hyper = hypers[i]
					for j in range(len(h)):
						hyper['pca_whiten'] = h[j]
						hypers1.append(copy.deepcopy(hyper))
			else:
				for i in range(len(h)):
					hyper = {}
					hyper['pca_whiten'] = h[i]
					hypers1.append(copy.deepcopy(hyper))
		elif path[1] == 'ISOMAP':
			h1 = self.pipeline['n_neighbors']
			h2 = self.pipeline['n_components']
			if len(hypers) > 0:
				for i in range(len(hypers)):
					hyper = hypers[i]
					for j in range(len(h1)):
						hyper['n_neighbors'] = h1[j]
						for k in range(len(h2)):
							hyper['n_components'] = h2[k]
							hypers1.append(copy.deepcopy(hyper))
			else:
				for j in range(len(h1)):
					hyper = {}
					hyper['n_neighbors'] = h1[j]
					for k in range(len(h2)):
						hyper['n_components'] = h2[k]
						hypers1.append(copy.deepcopy(hyper))

		hypers2 = []
		if path[2] == 'RF':
			h1 = self.pipeline['n_estimators']
			h2 = self.pipeline['max_features']
			if len(hypers1) > 0:
				for i in range(len(hypers1)):
					hyper = hypers1[i]
					for j in range(len(h1)):
						hyper['n_estimators'] = h1[j]
						for k in range(len(h2)):
							hyper['max_features'] = h2[k]
							hypers2.append(copy.deepcopy(hyper))
		elif path[2] == 'SVM':
			h1 = self.pipeline['svm_gamma']
			h2 = self.pipeline['svm_C']
			if len(hypers1) > 0:
				for i in range(len(hypers1)):
					hyper = hypers1[i]
					for j in range(len(h1)):
						hyper['svm_gamma'] = h1[j]
						for k in range(len(h2)):
							hyper['svm_C'] = h2[k]
							hypers2.append(copy.deepcopy(hyper))
		return hypers2

	def grid_parallel(self, i, output):
		pipelines = []
		times = []
		path = self.paths[i]
		hypers = self.populate_path(path)
		for j in range(len(hypers)):
			hyper = hypers[j]
			g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
											  data_loc=self.data_loc, type1='grid', fe=path[0], dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			t0 = time.time()
			g.run()
			t1 = time.time()
			pipelines.append(g)
			times.append(t1 - t0)
		output.put((i, pipelines, times))

	def gridMcmc(self):
		results = []
		pipelines = []
		times = []
		for z in range(4):
			output = mp.Queue()
			processes = [mp.Process(target=self.grid_parallel, args=(i, output)) for i in range(z*3, (z+1)*3)]

			for p in processes:
				p.start()

			for p in processes:
				p.join()

			results += [output.get() for p in processes]

		results.sort()
		for r in results:
			pipelines.append(r[1])
			times.append(r[2])
		self.times = times
		self.pipelines = pipelines
		pickle.dump(self, open(self.results_loc + 'intermediate/grid_MCMC/' + self.type1 + '_' + self.data_name +
							   '_run_' + str(self.run) + '_parallel.pkl', 'wb'))




