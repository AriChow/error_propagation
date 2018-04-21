from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
import os

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
		self.last_hyper = 0
		self.last_path = 0

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
					if 'naive' in path2[0] or 'naive' in path2[1] or 'naive' in path2[2]:
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

	def gridMcmc(self):
		paths = self.paths
		pipelines = {}
		times = {}
		t0 = time.time()
		if os.path.exists(self.results_loc + 'intermediate/grid_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(
							self.run) + '_naive_last_object.pkl'):
			last_object = pickle.load(open(self.results_loc + 'intermediate/grid_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(
							self.run) + '_naive_last_object.pkl', 'rb'))
			start = last_object.last_path
			self.pipelines = last_object.pipelines
			self.times = last_object.times
			self.run = last_object.run
			self.last_hyper = last_object.last_hyper
		else:
			start = 0
			last_object = None
		for i in range(start, len(paths)):
			pipelines[i] = []
			times[i] = []
			if last_object is not None and i == last_object.last_path:
				path = paths[last_object.last_path]
				start1 = last_object.last_hyper
			else:
				path = paths[i]
				start1 = 0
			hypers = self.populate_path(path)
			for j in range(start1, len(hypers)):
				hyper = hypers[j]
				g = image_classification_pipeline(hyper, ml_type='validation', data_name=self.data_name,
												  data_loc=self.data_loc, type1='grid', fe=path[0], dr=path[1],
												  la=path[2],
												  val_splits=3, test_size=0.2)
				g.run()
				t1 = time.time()
				pipelines[i].append(g)
				times[i].append(t1-t0)
				self.pipelines = pipelines
				self.times = times
				self.last_path = i
				self.last_hyper = j
				t1 = time.time()
				if t1 - t0 > 50000:
					pickle.dump(self, open(
						self.results_loc + 'intermediate/grid_MCMC/' + self.type1 + '_' + self.data_name + '_run_' + str(
							self.run)
						+ '_naive_last_object.pkl', 'wb'))
		pickle.dump(self, open(self.results_loc + 'intermediate/grid_MCMC/' + self.type1 + '_' + self.data_name +
							   '_run_' + str(self.run) + '_naive.pkl', 'wb'))




