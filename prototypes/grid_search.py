from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np

class grid_search(object):
	def __init__(self, pipeline, trials=None, data_name=None, data_loc=None):
		self.pipeline = pipeline
		self.trials = trials
		self.data_name = data_name
		self.data_location = data_loc
		self.error = []
		self.accuracy = []

	def get_trials(self):
		return self.trials

	def get_error(self):
		return self.error

	def get_accuracy(self):
		return self.accuracy

	def populate_grid_search(self):
		pipeline = self.pipeline
		trials = []
		for fe in pipeline['feature_extraction']:
			t = []
			t1 = list(t)
			t1.append(fe)
			hyper = {}
			if fe == 'haralick':
				for hd in pipeline['haralick_distance']:
					hyper1 = dict(hyper)
					hyper1['haralick_distance'] = hd
					self.feature_populate(t1, hyper1, pipeline, trials)
			else:
				hyper1 = dict(hyper)
				self.feature_populate(t1, hyper1, pipeline, trials)
		r = np.random.permutation(len(trials))
		trials1 = []
		for i in range(len(r)):
			trials1.append(trials[r[i]])
		trials = trials1
		self.trials = trials

	def dimensionality_populate(self, t, hyper, pipeline, trials):
		for la in pipeline['learning_algorithm']:
			t1 = list(t)
			t1.append(la)
			if la == 'RF':
				for ne in pipeline['n_estimators']:
					for mf in pipeline['max_features']:
						hyper1 = dict(hyper)
						hyper1['n_estimators'] = ne
						hyper1['max_features'] = mf
						trials.append(t1 + [hyper1])
			elif la == 'SVM':
				for sc in pipeline['svm_C']:
					for sg in pipeline['svm_gamma']:
						hyper1 = dict(hyper)
						hyper1['svm_C'] = sg
						hyper1['svm_gamma'] = sc
						trials.append(t1 + [hyper1])

	def feature_populate(self, t, hyper, pipeline, trials):
		for dr in pipeline['dimensionality_reduction']:
			t1 = list(t)
			t1.append(dr)
			if dr == 'PCA':
				for p in pipeline['pca_whiten']:
					hyper1 = dict(hyper)
					hyper1['pca_whiten'] = p
					self.dimensionality_populate(t1, hyper1, pipeline, trials)
			elif dr == 'ISOMAP':
				for nn in pipeline['n_neighbors']:
					for nc in pipeline['n_components']:
						hyper1 = dict(hyper)
						hyper1['n_neighbors'] = nn
						hyper1['n_components'] = nc
						self.dimensionality_populate(t1, hyper1, pipeline, trials)

	def run_grid_search(self, max_time=36000):
		import time
		start = time.time()
		for tr in self.trials:
			if len(tr) == 1:
				tr = tr[0]
			pipeline = image_classification_pipeline(tr[3], 'validation', self.data_name, self.data_location, 3, 0.2, tr[0], tr[1], tr[2])
			pipeline.run()
			err = pipeline.get_error()
			acc = pipeline.get_accuracy()
			self.error.append(err)
			self.accuracy.append(acc)
			now = time.time()
			if now - start > max_time:
				break
