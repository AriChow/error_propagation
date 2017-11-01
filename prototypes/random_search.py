from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np

class random_search(object):
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

	def populate_random_search(self, max_trials=10000):
		pipeline = self.pipeline
		trials = []
		for i in range(max_trials):
			step = pipeline['feature_extraction']
			r = np.random.choice(range(len(step)), 1)[0]
			t = [step[r]]
			hyper = {}
			if step[r] == "haralick":
				r = np.random.choice([1, 2, 3], 1)[0]
				hyper['haralick_distance'] = r
			step = pipeline['dimensionality_reduction']
			r = np.random.choice(range(len(step)), 1)[0]
			t.append(step[r])
			if step[r] == "PCA":
				r = np.random.choice([True, False], 1)[0]
				hyper['pca_whiten'] = r
			elif step[r] == 'ISOMAP':
				isomap_n_neighbors = np.random.choice([3, 4, 5, 6, 7], 1)[0]
				isomap_n_components = np.random.choice([2, 3, 4], 1)[0]
				hyper['n_neighbors'] = isomap_n_neighbors
				hyper['n_components'] = isomap_n_components
			step = pipeline['learning_algorithm']
			r = np.random.choice(range(len(step)), 1)[0]
			t.append(step[r])
			if step[r] == "RF":
				rf_n_estimators = np.random.choice(range(8, 300), 1)[0]
				rf_max_features = np.random.uniform(0.3, 0.8, 1)[0]
				hyper['n_estimators'] = rf_n_estimators
				hyper['max_features'] = rf_max_features
			elif step[r] == 'SVM':
				svm_C = np.random.uniform(0.1, 100, 1)[0]
				svm_gamma = np.random.uniform(0.01, 8, 1)[0]
				hyper['svm_C'] = svm_C
				hyper['svm_gamma'] = svm_gamma
			t1 = t + [hyper]
			trials.append(t1)
		self.trials = trials

	def run_random_search(self, max_time=36000):
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