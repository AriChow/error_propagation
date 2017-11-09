import numpy as np
from prototypes.grid_search import grid_search
import copy
from prototypes.data_analytic_pipeline import image_classification_pipeline

class GradientQuantification(object):
	def __init__(self, pipeline, hyper=None, data_name=None, data_loc=None):
		self.pipeline = pipeline  # All the information of the pipeline
		self.hyper = hyper
		self.data_name = data_name
		self.data_location = data_loc
		self.gradients = {'inputs': {}, 'feature_extraction': {}, 'dimensionality_reduction': {}}
		self.edge_probabilities = {'inputs': {}, 'feature_extraction': {}, 'dimensionality_reduction': {}}
		self.error = 0
		self.eval_error = 0
		self.trials = []
		self.errors = []

	def get_gradients(self):
		return self.gradients

	def get_error(self):
		return self.expected_error(self.eval_error, self.edge_probabilities)

	def initialize(self):
		output = self.pipeline['feature_extraction']
		probability = [1 / len(output)] * len(output)
		self.edge_probabilities['inputs'] = {'X': probability}
		self.gradients['inputs'] = {'X': [0] * len(probability)}
		inputs = self.pipeline['feature_extraction']
		output = self.pipeline['dimensionality_reduction']
		probability = [1 / len(output)] * len(output)
		probabilities = {}
		ginits = {}
		for i in inputs:
			probabilities[i] = probability
			ginits[i] = [0] * len(probability)
		self.edge_probabilities['feature_extraction'] = probabilities
		self.gradients['feature_extraction'] = ginits

		inputs = self.pipeline['dimensionality_reduction']
		output = self.pipeline['learning_algorithm']
		probability = [1 / len(output)] * len(output)
		probabilities = {}
		ginits = {}
		for i in inputs:
			probabilities[i] = probability
			ginits[i] = [0] * len(probability)
		self.edge_probabilities['dimensionality_reduction'] = probabilities
		self.gradients['dimensionality_reduction'] = ginits
		g = grid_search(self.pipeline, None, self.data_name, self.data_location)
		g.populate_grid_search()
		self.trials = g.get_trials()
		g.run_grid_search()
		self.eval_error = g.get_error()
		self.error = self.expected_error(self.eval_error, self.edge_probabilities)
	
	def gradient_search(self, epochs=1000, delta=0.001, alpha=0.9, agnostic=None):
		# Gradients of the expected error w.r.t the probabilities
		key = list(self.edge_probabilities.keys())
		errors = []
		for epoch in range(epochs):
			# if (epoch + 1) % 31 == 0:
			if True:
				# Hyperparameter search
				key1 = list(self.hyper.keys())
				p = np.random.permutation(len(key1))
				key2 = []
				for i in range(len(key1)):
					key2.append(key1[p[i]])
				for i in range(len(key2)):
					h = key2[i]
					if self.hyper[h]['family'] != agnostic:
						for t in range(len(self.trials)):
							trial = self.trials[t]
							if self.hyper[h]['parent'] in trial:
								# Do local search on h if discrete
								if self.hyper[h]['type'] == 'discrete':
									# Check 5 points in locality, pick the best one
									h1 = self.pipeline[h]
									h_list = []
									if h == 'pca_whiten':
										h_list = [True, False]
									else:
										for j in range(1, 3):
											h_test = h1[0] - self.hyper[h]['jump'] * j
											if h_test < self.hyper[h]['min']:
												h_test = self.hyper[h]['min']
											h_list.append(h_test)
										for j in range(1, 3):
											h_test = h1[0] + self.hyper[h]['jump'] * j
											if h_test > self.hyper[h]['max']:
												h_test = self.hyper[h]['max']
											h_list.append(h_test)
									h_list = np.unique(h_list).tolist()
									err = []
									for j in range(len(h_list)):
										h_test = h_list[j]
										kwargs = trial[3]
										kwargs[h] = h_test
										g1 = image_classification_pipeline(kwargs=kwargs, ml_type='validation', data_name=self.data_name, data_loc=self.data_location, val_splits=3, test_size=0.2, fe=trial[0], dr=trial[1], la=trial[2])
										g1.run()
										err.append(g1.get_error())
									best = np.argmin(err)
									self.pipeline[h] = [h_list[best]]
									self.eval_error[t] = err[best]
								# Do gradient descent on h if continuous
								if self.hyper[h]['type'] == 'continuous':
									h1 = self.pipeline[h]
									h_test = h1[0] - self.hyper[h]['jump']
									if h_test < self.hyper[h]['min']:
										h_test = self.hyper[h]['min']
									kwargs = trial[3]
									kwargs[h] = h_test
									g1 = image_classification_pipeline(kwargs=kwargs, ml_type='validation', data_name=self.data_name, data_loc=self.data_location, val_splits=3, test_size=0.2, fe=trial[0], dr=trial[1], la=trial[2])
									g1.run()
									e1 = copy.deepcopy(self.eval_error)
									e1[t] = g1.get_error()
									err1 = self.expected_error(e1, self.edge_probabilities)
									h_test = h1[0] + self.hyper[h]['jump']
									if h_test > self.hyper[h]['max']:
										h_test = self.hyper[h]['max']
									kwargs = trial[3]
									kwargs[h] = h_test
									g1 = image_classification_pipeline(kwargs=kwargs, ml_type='validation',
																	   data_name=self.data_name,
																	   data_loc=self.data_location, val_splits=3,
																	   test_size=0.2, fe=trial[0], dr=trial[1], la=trial[2])
									g1.run()
									e1 = copy.deepcopy(self.eval_error)
									e1[t] = g1.get_error()
									err2 = self.expected_error(e1, self.edge_probabilities)
									self.pipeline[h] += -alpha * (err2 - err1) / (2 * self.hyper[h]['jump'])
									h1 = self.pipeline[h]
									kwargs = trial[3]
									kwargs[h] = h1[0]
									g1 = image_classification_pipeline(kwargs=kwargs, ml_type='validation',
																	   data_name=self.data_name,
																	   data_loc=self.data_location, val_splits=3,
																	   test_size=0.2, fe=trial[0], dr=trial[1], la=trial[2])
									g1.run()
									self.eval_error[t] = g1.get_error()
			else:
				# Algorithm search
				l = copy.deepcopy(self.edge_probabilities)
				for i in range(len(key)):
					step = l[key[i]]
					if agnostic == key[i]:
						continue
					k1 = list(step.keys())
					for j in range(len(k1)):
						v = step[k1[j]]
						for k in range(len(v)):
							l1 = copy.deepcopy(l)
							l2 = copy.deepcopy(l1[key[i]][k1[j]])
							l2[k] += delta
							l1[key[i]][k1[j]] = copy.deepcopy(l2)
							l1[key[i]][k1[j]] = self.normalize(l1[key[i]][k1[j]])
							err1 = self.expected_error(self.eval_error, l1)
							l1 = copy.deepcopy(l)
							l2 = copy.deepcopy(l1[key[i]][k1[j]])
							l2[k] -= delta
							l1[key[i]][k1[j]] = copy.deepcopy(l2)
							l1[key[i]][k1[j]] = self.normalize(l1[key[i]][k1[j]])
							err2 = self.expected_error(self.eval_error, l1)
							# self.gradients[key[i]][k1[j]][k] = (err1 - err2) / (2 * delta)
							l2 = copy.deepcopy(self.edge_probabilities[key[i]][k1[j]])
							l2[k] += -alpha * (err1 - err2) / (2 * delta)
							self.edge_probabilities[key[i]][k1[j]] = copy.deepcopy(l2)
							# self.edge_probabilities[key[i]][k1[j]] = self.normalize(self.edge_probabilities[key[i]][k1[j]])
				for i in range(len(key)):
					step = self.edge_probabilities[key[i]]
					k1 = list(step.keys())
					for j in range(len(k1)):
						self.edge_probabilities[key[i]][k1[j]] = self.normalize(self.edge_probabilities[key[i]][k1[j]])
			errors.append(self.expected_error(self.eval_error, self.edge_probabilities))

		final_error = self.expected_error(self.eval_error, self.edge_probabilities)
		self.error = final_error
		self.errors =errors
		return errors

	@staticmethod
	def normalize(dr):
		s = 0
		for i in range(len(dr)):
			s += np.exp(dr[i])
		for i in range(len(dr)):
			dr[i] = np.exp(dr[i]) / s
		return dr

	def expected_error(self, err, l):
		trials = self.trials
		key = list(l.keys())
		key1 = l[key[0]]['X']
		key2 = l[key[1]]
		key2_key = list(key2.keys())
		key3 = l[key[2]]
		key3_key = list(key3.keys())
		key4_key = ['SVM', 'RF']
		l1 = []
		err1 = []
		for i in range(len(key1)):
			p = key1[i]
			k2 = key2[key2_key[i]]
			for j in range(len(k2)):
				p1 = copy.deepcopy(p)
				p1 *= k2[j]
				k3 = key3[key3_key[j]]
				for k in range(len(k3)):
					p2 = copy.deepcopy(p1)
					p2 *= k3[k]
					l1.append(p2)
					for t in range(len(trials)):
						if trials[t][0] == key2_key[i] and trials[t][1] == key3_key[j] and trials[t][2] == key4_key[k]:
							err1.append(err[t])
							break
		return np.dot(l1, err1)

