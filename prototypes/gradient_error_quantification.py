import numpy as np
from prototypes.grid_search import grid_search

class gradient_quantification(object):
	def __init__(self, pipeline, data_name=None, data_loc=None):
		self.pipeline = pipeline
		self.data_name = data_name
		self.data_location = data_loc
		self.fe_grad = [0] * len(pipeline['feature_extraction'])
		self.dr_grad = [0] * len(pipeline['dimensionality_reduction'])
		self.la_grad = [0] * len(pipeline['learning_algorithm'])
		# self.hd_grad = 0
		# self.pw_grad = 0
		# self.nn_grad = 0
		# self.nc_grad = 0
		# self.ne_grad = 0
		# self.mf_grad = 0
		# self.c_grad = 0
		# self.gamma_grad = 0

	def calculate_gradients(self):
		g = grid_search(self.pipeline, None, self.data_name, self.data_location)
		g.populate_grid_search()
		g.run_grid_search()
		trials = g.get_trials()
		errors = g.get_error()
		fe = self.pipeline['feature_extraction']
		fe_prob = [1 / len(fe)] * len(fe)
		dr = self.pipeline['dimensionality_reduction']
		dr_prob = [1 / len(dr)] * len(dr)
		la = self.pipeline['learning_algorithm']
		la_prob = [1 / len(la)] * len(la)

		# Gradients for algorithms
		delta = 0.01
		for i in range(len(fe_prob)):
			l = list(fe_prob)
			l[i] += delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error1 = np.dot(np.asarray(probs), np.asarray(errors))
			l = list(fe_prob)
			l[i] -= delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error2 = np.dot(np.asarray(probs), np.asarray(errors))
			self.fe_grad[i] = (exp_error1 - exp_error2) / (2 * delta)

		for i in range(len(dr_prob)):
			l = list(dr_prob)
			l[i] += delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error1 = np.dot(np.asarray(probs), np.asarray(errors))
			l = list(dr_prob)
			l[i] -= delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error2 = np.dot(np.asarray(probs), np.asarray(errors))
			self.dr_grad[i] = (exp_error1 - exp_error2) / (2 * delta)

		for i in range(len(la_prob)):
			l = list(la_prob)
			l[i] += delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error1 = np.dot(np.asarray(probs), np.asarray(errors))
			l = list(la_prob)
			l[i] -= delta
			probs = self.calculate_probabilities(l, dr_prob, la_prob)
			exp_error2 = np.dot(np.asarray(probs), np.asarray(errors))
			self.la_grad[i] = (exp_error1 - exp_error2) / (2 * delta)

		# TODO: Differentials of hyper-parameters

	def get_gradients(self):
		return self.fe_grad, self.dr_grad, self.la_grad

	def calculate_probabilities(self, fe, dr, la):
		probs = []
		s = 0
		for i in range(len(fe)):
			s += np.exp(fe[i])
		for i in range(len(fe)):
			fe[i] = np.exp(fe[i]) / s

		s = 0
		for i in range(len(dr)):
			s += np.exp(dr[i])
		for i in range(len(dr)):
			dr[i] = np.exp(dr[i]) / s

		s = 0
		for i in range(len(la)):
			s += np.exp(la[i])
		for i in range(len(la)):
			la[i] = np.exp(la[i]) / s

		probs = []
		for i in range(len(fe)):
			for j in range(len(dr)):
				for k in range(len(la)):
					probs.append(fe[i] * dr[j] * la[k])

		return probs
