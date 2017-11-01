import numpy as np
from prototypes.grid_search import grid_search


class GradientQuantification(object):
	def __init__(self, pipeline, data_name=None, data_loc=None):
		self.pipeline = pipeline # All the information of the pipeline
		self.data_name = data_name
		self.data_location = data_loc
		self.gradients = {'inputs': [], 'feature_extraction': [], 'dimensionality_reduction': [], 'learning_algorithm': []}

	def initialize_probabilities(self):
		edge_probabilities = {'inputs': [], 'feature_extraction': [], 'dimensionality_reduction': [], 'learning_algorithm': []}
		output = self.pipeline['feature_extraction']
		probability = [1 / len(output)] * len(output)
		probabilities = {'X': probability}
		edge_probabilities['inputs'].append(probabilities)

		inputs = self.pipeline['feature_extraction']
		output = self.pipeline['dimensionality_reduction']
		probability = [1 / len(output)] * len(output)
		probabilities = {}
		for i in range(inputs.keys()):
			probabilities[i] = probability
			edge_probabilities['feature_extraction'].append(probabilities[i])

		inputs = self.pipeline['dimensionality_reduction']
		output = self.pipeline['learning_algorithm']
		probability = [1 / len(output)] * len(output)
		probabilities = {}
		for i in range(inputs.keys()):
			probabilities[i] = probability
			edge_probabilities['dimensionality_reduction'].append(probabilities[i])
		return edge_probabilities
	
	def calculate_gradients(self):
		g = grid_search(self.pipeline, None, self.data_name, self.data_location)
		g.populate_grid_search()
		g.run_grid_search()
		trials = g.get_trials()
		errors = g.get_error()
		edge_probabilities = self.initialize_probabilities()

		# Gradients of the expected error w.r.t the probabilities
		delta = 0.01
		k = edge_probabilities.keys()

		for i in range(len(k)):
			key = k[i]
			step = edge_probabilities[key]




		
