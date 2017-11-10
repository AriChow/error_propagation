import numpy as np
from prototypes.grid_search import grid_search
from prototypes.random_search import random_search
import os
from prototypes.gradient_search import GradientQuantification
from prototypes.data_analytic_pipeline import image_classification_pipeline
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
	home = os.path.expanduser('~')
	dataset = 'breast'
	data_home = home + '/Documents/research/EP_project/data/'
	results_home = home + '/Documents/research/EP_project/results/'

	# Random search
	import glob
	files = glob.glob(data_home + 'features/*.npz')
	for f in files:
		os.remove(f)
	pipeline = {}
	pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	pipeline['learning_algorithm'] = ["SVM", "RF"]
	g = random_search(pipeline, dataset, data_home)
	g.populate_random_search(max_trials=1000)
	g.run_random_search()
	pickle.dump(g, open(results_home + 'intermediate/random_search_' + dataset + '.pkl', 'wb'), -1)
	trials = g.get_trials()
	error = g.get_error()
	accuracy = g.get_accuracy()
	min_index = np.argmin(error)
	tr = trials[min_index]
	test_pipeline = image_classification_pipeline(tr[3], 'testing', dataset, data_home, 3, 0.2, tr[0], tr[1], tr[2])
	test_pipeline.run()
	err = test_pipeline.get_error()
	acc = test_pipeline.get_accuracy()


	print("CONTROL RESULTS RANDOM SEARCH: \n")
	print("Validation error: " + str(error[min_index]) + '\n')
	print("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	print("Algorithms and hyper-parameters: \n")
	print('Feature extraction:' + tr[0] + '\n')
	print('Dimensionality reduction:' + tr[1] + '\n')
	print('Learning algorithm:' + tr[2] + '\n')
	print('Hyperparameters: \n')
	hyper = tr[3]
	for key, val in hyper.items():
		print(key + ':' + str(val) + '\n')
	print("Test error: " + str(err) + '\n')
	print("Test accuracy: " + str(acc) + '\n')

	f = open(results_home + 'experiments/random_search_' + dataset + '.txt', 'a')
	f.write("CONTROL RESULTS: \n")
	f.write("Validation error: " + str(error[min_index]) + '\n')
	f.write("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	f.write("Algorithms and hyper-parameters: \n")
	f.write('Feature extraction:' + tr[0] + '\n')
	f.write('Dimensionality reduction:' + tr[1] + '\n')
	f.write('Learning algorithm:' + tr[2] + '\n')
	f.write('Hyperparameters: \n')
	hyper = tr[3]
	for key, val in hyper.items():
		f.write(key + ':' + str(val) + '\n')
	f.write("Test error: " + str(err) + '\n')
	f.write("Test accuracy: " + str(acc) + '\n')
	# f.close()
