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
	# Grid search
	import glob
	files = glob.glob(data_home + 'features/*.npz')
	for f in files:
		os.remove(f)
	pipeline = {}
	pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	pipeline['haralick_distance'] = range(1, 4)

	pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	pipeline['pca_whiten'] = [True, False]
	pipeline['n_neighbors'] = range(3, 8)
	pipeline['n_components'] = range(2, 5)

	pipeline['learning_algorithm'] = ["SVM", "RF"]
	pipeline['n_estimators'] = np.round(np.linspace(8, 300, 10))
	pipeline['max_features'] = np.arange(0.3, 0.8, 0.1)
	pipeline['svm_gamma'] = np.linspace(0.01, 8, 10)
	pipeline['svm_C'] = np.linspace(0.1, 100, 10)
	g = grid_search(pipeline, None, dataset, data_home)
	g.populate_grid_search()
	g.run_grid_search()
	pickle.dump(g, open(results_home + 'intermediate/grid_search_' + dataset + '.pkl', 'wb'), -1)
	trials = g.get_trials()
	error = g.get_error()
	accuracy = g.get_accuracy()
	min_index = np.argmin(error)
	tr = trials[min_index]
	if len(tr) == 1:
		tr = tr[0]
	test_pipeline = image_classification_pipeline(tr[3], 'testing', dataset, data_home, 3, 0.2, tr[0], tr[1], tr[2])
	test_pipeline.run()
	err = test_pipeline.get_error()
	acc = test_pipeline.get_accuracy()

	print("CONTROL RESULTS GRID SEARCH: \n")
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

	f = open(results_home + 'experiments/grid_search_' + dataset + '.txt', 'a')
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
	f.close()
