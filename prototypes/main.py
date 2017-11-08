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
	# # Grid search
	# import glob
	# files = glob.glob(data_home + 'features/*.npz')
	# for f in files:
	# 	os.remove(f)
	# pipeline = {}
	# pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	# pipeline['haralick_distance'] = range(1, 4)
	#
	# pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	# pipeline['pca_whiten'] = [True, False]
	# pipeline['n_neighbors'] = range(3, 8)
	# pipeline['n_components'] = range(2, 5)
	#
	# pipeline['learning_algorithm'] = ["SVM", "RF"]
	# pipeline['n_estimators'] = np.round(np.linspace(8, 300, 10))
	# pipeline['max_features'] = np.arange(0.3, 0.8, 0.1)
	# pipeline['svm_gamma'] = np.linspace(0.01, 8, 10)
	# pipeline['svm_C'] = np.linspace(0.1, 100, 10)
	# g = grid_search(pipeline, None, dataset, data_home)
	# g.populate_grid_search()
	# g.run_grid_search()
	# pickle.dump(g, open(results_home + 'intermediate/grid_search_' + dataset + '.pkl', 'wb'), -1)
	# trials = g.get_trials()
	# error = g.get_error()
	# accuracy = g.get_accuracy()
	# min_index = np.argmin(error)
	# tr = trials[min_index]
	# if len(tr) == 1:
	# 	tr = tr[0]
	# test_pipeline = image_classification_pipeline(tr[3], 'testing', dataset, data_home, 3, 0.2, tr[0], tr[1], tr[2])
	# test_pipeline.run()
	# err = test_pipeline.get_error()
	# acc = test_pipeline.get_accuracy()
	#
	# print("CONTROL RESULTS GRID SEARCH: \n")
	# print("Validation error: " + str(error[min_index]) + '\n')
	# print("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	# print("Algorithms and hyper-parameters: \n")
	# print('Feature extraction:' + tr[0] + '\n')
	# print('Dimensionality reduction:' + tr[1] + '\n')
	# print('Learning algorithm:' + tr[2] + '\n')
	# print('Hyperparameters: \n')
	# hyper = tr[3]
	# for key, val in hyper.items():
	# 	print(key + ':' + str(val) + '\n')
	# print("Test error: " + str(err) + '\n')
	# print("Test accuracy: " + str(acc) + '\n')
	#
	# f = open(results_home + 'experiments/grid_search_' + dataset + '.txt', 'a')
	# f.write("CONTROL RESULTS: \n")
	# f.write("Validation error: " + str(error[min_index]) + '\n')
	# f.write("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	# f.write("Algorithms and hyper-parameters: \n")
	# f.write('Feature extraction:' + tr[0] + '\n')
	# f.write('Dimensionality reduction:' + tr[1] + '\n')
	# f.write('Learning algorithm:' + tr[2] + '\n')
	# f.write('Hyperparameters: \n')
	# hyper = tr[3]
	# for key, val in hyper.items():
	# 	f.write(key + ':' + str(val) + '\n')
	# f.write("Test error: " + str(err) + '\n')
	# f.write("Test accuracy: " + str(acc) + '\n')
	# f.close()

	# # Random search
	# import glob
	# files = glob.glob(data_home + 'features/*.npz')
	# for f in files:
	# 	os.remove(f)
	# pipeline = {}
	# pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	# pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	# pipeline['learning_algorithm'] = ["SVM", "RF"]
	# g = random_search(pipeline, None, dataset, data_home)
	# g.populate_random_search()
	# g.run_random_search()
	# pickle.dump(g, open(results_home + 'intermediate/random_search_' + dataset + '.pkl', 'wb'), -1)
	# trials = g.get_trials()
	# error = g.get_error()
	# accuracy = g.get_accuracy()
	# min_index = np.argmin(error)
	# tr = trials[min_index]
	# test_pipeline = image_classification_pipeline(tr[3], 'testing', dataset, data_home, 3, 0.2, tr[0], tr[1], tr[2])
	# test_pipeline.run()
	# err = test_pipeline.get_error()
	# acc = test_pipeline.get_accuracy()
	#
	#
	# print("CONTROL RESULTS RANDOM SEARCH: \n")
	# print("Validation error: " + str(error[min_index]) + '\n')
	# print("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	# print("Algorithms and hyper-parameters: \n")
	# print('Feature extraction:' + tr[0] + '\n')
	# print('Dimensionality reduction:' + tr[1] + '\n')
	# print('Learning algorithm:' + tr[2] + '\n')
	# print('Hyperparameters: \n')
	# hyper = tr[3]
	# for key, val in hyper.items():
	# 	print(key + ':' + str(val) + '\n')
	# print("Test error: " + str(err) + '\n')
	# print("Test accuracy: " + str(acc) + '\n')
	#
	# f = open(results_home + 'experiments/random_search_' + dataset + '.txt', 'a')
	# f.write("CONTROL RESULTS: \n")
	# f.write("Validation error: " + str(error[min_index]) + '\n')
	# f.write("Validation accuracy: " + str(accuracy[min_index]) + '\n')
	# f.write("Algorithms and hyper-parameters: \n")
	# f.write('Feature extraction:' + tr[0] + '\n')
	# f.write('Dimensionality reduction:' + tr[1] + '\n')
	# f.write('Learning algorithm:' + tr[2] + '\n')
	# f.write('Hyperparameters: \n')
	# hyper = tr[3]
	# for key, val in hyper.items():
	# 	f.write(key + ':' + str(val) + '\n')
	# f.write("Test error: " + str(err) + '\n')
	# f.write("Test accuracy: " + str(acc) + '\n')
	# f.close()

	# Gradient calculation
	# Empty features directory
	import glob
	files = glob.glob(data_home + 'features/*.npz')
	for f in files:
		os.remove(f)

	pipeline = {}
	pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
	# pipeline['haralick_distance'] = list(np.random.choice([1, 2, 3, 4], 1))
	pipeline['haralick_distance'] = [3]

	pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
	# pipeline['pca_whiten'] = list(np.random.choice([True, False], 1))
	pipeline['pca_whiten'] = [False]
	# pipeline['n_neighbors'] = list(np.random.choice(range(3, 8), 1))
	pipeline['n_neighbors'] = [4]
	# pipeline['n_components'] = list(np.random.choice(range(2, 5), 1))
	pipeline['n_components'] = [3]

	pipeline['learning_algorithm'] = ["SVM", "RF"]
	# pipeline['n_estimators'] = list(np.random.choice(range(8, 301), 1))
	pipeline['n_estimators'] = [200]
	# pipeline['max_features'] = list(np.random.uniform(0.3, 0.8, 1))
	pipeline['max_features'] = [0.9]
	# pipeline['svm_gamma'] = list(np.random.uniform(0.01, 8, 1))
	pipeline['svm_gamma'] = [3]
	# pipeline['svm_C'] = list(np.random.uniform(0.1, 100, 1))
	pipeline['svm_C'] = [40]

	hyper = {}
	hyper['svm_C'] = {'max': 100, 'min': 0.1, 'jump': 0.1, 'type': 'continuous', 'parent': 'SVM', 'family': 'dimensionality_reduction'}
	hyper['svm_gamma'] = {'max': 8, 'min': 0.01, 'jump': 0.01, 'type': 'continuous', 'parent': 'SVM', 'family': 'dimensionality_reduction'}
	hyper['max_features'] = {'max': 0.95, 'min': 0.3, 'jump': 0.01, 'type': 'continuous', 'parent': 'RF', 'family': 'dimensionality_reduction'}
	hyper['n_estimators'] = {'max': 500, 'min': 8, 'jump': 10, 'type': 'discrete', 'parent': 'RF', 'family': 'dimensionality_reduction'}
	hyper['n_components'] = {'max': 5, 'min': 2, 'jump': 1, 'type': 'discrete', 'parent': 'ISOMAP', 'family': 'feature_extraction'}
	hyper['n_neighbors'] = {'max': 8, 'min': 3, 'jump': 1, 'type': 'discrete', 'parent': 'ISOMAP', 'family': 'feature_extraction'}
	hyper['pca_whiten'] = {'max': 2, 'min': 1, 'jump': 1, 'type': 'discrete', 'parent': 'PCA', 'family': 'feature_extraction'}
	hyper['haralick_distance'] = {'max': 4, 'min': 1, 'jump': 1, 'type': 'discrete', 'parent': 'haralick', 'family': 'inputs'}
	g = GradientQuantification(pipeline, hyper, dataset, data_home)
	g.initialize()
	pickle.dump(g, open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'wb'), -1)

	# g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err0 = g.get_error()
	n_epochs = 100000

	errors = g.gradient_search(epochs=n_epochs)
	err = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_' + dataset + '.pkl', 'wb'), -1)

	# Feature extraction agnostic gradients
	# g = GradientQuantification(pipeline, hyper, dataset, data_home)
	# g.initialize()
	g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err01 = g.get_error()
	errors_f = g.gradient_search(epochs=n_epochs, agnostic='inputs')
	err1 = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_feature_extraction_' + dataset + '.pkl', 'wb'), -1)

	# Dimensionality reduction agnostic gradients
	# g = GradientQuantification(pipeline, hyper, dataset, data_home)
	# g.initialize()
	g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err02 = g.get_error()
	errors_d = g.gradient_search(epochs=n_epochs, agnostic='feature_extraction')
	err2 = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_dimensionality_reduction_' + dataset + '.pkl', 'wb'), -1)

	# Learning algorithm agnostic gradients
	# g = GradientQuantification(pipeline, hyper, dataset, data_home)
	# g.initialize()
	g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err03 = g.get_error()
	errors_l = g.gradient_search(epochs=n_epochs, agnostic='dimensionality_reduction')
	err3 = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_learning_algorithm_' + dataset + '.pkl', 'wb'), -1)
	f = open(results_home + 'experiments/gradient_search_' + dataset + '.txt', 'w')
	f.write("Initial error (control):" + str(err0) + '\n')
	f.write("Final error (control):" + str(err) + '\n')
	f.write("Initial error (feature extraction):" + str(err01) + '\n')
	f.write("Final error (feature extraction):" + str(err1) + '\n')
	f.write("Initial error (dimensionality reduction):" + str(err02) + '\n')
	f.write("Final error (dimensionality reduction):" + str(err2) + '\n')
	f.write("Initial error (learning_algorithm):" + str(err03) + '\n')
	f.write("Final error (learning algorithm):" + str(err3) + '\n')
	f.write("Feature extraction error: " + str(err1-err) + '\n')
	f.write("Dimensionality reduction error: " + str(err2 - err) + '\n')
	f.write("Learning algorithm error: " + str(err3 - err) + '\n')
	plt.plot(range(n_epochs), errors, 'r', range(n_epochs), errors_f, 'b', range(n_epochs), errors_d, 'g', range(n_epochs), errors_l, 'y')
	plt.title('Error plots')
	plt.ylabel('Log-loss')
	plt.xlabel('Iterations')
	plt.savefig(results_home + 'experiments/gradient_search_' + dataset + '.jpg')
	f.close()
