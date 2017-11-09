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
	pipeline['max_features'] = [0.75]
	# pipeline['svm_gamma'] = list(np.random.uniform(0.01, 8, 1))
	pipeline['svm_gamma'] = [3]
	# pipeline['svm_C'] = list(np.random.uniform(0.1, 100, 1))
	pipeline['svm_C'] = [40]

	hyper = {}
	hyper['svm_C'] = {'max': 100, 'min': 0.1, 'jump': 0.1, 'type': 'continuous', 'parent': 'SVM', 'family': 'dimensionality_reduction'}
	hyper['svm_gamma'] = {'max': 8, 'min': 0.01, 'jump': 0.01, 'type': 'continuous', 'parent': 'SVM', 'family': 'dimensionality_reduction'}
	hyper['max_features'] = {'max': 0.8, 'min': 0.3, 'jump': 0.01, 'type': 'continuous', 'parent': 'RF', 'family': 'dimensionality_reduction'}
	hyper['n_estimators'] = {'max': 300, 'min': 8, 'jump': 10, 'type': 'discrete', 'parent': 'RF', 'family': 'dimensionality_reduction'}
	hyper['n_components'] = {'max': 4, 'min': 2, 'jump': 1, 'type': 'discrete', 'parent': 'ISOMAP', 'family': 'feature_extraction'}
	hyper['n_neighbors'] = {'max': 7, 'min': 3, 'jump': 1, 'type': 'discrete', 'parent': 'ISOMAP', 'family': 'feature_extraction'}
	hyper['pca_whiten'] = {'max': 2, 'min': 1, 'jump': 1, 'type': 'discrete', 'parent': 'PCA', 'family': 'feature_extraction'}
	hyper['haralick_distance'] = {'max': 3, 'min': 1, 'jump': 1, 'type': 'discrete', 'parent': 'haralick', 'family': 'inputs'}
	g = GradientQuantification(pipeline, hyper, dataset, data_home)
	g.initialize()
	pickle.dump(g, open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'wb'), -1)

	# g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err0 = g.get_error()
	n_epochs = 10000

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
	pickle.dump(g, open(results_home + 'intermediate/gradients_feature_extraction_full_' + dataset + '.pkl', 'wb'), -1)

	# Dimensionality reduction agnostic gradients
	# g = GradientQuantification(pipeline, hyper, dataset, data_home)
	# g.initialize()
	g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err02 = g.get_error()
	errors_d = g.gradient_search(epochs=n_epochs, agnostic='feature_extraction')
	err2 = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_dimensionality_reduction_full_' + dataset + '.pkl', 'wb'), -1)

	# Learning algorithm agnostic gradients
	# g = GradientQuantification(pipeline, hyper, dataset, data_home)
	# g.initialize()
	g = pickle.load(open(results_home + 'intermediate/initial_gradients_' + dataset + '.pkl', 'rb'))
	err03 = g.get_error()
	errors_l = g.gradient_search(epochs=n_epochs, agnostic='dimensionality_reduction')
	err3 = g.get_error()
	pickle.dump(g, open(results_home + 'intermediate/gradients_learning_algorithm_full_' + dataset + '.pkl', 'wb'), -1)
	f = open(results_home + 'experiments/gradient_search_full_' + dataset + '.txt', 'w')
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
	plt.figure()
	# plt.plot(range(n_epochs), errors, 'r', range(n_epochs), errors_f, 'b', range(n_epochs), errors_d, 'g', range(n_epochs), errors_l, 'y')
	plt.plot(range(n_epochs), errors, 'r', label='Control')
	plt.plot(range(n_epochs), errors_f, 'b', label='Agnostic to feature extraction')
	plt.plot(range(n_epochs), errors_d, 'g', label='Agnostic to dimensionality reduction')
	plt.plot(range(n_epochs), errors_l, 'y', label='Agnostic to learning algorithms')

	plt.title('Error plots')
	plt.ylabel('Log-loss')
	plt.xlabel('Iterations')
	plt.savefig(results_home + 'experiments/gradient_search_full_' + dataset + '.jpg')
	f.close()
