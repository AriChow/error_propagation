import numpy as np
import os
from sklearn.model_selection import train_test_split
from mahotas.features import haralick
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from mahotas.features import surf
import pickle
from sklearn.model_selection import StratifiedKFold

dataset = 'matsc_dataset1'

def haralick_all_features(X, distance=1):
	f = []
	for i in range(len(X)):
		I = cv2.imread(X[i])
		if I is None or I.size == 0 or np.sum(I[:]) == 0 or I.shape[0] == 0 or I.shape[1] == 0:
			h = np.zeros((1, 13))
		else:
			I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
			h = haralick(I, distance=distance, return_mean=True, ignore_zeros=False)
			h = np.expand_dims(h, 0)
		if i == 0:
			f = h
		else:
			f = np.vstack((f, h))
	return f

def surf_all_features(X, octaves=4, scales=6):
	f = []
	for i in range(len(X)):
		I = cv2.imread(X[i])
		I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		h = surf.surf(I, nr_octaves=octaves, nr_scales=scales, max_points=100)
		h = h.flatten()
		h = np.expand_dims(h, 0)
		if i == 0:
			f = h
		else:
			f = np.vstack((f, h))
	return f

def VGG_all_features(X):
	home = os.path.expanduser('~')
	f = np.load(open(home + '/Documents/research/EP_project/data/VGG_' + dataset + '.npz', 'rb'))
	return f.f.arr_0[X, :]


def inception_all_features(X):
	home = os.path.expanduser('~')
	f = np.load(open(home + '/Documents/research/EP_project/data/inception_' + dataset + '.npz', 'rb'))
	return f.f.arr_0[X, :]


def principal_components(X, whiten=True):
	pca = PCA(whiten=whiten)
	maxvar = 0.95
	data = X
	X1 = pca.fit(X)
	var = pca.explained_variance_ratio_
	s = 0
	for i in range(len(var)):
		s += var[i]
		if s >= maxvar:
			break
	pca = PCA(n_components=i+1)
	pca.fit(data)
	return pca


def isomap(X, n_neighbors=5, n_components=2):
	iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
	iso.fit(X)
	return iso

def random_forests(X, y, n_estimators, max_features):
	clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, class_weight='balanced')
	clf.fit(X, y)
	return clf

def support_vector_machines(X, y, C, gamma):
	clf = svm.SVC(C=C, gamma=gamma, class_weight='balanced')
	clf.fit(X, y)
	return clf


def grid_search():
	# Load the data
	home = os.path.expanduser('~')
	data_home = home + '/Documents/research/EP_project/data/' + dataset + '/'
	l1 = os.listdir(data_home)
	y = []
	names = []
	cnt = 0
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home + l1[z] + '/')
		y += [cnt] * len(l)
		cnt += 1
		for i in range(len(l)):
			names.append(data_home + l1[z] + '/' + l[i])
	# Train val split
	X = np.empty((len(y), 1))
	indices = np.arange(len(y))
	X1, _, y1, y_val, id1, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	s = []
	val_splits = 5
	kf = StratifiedKFold(n_splits=val_splits, random_state=42, shuffle=True)
	names1 = []
	for i in range(len(id1)):
		names1.append((names[id1[i]]))
	for idx1, idx2 in kf.split(X1, y1):
		X_train = []
		y_train = []
		for i in idx1:
			X_train.append(names1[i])
			y_train.append(y1[i])
		X_val = []
		y_val = []
		for i in idx2:
			X_val.append(names1[i])
			y_val.append(y1[i])


	feature_extraction_hyperparameters = {}
	feature_extraction_hyperparameters["haralick"] = {"haralick_distance": range(1, 4)}
	feature_extraction_hyperparameters["VGG"] = {}
	feature_extraction_hyperparameters["inception"] = {}

	dimensionality_reduction_hyperparameters = {}
	dimensionality_reduction_hyperparameters["PCA"] = {"pca_whiten": [True, False]}
	dimensionality_reduction_hyperparameters["ISOMAP"] = {"isomap_n_neighbors": range(3, 8), "isomap_n_components" : range(2, 5)}

	learning_algorithm_hyperparameters = {}
	learning_algorithm_hyperparameters["RF"] = {"rf_n_estimators": np.linspace(8, 300, 10), "rf_max_features": np.arange(0.3, 0.8, 0.1)}
	learning_algorithm_hyperparameters["SVM"] = {"svm_C": np.linspace(0.1, 100, 10), "svm_gamma": np.linspace(0.01, 8, 10)}

	trials = {}
	k = feature_extraction_hyperparameters.keys()
	for i in range(len(k)):
		alg = feature_extraction_hyperparameters[k[i]]

