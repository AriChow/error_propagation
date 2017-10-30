import numpy as np
import os
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import train_test_split
from mahotas.features import haralick
import cv2
# from keras.applications.vgg19 import VGG19
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg19 import preprocess_input
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

# def VGG_all_features(X):
# 	f = []
# 	model = VGG19(weights='imagenet')
# 	for i in range(len(X)):
# 		img = cv2.imread(X[i])
# 		img = cv2.resize(img, dsize=(224, 224))
# 		img = img.astype('float32')
# 		x = np.expand_dims(img, axis=0)
# 		x = preprocess_input(x)
# 		features = model.predict(x)
# 		if i == 0:
# 			f = features
# 		else:
# 			f = np.vstack((f, features))
# 	return f
#
# def inception_all_features(X):
# 	f = []
# 	model = InceptionV3(weights='imagenet')
# 	for i in range(len(X)):
# 		img = cv2.imread(X[i])
# 		img = cv2.resize(img, dsize=(299, 299))
# 		img = img.astype('float32')
# 		x = np.expand_dims(img, axis=0)
# 		x = preprocess_input(x)
# 		features = model.predict(x)
# 		if i == 0:
# 			f = features
# 		else:
# 			f = np.vstack((f, features))
# 	return f

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

cs = ConfigurationSpace()

feature_extraction = CategoricalHyperparameter("feature_extraction", ["haralick", "VGG", "Inception"], default="haralick")
cs.add_hyperparameter(feature_extraction)

dimensionality_reduction = CategoricalHyperparameter("dimensionality_reduction", ["PCA", "ISOMAP"], default="PCA")
cs.add_hyperparameter(dimensionality_reduction)

learning_algorithm = CategoricalHyperparameter("learning_algorithm", ["SVM", "RF"], default="RF")
cs.add_hyperparameter(learning_algorithm)

haralick_distance = UniformIntegerHyperparameter("haralick_distance", 1, 3, default=1)
cs.add_hyperparameter(haralick_distance)
cond1 = InCondition(child=haralick_distance, parent=feature_extraction, values=["haralick"])
cs.add_condition(cond1)

# surf_octaves = UniformIntegerHyperparameter("surf_octaves", 3, 6, default=4)
# surf_scales = UniformIntegerHyperparameter("surf_scales", 5, 8, default=6)
# cs.add_hyperparameters([surf_octaves, surf_scales])
# cond1 = InCondition(child=surf_octaves, parent=feature_extraction, values=["SURF"])
# cond2 = InCondition(child=surf_scales, parent=feature_extraction, values=["SURF"])
# cs.add_conditions([cond1, cond2])

pca_whiten = CategoricalHyperparameter("pca_whiten", ["true", "false"], default="true")
cs.add_hyperparameter(pca_whiten)
cs.add_condition(InCondition(child=pca_whiten, parent=dimensionality_reduction, values=["PCA"]))

isomap_n_neighbors = UniformIntegerHyperparameter("isomap_n_neighbors", 3, 7, default=5)
isomap_n_components = UniformIntegerHyperparameter("isomap_n_components", 2, 4, default=2)
cs.add_hyperparameters([isomap_n_neighbors, isomap_n_components])
cs.add_condition(InCondition(child=isomap_n_components, parent=dimensionality_reduction, values=["ISOMAP"]))
cs.add_condition(InCondition(child=isomap_n_neighbors, parent=dimensionality_reduction, values=["ISOMAP"]))


svm_C = UniformFloatHyperparameter("svm_C", 0.001, 100.0, default=1.0)
cs.add_hyperparameter(svm_C)
svm_gamma = UniformFloatHyperparameter("svm_gamma", 0.0001, 8, default=1)
cs.add_hyperparameter(svm_gamma)
cond1 = InCondition(child=svm_C, parent=learning_algorithm, values=["SVM"])
cond2 = InCondition(child=svm_gamma, parent=learning_algorithm, values=["SVM"])
cs.add_conditions([cond1, cond2])

rf_n_estimators = UniformIntegerHyperparameter("rf_n_estimators", 8, 400, default=10)
rf_max_features = UniformFloatHyperparameter("rf_max_features", 0.3, 0.95, default=0.5)
cs.add_hyperparameters([rf_max_features, rf_n_estimators])
cond1 = InCondition(child=rf_n_estimators, parent=learning_algorithm, values=["RF"])
cond2 = InCondition(child=rf_max_features, parent=learning_algorithm, values=["RF"])
cs.add_conditions([cond1, cond2])

def pipeline_from_cfg(cfg):
	cfg = {k : cfg[k] for k in cfg if cfg[k]}
	# Load the data
	home = os.path.expanduser('~')
	data_home = home + '/Documents/research/EP_project/data/' + dataset + '/'
	l1 = os.listdir(data_home)
	y = []
	names = []
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home + l1[z] + '/')
		y += [z] * len(l)
		for i in range(len(l)):
			names.append(data_home+l1[z]+'/'+l[i])
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
		y_train =[]
		for i in idx1:
			X_train.append(names1[i])
			y_train.append(y1[i])
		X_val = []
		y_val =[]
		for i in idx2:
			X_val.append(names1[i])
			y_val.append(y1[i])

		# X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.33, random_state=42)

		# Feature extraction
		f_train = []
		# f_test = []
		f_val = []
		s = []
		if cfg["feature_extraction"] == "haralick":
			f_val = haralick_all_features(X_val, cfg["haralick_distance"])
			f_train = haralick_all_features(X_train, cfg["haralick_distance"])
		elif cfg["feature_extraction"] == "VGG":
			f_val = VGG_all_features(idx2)
			f_train = VGG_all_features(idx1)
		elif cfg["feature_extraction"] == "Inception":
			f_val = inception_all_features(idx2)
			f_train = inception_all_features(idx1)

		# Dimensionality reduction
		if cfg["dimensionality_reduction"] == "PCA":
			cfg["pca_whiten"] = True if cfg["pca_whiten"] == "true" else False
			dr = principal_components(f_train, cfg["pca_whiten"])
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		elif cfg["dimensionality_reduction"] == "ISOMAP":
			dr = isomap(f_train, cfg["isomap_n_neighbors"], cfg["isomap_n_components"])
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		# Learning algorithms
		r1 = np.random.choice([1, 2], 1)
		if r1[0] == 1:
			cfg["learning_algorithm"] = "RF"
		else:
			cfg["learning_algorithm"] = "SVM"

		# Learning algorithms
		if cfg["learning_algorithm"] == "RF":
			cfg["rf_n_estimators"] = np.random.choice(range(8, 300), 1)[0]
			cfg["rf_max_features"] = np.random.uniform(0.3, 0.8, 1)[0]
			clf = random_forests(f_train, y_train, cfg["rf_n_estimators"], cfg["rf_max_features"])
			s.append(clf.score(f_val, y_val))
		elif cfg["learning_algorithm"] == "SVM":
			cfg["svm_C"] = np.random.uniform(0.1, 100, 1)[0]
			cfg["svm_gamma"] = np.random.uniform(0.01, 8, 1)[0]
			clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
			s.append(clf.score(f_val, y_val))

	return 1-np.mean(s)

def test_pipeline_from_cfg(cfg):
	cfg = {k : cfg[k] for k in cfg if cfg[k]}
	# Load the data
	home = os.path.expanduser('~')
	data_home = home + '/Documents/research/EP_project/data/' + dataset + '/'
	l1 = os.listdir(data_home)
	s = []
	for k in range(50):
		cfg = incumbent
		r1 = np.random.choice([1, 2], 1)
		learning_algorithm = 'RF'
		if r1[0] == 1:
			cfg["learning_algorithm"] = "RF"
		elif r1[0] == 2:
			cfg["learning_algorithm"] = "SVM"
		y = []
		names = []
		cnt = 0
		for z in range(len(l1)):
			if l1[z][0] == '.':
				continue
			l = os.listdir(data_home + l1[z] + '/')
			y += [z] * len(l)
			cnt += 1
			for i in range(len(l)):
				names.append(data_home+l1[z]+'/'+l[i])
		# Train val split
		X = np.empty((len(y), 1))
		indices = np.arange(len(y))
		_, _, y_train, y_val, idx1, idx2 = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)

		X_train = []
		for i in idx1:
			X_train.append(names[i])
		X_val = []
		for i in idx2:
			X_val.append(names[i])

		# Feature extraction
		f_train = []
		# f_test = []
		f_val = []
		if cfg["feature_extraction"] == "haralick":
			f_val = haralick_all_features(X_val, cfg["haralick_distance"])
			f_train = haralick_all_features(X_train, cfg["haralick_distance"])
		elif cfg["feature_extraction"] == "VGG":
			f_val = VGG_all_features(idx2)
			f_train = VGG_all_features(idx1)
		elif cfg["feature_extraction"] == "Inception":
			f_val = inception_all_features(idx2)
			f_train = inception_all_features(idx1)

		# Dimensionality reduction
		if cfg["dimensionality_reduction"] == "PCA":
			pca_whiten = True if cfg["pca_whiten"] == "true" else False
			dr = principal_components(f_train, pca_whiten)
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		elif cfg["dimensionality_reduction"] == "ISOMAP":
			dr = isomap(f_train, cfg["isomap_n_neighbors"], cfg["isomap_n_components"])
			f_train = dr.transform(f_train)
			f_val = dr.transform(f_val)

		# Learning algorithms

		# Learning algorithms
		if cfg["learning_algorithm"] == "RF":
			cfg["rf_n_estimators"] = np.random.choice(range(8, 300), 1)[0]
			cfg["rf_max_features"] = np.random.uniform(0.3, 0.8, 1)[0]
			clf = random_forests(f_train, y_train, cfg["rf_n_estimators"], cfg["rf_max_features"])
			s.append(clf.score(f_val, y_val))
		elif cfg["learning_algorithm"] == "SVM":
			cfg["svm_C"] = np.random.uniform(0.1, 100, 1)[0]
			cfg["svm_gamma"] = np.random.uniform(0.01, 8, 1)[0]
			clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
			s.append(clf.score(f_val, y_val))
	return 1-np.mean(s)


scenario = Scenario({"run_obj": "quality",
					 "run_count-limit": 200,
					 "cs": cs,
					 "deterministic": "true"})


def write_dict(f, dict):
	k = dict.keys()
	for i in k:
		f.write(i + ':' + str(dict[i]) + '\n')


smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=pipeline_from_cfg)
incumbent = smac.optimize()
inc_value = pipeline_from_cfg(incumbent)
print("LEARNING ALGORITHM AGNOSTIC RESULTS: \n")
print("Validation score: " + str(inc_value) + '\n')
print("Algorithms and hyper-parameters: \n")
print(incumbent._values)

inc_value1 = test_pipeline_from_cfg(incumbent)
print("Test score: " + str(inc_value1) + '\n')

home = os.path.expanduser('~')
results_home = home + '/Dropbox/research/EP_project/results/'
f = open(results_home + 'smac_' + dataset + '.txt', 'a')
f.write("LEARNING ALGORITHM AGNOSTIC RESULTS: \n")
f.write("Validation score: " + str(inc_value) + '\n')
f.write("Algorithms and hyper-parameters: \n")
write_dict(f, incumbent._values)
f.write("Test score: " + str(inc_value1) + '\n')
f.close()
