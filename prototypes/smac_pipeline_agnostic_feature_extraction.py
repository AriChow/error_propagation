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
from sklearn import metrics
from sklearn.preprocessing import Normalizer

dataset = 'breast'

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

def CNN_all_features(names, cnn):
	from keras.applications.vgg19 import VGG19
	from keras.applications.inception_v3 import InceptionV3
	from keras.applications.vgg19 import preprocess_input
	f = []
	if cnn == 'VGG':
		model = VGG19(weights='imagenet')
		dsize = (224, 224)
	else:
		model = InceptionV3(weights='imagenet')
		dsize = (299, 299)
	for i in range(len(names)):
		img = cv2.imread(names[i])
		img = cv2.resize(img, dsize=dsize)
		img = img.astype('float32')
		x = np.expand_dims(img, axis=0)
		x = preprocess_input(x)
		features = model.predict(x)
		if i == 0:
			f = features
		else:
			f = np.vstack((f, features))
	return f


def VGG_all_features(names, X):
	home = os.path.expanduser('~')
	if os.path.exists(home + '/Documents/research/EP_project/data/features/VGG_' + dataset + '.npz'):
		f = np.load(open(home + '/Documents/research/EP_project/data/features/VGG_' + dataset + '.npz', 'rb'))
		return f.f.arr_0[X, :]
	else:
		f = CNN_all_features(names, 'VGG')
		np.savez(open(home + '/Documents/research/EP_project/data/features/VGG_' + dataset + '.npz', 'wb'), f)
		return f[X, :]

def inception_all_features(names, X):
	home = os.path.expanduser('~')
	if os.path.exists(home + '/Documents/research/EP_project/data/features/inception_' + dataset + '.npz'):
		f = np.load(open(home + '/Documents/research/EP_project/data/features/inception_' + dataset + '.npz', 'rb'))
		return f.f.arr_0[X, :]
	else:
		f = CNN_all_features(names, 'inception')
		np.savez(open(home + '/Documents/research/EP_project/data/features/inception_' + dataset + '.npz', 'wb'), f)
		return f[X, :]

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
	clf = svm.SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
	clf.fit(X, y)
	return clf

cs = ConfigurationSpace()



dimensionality_reduction = CategoricalHyperparameter("dimensionality_reduction", ["PCA", "ISOMAP"], default="PCA")
cs.add_hyperparameter(dimensionality_reduction)

learning_algorithm = CategoricalHyperparameter("learning_algorithm", ["SVM", "RF"], default="RF")
cs.add_hyperparameter(learning_algorithm)


pca_whiten = CategoricalHyperparameter("pca_whiten", ["true", "false"], default="true")
cs.add_hyperparameter(pca_whiten)
cs.add_condition(InCondition(child=pca_whiten, parent=dimensionality_reduction, values=["PCA"]))

isomap_n_neighbors = UniformIntegerHyperparameter("isomap_n_neighbors", 3, 7, default=5)
isomap_n_components = UniformIntegerHyperparameter("isomap_n_components", 2, 4, default=2)
cs.add_hyperparameters([isomap_n_neighbors, isomap_n_components])
cs.add_condition(InCondition(child=isomap_n_components, parent=dimensionality_reduction, values=["ISOMAP"]))
cs.add_condition(InCondition(child=isomap_n_neighbors, parent=dimensionality_reduction, values=["ISOMAP"]))


svm_C = UniformFloatHyperparameter("svm_C", 0.1, 100.0, default=1.0)
cs.add_hyperparameter(svm_C)
svm_gamma = UniformFloatHyperparameter("svm_gamma", 0.01, 8, default=1)
cs.add_hyperparameter(svm_gamma)
cond1 = InCondition(child=svm_C, parent=learning_algorithm, values=["SVM"])
cond2 = InCondition(child=svm_gamma, parent=learning_algorithm, values=["SVM"])
cs.add_conditions([cond1, cond2])

rf_n_estimators = UniformIntegerHyperparameter("rf_n_estimators", 8, 300, default=10)
rf_max_features = UniformFloatHyperparameter("rf_max_features", 0.3, 0.8, default=0.5)
cs.add_hyperparameters([rf_max_features, rf_n_estimators])
cond1 = InCondition(child=rf_n_estimators, parent=learning_algorithm, values=["RF"])
cond2 = InCondition(child=rf_max_features, parent=learning_algorithm, values=["RF"])
cs.add_conditions([cond1, cond2])

def pipeline_from_cfg(cfg):
	cfg = {k : cfg[k] for k in cfg if cfg[k]}
	# Load the data
	home = os.path.expanduser('~')
	data_home = home + '/Documents/research/EP_project/data/datasets/' + dataset + '/'
	l1 = os.listdir(data_home)
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
	X1, _, y1, y_val, id1, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	s = []
	f11 = []
	val_splits = 3
	kf = StratifiedKFold(n_splits=val_splits, random_state=42, shuffle=True)
	names1 = []
	for i in range(len(id1)):
		names1.append((names[id1[i]]))
	for idx1, idx2 in kf.split(X1, y1):
		# Feature extraction
		f_train = []
		# f_test = []
		f_val = []
		ids1 = []
		X_train = []
		y_train = []
		for i in idx1:
			X_train.append(names1[i])
			y_train.append(y1[i])
			ids1.append(id1[i])
		X_val = []
		y_val = []
		ids2 = []
		for i in idx2:
			X_val.append(names1[i])
			y_val.append(y1[i])
			ids2.append(id1[i])
		r1 = np.random.choice([1, 2, 3], 1)
		if r1[0] == 1:
			haralick_distance = np.random.choice([1, 2, 3], 1)[0]
			f_val = haralick_all_features(X_val, haralick_distance)
			f_train = haralick_all_features(X_train, haralick_distance)
		elif r1[0] == 2:
			f_val = VGG_all_features(names, ids2)
			f_train = VGG_all_features(names, ids1)
		elif r1[0] == 3:
			f_val = inception_all_features(names, ids2)
			f_train = inception_all_features(names, ids1)

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

		# Pre-processing
		normalizer = Normalizer().fit(f_train)
		f_train = normalizer.transform(f_train)
		f_val = normalizer.transform(f_val)

		clf = []
		# Learning algorithms
		if cfg["learning_algorithm"] == "RF":
			clf = random_forests(f_train, y_train, cfg["rf_n_estimators"], cfg["rf_max_features"])
		elif cfg["learning_algorithm"] == "SVM":
			clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
		p_pred = clf.predict_proba(f_val)
		f11.append(metrics.log_loss(y_val, p_pred))
		s.append(clf.score(f_val, y_val))
	return np.mean(f11)

def test_pipeline_from_cfg(cfg):
	cfg = {k: cfg[k] for k in cfg if cfg[k]}
	# Load the data
	home = os.path.expanduser('~')
	data_home = home + '/Documents/research/EP_project/data/datasets/' + dataset + '/'
	l1 = os.listdir(data_home)
	f11 = []
	s = []
	for i in range(50):
		r1 = np.random.choice([1, 2, 3], 1)
		feature_extraction = "haralick"
		if r1[0] == 1:
			feature_extraction = "haralick"
		elif r1[0] == 2:
			feature_extraction = "VGG"
		elif r1[0] == 3:
			feature_extraction = "inception"
		y = []
		names = []
		cnt = 0
		for z in range(len(l1)):
			if l1[z][0] == '.':
				continue
			l = os.listdir(data_home + l1[z] + '/')
			y += [z] * len(l)
			cnt += 1
			for j in range(len(l)):
				names.append(data_home+l1[z]+'/'+l[j])
		# Train val split
		X = np.empty((len(y), 1))
		indices = np.arange(len(y))
		_, _, y_train, y_val, idx1, idx2 = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)

		X_train = []
		for j in idx1:
			X_train.append(names[j])
		X_val = []
		for j in idx2:
			X_val.append(names[j])

		# Feature extraction
		f_train = []
		# f_test = []
		f_val = []

		if feature_extraction == "haralick":
			haralick_distance = np.random.choice([1, 2, 3], 1)[0]
			f_val = haralick_all_features(X_val, haralick_distance)
			f_train = haralick_all_features(X_train, haralick_distance)
		elif feature_extraction == "VGG":
			f_val = VGG_all_features(names, idx2)
			f_train = VGG_all_features(names, idx1)
		elif feature_extraction == "inception":
			f_val = inception_all_features(names, idx2)
			f_train = inception_all_features(names, idx1)

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

		# Pre-processing
		normalizer = Normalizer().fit(f_train)
		f_train = normalizer.transform(f_train)
		f_val = normalizer.transform(f_val)

		# Learning algorithms
		if cfg["learning_algorithm"] == "RF":
			clf = random_forests(f_train, y_train, cfg["rf_n_estimators"], cfg["rf_max_features"])
		elif cfg["learning_algorithm"] == "SVM":
			clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
		p_pred = clf.predict_proba(f_val)
		f11.append(metrics.log_loss(y_val, p_pred))
		s = clf.score(f_val, y_val)
	return np.mean(f11)


scenario = Scenario({"run_obj": "quality",
					 "run_count-limit": 200,
					 "cs": cs,
					 "deterministic": "true"})



def write_dict(f, dict):
	k = dict.keys()
	for i in k:
		f.write(i + ':' + str(dict[i]) + '\n')
home = os.path.expanduser('~')
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=pipeline_from_cfg)
incumbent = smac.optimize()
pickle.dump(incumbent, open(home + '/Documents/research/EP_project/results/intermediate/smac_feature_extraction_' + dataset + '.pkl', 'wb'), -1)
# pickle.dump(smac, open(home + '/Documents/research/EP_project/results/intermediate/smac_object_feature_extraction_' + dataset + '.pkl', 'wb'), -1)
inc_value = pipeline_from_cfg(incumbent)
print("FEATURE EXTRACTION AGNOSTIC RESULTS: \n")
print("Validation score: " + str(inc_value) + '\n')
print("Algorithms and hyper-parameters: \n")
print(incumbent._values)

inc_value1 = test_pipeline_from_cfg(incumbent)
print("Test score: " + str(inc_value1) + '\n')


results_home = home + '/Documents/research/EP_project/results/experiments/'
f = open(results_home + 'smac_' + dataset + '.txt', 'a')
f.write("FEATURE EXTRACTION AGNOSTIC RESULTS: \n")
f.write("Validation score: " + str(inc_value) + '\n')
f.write("Algorithms and hyper-parameters: \n")
write_dict(f, incumbent._values)
f.write("Test score: " + str(inc_value1) + '\n')
f.close()