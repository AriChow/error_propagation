from prototypes.data_analytic_pipeline import image_classification_pipeline
import numpy as np
import copy
import pickle
import time
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class bayesian_MCMC():
	def __init__(self, data_name, data_loc, results_loc, run, pipeline):
		self.pipeline = pipeline
		self.paths = []
		self.times = []
		self.error_curves = []
		self.data_name = data_name
		self.data_loc = data_loc
		self.best_pipelines = []
		self.best_pipelines_incumbents = []
		self.potential = []
		self.run = run
		self.results_loc = results_loc

	def populate_paths(self):
		pipeline = self.pipeline
		paths = []
		for i in pipeline['feature_extraction']:
			path = [i]
			for j in pipeline['dimensionality_reduction']:
				path1 = copy.deepcopy(path)
				path1.append(j)
				for k in pipeline['learning_algorithm']:
					path2 = copy.deepcopy(path1)
					path2.append(k)
					paths.append(path2)
		self.paths = paths


	def bayesianmcmc(self):
		from sklearn.decomposition import PCA
		from sklearn.manifold import Isomap
		from sklearn import svm
		from sklearn.ensemble import RandomForestClassifier
		from mahotas.features import haralick
		import os
		from sklearn.model_selection import StratifiedKFold, train_test_split
		from sklearn import metrics
		from sklearn.preprocessing import StandardScaler
		import cv2

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
			if os.path.exists(self.data_loc + 'features/bayesian/VGG_' + self.data_name + '.npz'):
				f = np.load(open(self.data_loc + 'features/bayesian/VGG_' + self.data_name + '.npz', 'rb'))
				return f.f.arr_0[X, :]
			else:
				f = CNN_all_features(names, 'VGG')
				np.savez(open(self.data_loc + 'features/bayesian/VGG_' + self.data_name + '.npz', 'wb'), f)
				return f[X, :]

		def inception_all_features(names, X):
			home = os.path.expanduser('~')
			if os.path.exists(self.data_loc + 'features/bayesian/inception_' + self.data_name + '.npz'):
				f = np.load(open(self.data_loc + 'features/bayesian/inception_' + self.data_name + '.npz', 'rb'))
				return f.f.arr_0[X, :]
			else:
				f = CNN_all_features(names, 'inception')
				np.savez(open(self.data_loc + 'features/bayesian/inception_' + self.data_name + '.npz', 'wb'), f)
				return f[X, :]

		def principal_components(X, whiten=True):
			pca = PCA(whiten=whiten)
			maxvar = 0.95
			data = X
			X1 = pca.fit(X)
			var = pca.explained_variance_ratio_
			s1 = 0
			for i in range(len(var)):
				s1 += var[i]
			s = 0
			for i in range(len(var)):
				s += var[i]
				if (s * 1.0 / s1) >= maxvar:
					break
			pca = PCA(n_components=i + 1)
			pca.fit(data)
			return pca

		def isomap(X, n_neighbors=5, n_components=2):
			iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
			iso.fit(X)
			return iso

		def random_forests(X, y, n_estimators, max_features):
			clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
										 class_weight='balanced')
			clf.fit(X, y)
			return clf

		def support_vector_machines(X, y, C, gamma):
			clf = svm.SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
			clf.fit(X, y)
			return clf

		def pipeline_from_cfg(cfg):
			cfg = {k: cfg[k] for k in cfg if cfg[k]}
			# Load the data
			data_home = self.data_loc + 'datasets/' + self.data_name + '/'
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
			val_splits = 3
			kf = StratifiedKFold(n_splits=val_splits, random_state=42, shuffle=True)
			names1 = []
			for i in range(len(id1)):
				names1.append((names[id1[i]]))
			f11 = []
			for idx1, idx2 in kf.split(X1, y1):
				# Feature extraction
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
				# Feature extraction
				f_train = []
				# f_test = []
				f_val = []
				if path[0] == "haralick":
					f_val = haralick_all_features(X_val, cfg["haralick_distance"])
					f_train = haralick_all_features(X_train, cfg["haralick_distance"])
				elif path[0] == "VGG":
					f_val = VGG_all_features(names, ids2)
					f_train = VGG_all_features(names, ids1)
				elif path[0] == "inception":
					f_val = inception_all_features(names, ids2)
					f_train = inception_all_features(names, ids1)

				# Dimensionality reduction
				if path[1] == "PCA":
					cfg["pca_whiten"] = True if cfg["pca_whiten"] == "true" else False
					dr = principal_components(f_train, cfg["pca_whiten"])
					f_train = dr.transform(f_train)
					f_val = dr.transform(f_val)

				elif path[1] == "ISOMAP":
					dr = isomap(f_train, cfg["isomap_n_neighbors"], cfg["isomap_n_components"])
					f_train = dr.transform(f_train)
					f_val = dr.transform(f_val)

				# Pre-processing
				normalizer = StandardScaler().fit(f_train)
				f_train = normalizer.transform(f_train)
				f_val = normalizer.transform(f_val)

				# Learning algorithms
				if path[2] == "RF":
					clf = random_forests(f_train, y_train, cfg["rf_n_estimators"], cfg["rf_max_features"])
				elif path[2] == "SVM":
					clf = support_vector_machines(f_train, y_train, cfg["svm_C"], cfg["svm_gamma"])
				p_pred = clf.predict_proba(f_val)
				f11.append(metrics.log_loss(y_val, p_pred))
				s.append(clf.score(f_val, y_val))
			return np.mean(f11)


		self.times = []
		self.error_curves = []
		for i_path in range(len(self.paths)):
			path = self.paths[i_path]
			cs = ConfigurationSpace()
			if path[0] == 'haralick':
				haralick_distance = UniformIntegerHyperparameter("haralick_distance", 1, 3, default=1)
				cs.add_hyperparameter(haralick_distance)
			if path[1] == 'PCA':
				pca_whiten = CategoricalHyperparameter("pca_whiten", ["true", "false"], default="true")
				cs.add_hyperparameter(pca_whiten)
			elif path[1] == 'ISOMAP':
				isomap_n_neighbors = UniformIntegerHyperparameter("isomap_n_neighbors", 3, 7, default=5)
				isomap_n_components = UniformIntegerHyperparameter("isomap_n_components", 2, 4, default=2)
				cs.add_hyperparameters([isomap_n_neighbors, isomap_n_components])
			if path[2] == 'SVM':
				svm_C = UniformFloatHyperparameter("svm_C", 0.1, 100.0, default=1.0)
				cs.add_hyperparameter(svm_C)
				svm_gamma = UniformFloatHyperparameter("svm_gamma", 0.01, 8, default=1)
				cs.add_hyperparameter(svm_gamma)
			elif path[2] == 'RF':
				rf_n_estimators = UniformIntegerHyperparameter("rf_n_estimators", 8, 300, default=10)
				rf_max_features = UniformFloatHyperparameter("rf_max_features", 0.3, 0.8, default=0.5)
				cs.add_hyperparameters([rf_max_features, rf_n_estimators])

			scenario = Scenario({"run_obj": "quality",
								 "cutoff_time": 100000,
								 "runcount_limit": 1000 * 10,
								 "cs": cs,
								 "maxR": 10000,
								 "wallclock_limit" : 100000,
								 "deterministic": "true"})
			t0 = time.time()
			smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=pipeline_from_cfg)
			incumbent, incs, incumbents = smac.optimize()
			inc_value = pipeline_from_cfg(incumbent)
			self.best_pipelines.append(incumbent)
			self.potential.append(inc_value)
			self.error_curves.append(incs)
			self.best_pipelines_incumbents.append(incumbents)
			# self.objects.append(smac)
			t1 = time.time()
			self.times.append(t1-t0)
		pickle.dump(self, open(self.results_loc + 'intermediate/bayesian_MCMC/bayesian_MCMC_' + self.data_name + '_run_' + str(self.run) + '.pkl', 'wb'))