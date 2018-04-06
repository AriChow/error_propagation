'''
This file is for analysing results on test data samples.
'''
import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline1 import image_classification_pipeline
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception", "naive_feature_extraction"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP", "naive_dimensionality_reduction"]
pipeline['learning_algorithm'] = ["SVM", "RF", "naive_learning_algorithm"]
pipeline['all'] = pipeline['feature_extraction'][:-1] + pipeline['dimensionality_reduction'][:-1] + pipeline[
																										'learning_algorithm'][
																									:-1]
lenp = len(pipeline['all'])

# Random1
type1 = 'random_MCMC'
start = 1
stop = 6
step_error_val = np.zeros((stop - 1, 3))
alg_error_val = np.zeros((stop - 1, 7))
alg1_error_val = np.zeros((stop - 1, 7))
step_error_test = np.zeros((stop - 1, 3))
alg_error_test = np.zeros((stop - 1, 7))
alg1_error_test = np.zeros((stop - 1, 7))

test_names = None
data_home1 = data_home + 'datasets/' + data_name + '/'
l1 = os.listdir(data_home1)
y = []
res = 0
names = []
for z in range(len(l1)):
	if l1[z][0] == '.':
		continue
	l = os.listdir(data_home + l1[z] + '/')
	y += [z] * len(l)
	for i in range(len(l)):
		names.append(data_home + l1[z] + '/' + l[i])
X = np.empty((len(y), 1))
indices = np.arange(len(y))
_, _, y_train, y_test, id_train, id_test  = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
step_error_tests = np.zeros((stop - 1, len(y_test), 3))
alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))
step_knn = np.zeros((stop - 1, len(y_test), 3))
alg_knn = np.zeros((stop - 1, len(y_test), 7))
alg1_knn = np.zeros((stop - 1, len(y_test), 7))

for run in range(start, stop):
	obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'), encoding='latin1')

	min_err_val = 100000000
	min_fe_val = [1000000] * len(pipeline['feature_extraction'])
	min_fe_pi = [None] * len(pipeline['feature_extraction'])
	min_dr_val = [1000000] * len(pipeline['dimensionality_reduction'])
	min_dr_pi = [None] * len(pipeline['dimensionality_reduction'])
	min_la_val = [1000000] * len(pipeline['learning_algorithm'])
	min_la_pi = [None] * len(pipeline['learning_algorithm'])

	min_all_val = [100000] * len(pipeline['all'])
	min_all_pi = [None] * len(pipeline['all'])
	min_all1_val = [100000] * len(pipeline['all'])
	min_all1_pi = [None] * len(pipeline['all'])
	min_alls_val = [0] * len(pipeline['all'])
	min_alls_test = [0] * len(pipeline['all'])
	min_alls_tests = np.zeros((len(y_test), len(pipeline['all'])))
	knn_alls = np.zeros(len(y_test), len(pipeline['all']))
	min_alls_pi = [None] * len(pipeline['all'])
	s = [0] * len(pipeline['all'])
	pipelines = obj.pipelines
	best_pi = None
	for i in range(len(pipelines)):
		pi = pipelines[i]
		pie = pi.get_error()
		if pie < min_err_val:
			min_err_val = pie
			best_pi = pi

		p = pipeline['feature_extraction']
		for j in range(len(p)):
			alg = p[j]
			if pi.feature_extraction.decode('latin1') == alg:
				if pie < min_fe_val[j]:
					min_fe_val[j] = pie
					min_fe_pi[j] = pi

		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if pi.dimensionality_reduction.decode('latin1') == alg:
				if pie < min_dr_val[j]:
					min_dr_val[j] = pie
					min_dr_pi[j] = pi

		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if pi.learning_algorithm.decode('latin1') == alg:
				if pie < min_la_val[j]:
					min_la_val[j] = pie
					min_la_pi[j] = pi

		p = pipeline['all']
		for j in range(len(p)):
			alg = p[j]
			if pi.learning_algorithm.decode('latin1') == alg or pi.feature_extraction.decode(
					'latin1') == alg or pi.dimensionality_reduction.decode('latin1') == alg:
				if pie < min_all1_val[j]:
					min_all1_val[j] = pie
					min_all1_pi[j] = pi

				min_alls_val[j] += pie
				hyper = pi.kwargs
				g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
												  data_loc=data_home, type1='random1',
												  fe=pi.feature_extraction.decode('latin1'),
												  dr=pi.dimensionality_reduction.decode('latin1'),
												  la=pi.learning_algorithm.decode('latin1'),
												  val_splits=3, test_size=0.2)
				g.run()
				min_alls_test[j] += g.get_error()
				clf = g.model
				fe_test = g.ft_test
				y_pred = g.y_pred
				y_test = g.y_test
				p_pred = g.p_pred
				fe_train = g.fe_train
				ft_train = g.ft_train
				test = g.fe_test
				knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
				knn.fit(fe_train, y_train)
				knn1 = KNeighborsClassifier(n=np.sqrt(len(y_train)))
				knn1.fit(ft_train, y_train)
				for k in range(len(y_test)):
					f_val = fe_test[k, :]
					y_val = y_test[k]
					p_pred = clf.predict_proba(f_val)
					min_alls_tests[k, j] += metrics.log_loss(y_val, p_pred)
					if pipeline['all'][j] in pipeline['feature_extraction']:
						_, ind = knn.kneighbors(test[k, :])
						frac = sum(np.asarray(ind) == y_val) / len(ind)
						knn_alls[k, j] += 1 - frac
					elif pipeline['all'][j] in pipeline['dimensionality_reduction']:
						_, ind = knn1.kneighbors(f_val)
						frac = sum(np.asarray(ind) == y_val) / len(ind)
						knn_alls[k, j] += 1 - frac
					else:
						knn_alls[k, j] += p_pred[k, y_val] - p_pred[k, y_pred[k]]
				s[j] += 1
			else:
				if pie < min_all_val[j]:
					min_all_val[j] = pie
					min_all_pi[j] = pi

	## Test results for all test samples
	hyper = best_pi.kwargs
	g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
									  data_loc=data_home, type1='random1',
									  fe=pi.feature_extraction.decode('latin1'),
									  dr=pi.dimensionality_reduction.decode('latin1'),
									  la=pi.learning_algorithm.decode('latin1'),
									  val_splits=3, test_size=0.2)
	g.run()
	min_err_test = g.get_error()

	clf = g.model
	fe_test = g.ft_test
	y_test = g.y_test
	p_pred = g.p_pred
	test = g.fe_test
	y_train = g.y_train
	fe_train = g.fe_train
	ft_train = g.ft_train
	knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
	knn.fit(ft_train, y_train)
	test_names = g.test_names
	min_err_tests = [0] * len(y_test)
	knn_min = [0] * len(y_test)
	for k in range(len(y_test)):
		f_val = fe_test[k, :]
		y_val = y_test[k]
		p_pred = clf.predict_proba(f_val)
		min_err_tests[k] = metrics.log_loss(y_val, p_pred)
		_, ind = knn.kneighbors(f_val)
		frac = sum(np.asarray(ind) == y_val) / len(ind)
		knn_min[k] = 1 - frac
	min_err_tests = np.asarray(min_err_tests)
	knn_min = np.asarray(knn_min)

	min_fe_test = [0] * len(pipeline['feature_extraction'])
	min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
	knn_fe = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
	for j in range(len(min_fe_pi)):
		hyper = min_fe_pi[j].kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_fe_test[j] = g.get_error()

		clf = g.model
		fe_test = g.ft_test
		test = g.fe_test
		y_test = g.y_test
		y_train = g.y_train
		fe_train = g.fe_train
		knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn.fit(fe_train, y_train)

		p_pred = g.p_pred
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			p_pred = clf.predict_proba(f_val)
			min_fe_tests[k, j] = metrics.log_loss(y_val, p_pred)
			_, ind = knn.kneighbors(test[k, :])
			frac = sum(np.asarray(ind) == y_val) / len(ind)
			knn_fe[k, j] = 1 - frac

	min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
	min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))
	knn_dr = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))

	for j in range(len(min_dr_pi)):
		hyper = min_dr_pi[j].kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_dr_test[j] = g.get_error()

		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		p_pred = g.p_pred
		y_train = g.y_train
		ft_train = g.ft_train
		knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn.fit(ft_train, y_train)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			p_pred = clf.predict_proba(f_val)
			min_dr_tests[k, j] = metrics.log_loss(y_val, p_pred)
			_, ind = knn.kneighbors(f_val)
			frac = sum(np.asarray(ind) == y_val) / len(ind)
			knn_dr[k, j] = 1 - frac

	min_la_test = [0] * len(pipeline['learning_algorithm'])
	min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
	knn_la = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
	for j in range(len(min_fe_pi)):
		hyper = min_la_pi[j].kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_la_test[j] = g.get_error()

		clf = g.model
		fe_test = g.ft_test
		ft_test = g.ft_test
		y_test = g.y_test
		y_pred = g.y_pred
		p_pred = g.p_pred

		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			p_pred = clf.predict_proba(f_val)
			min_la_tests[k, j] = metrics.log_loss(y_val, p_pred)
			knn_la[k, j] = p_pred[k, y_val] - p_pred[k, y_pred[k]]

	min_all_test = [0] * len(pipeline['all'])
	min_all1_test = [0] * len(pipeline['all'])
	min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
	min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
	knn_all1 = np.zeros(len(y_test), len(pipeline['all']))
	knn_all = np.zeros(len(y_test), len(pipeline['all']))
	for j in range(len(min_all_pi)):
		hyper = min_all_pi[j].kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_all_test[j] = g.get_error()
		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		y_pred = g.y_pred
		p_pred = g.p_pred
		fe_train = g.fe_train
		ft_train = g.ft_train
		test = g.fe_test
		knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn.fit(fe_train, y_train)
		knn1 = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn1.fit(ft_train, y_train)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			p_pred = clf.predict_proba(f_val)
			min_all_tests[k, j] = metrics.log_loss(y_val, p_pred)
			if pipeline['all'][j] in pipeline['feature_extraction']:
				_, ind = knn.kneighbors(test[k, :])
				frac = sum(np.asarray(ind) == y_val) / len(ind)
				knn_all[k, j] = 1 - frac
			elif pipeline['all'][j] in pipeline['dimensionality_reduction']:
				_, ind = knn1.kneighbors(f_val)
				frac = sum(np.asarray(ind) == y_val) / len(ind)
				knn_all[k, j] = 1 - frac
			else:
				knn_all[k, j] = p_pred[k, y_val] - p_pred[k, y_pred[k]]

		hyper = min_all1_pi[j].kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=pi.feature_extraction.decode('latin1'),
										  dr=pi.dimensionality_reduction.decode('latin1'),
										  la=pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		min_all1_test[j] = g.get_error()
		clf = g.model
		fe_test = g.fe_test
		y_test = g.y_test
		p_pred = g.p_pred
		fe_train = g.fe_train
		ft_train = g.ft_train
		test = g.fe_test
		knn = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn.fit(fe_train, y_train)
		knn1 = KNeighborsClassifier(n=np.sqrt(len(y_train)))
		knn1.fit(ft_train, y_train)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			p_pred = clf.predict_proba(f_val)
			min_all1_tests[k, j] = metrics.log_loss(y_val, p_pred)
			if pipeline['all'][j] in pipeline['feature_extraction']:
				_, ind = knn.kneighbors(test[k, :])
				frac = sum(np.asarray(ind) == y_val) / len(ind)
				knn_all1[k, j] = 1 - frac
			elif pipeline['all'][j] in pipeline['dimensionality_reduction']:
				_, ind = knn1.kneighbors(f_val)
				frac = sum(np.asarray(ind) == y_val) / len(ind)
				knn_all1[k, j] = 1 - frac
			else:
				knn_all1[k, j] = p_pred[k, y_val] - p_pred[k, y_pred[k]]

	for j in range(len(s)):
		if s[j] == 0:
			min_alls_test[j] = min_all1_val[j]
			min_alls_val[j] = min_all1_test[j]
			min_alls_tests[:, j] = min_all1_tests[:, j]
			knn_alls[:, j] = knn_all1[:, j]
		else:
			min_alls_val[j] /= s[j]
			min_alls_test[j] /= s[j]
			min_alls_tests[:, j] /= s[j]
			knn_alls[:, j] /= s[j]
	errors = [np.mean(min_fe_val) - min_err_val, np.mean(min_dr_val) - min_err_val, np.mean(min_la_val) - min_err_val]
	errors = np.asarray(errors)
	step_error_val[run - 1, :] = errors
	min_all_val = np.asarray(min_all_val)
	min_all1_val = np.asarray(min_all1_val)
	min_alls_val = np.asarray(min_alls_val)
	alg_error_val[run - 1, :] = min_all_val - np.asarray([min_err_val] * 7)
	alg1_error_val[run - 1, :] = min_alls_val - min_all1_val

	errors = [np.mean(min_fe_test) - min_err_test, np.mean(min_dr_test) - min_err_test,
			  np.mean(min_la_test) - min_err_test]
	errors = np.asarray(errors)
	step_error_test[run - 1, :] = errors
	min_all_test = np.asarray(min_all_test)
	min_all1_test = np.asarray(min_all1_test)
	min_alls_test = np.asarray(min_alls_test)
	alg_error_test[run - 1, :] = min_all_test - np.asarray([min_err_test] * 7)
	alg1_error_test[run - 1, :] = min_alls_test - min_all1_test

	errors = np.zeros((len(y_test), 3))
	errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
	errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
	errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
	step_error_tests[run - 1, :, :] = errors
	alg_error_tests[run - 1, :, :] = min_all_tests - np.tile(min_err_tests, (1, 7))
	alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests

	errors = np.zeros((len(y_test), 3))
	errors[:, 0] = np.mean(knn_fe, 1) - knn_min
	errors[:, 1] = np.mean(knn_dr, 1) - knn_min
	errors[:, 2] = np.mean(knn_la, 1) - knn_min
	step_knn[run - 1, :, :] = errors
	alg_knn[run - 1, :, :] = knn_all - np.tile(knn_min, (1, 7))
	alg1_knn[run - 1, :, :] = knn_alls - knn_all1

std_error = np.std(step_error_val, 0)
step_error = np.mean(step_error_val, 0)
random1_step_error_val = step_error.astype('float32')
random1_std_error_val = std_error.astype('float32')
std_error = np.std(alg_error_val, 0)
step_error = np.mean(alg_error_val, 0)
random1_alg_error_val = step_error.astype('float32')
random1_alg_std_error_val = std_error.astype('float32')
std_error = np.std(alg1_error_val, 0)
step_error = np.mean(alg1_error_val, 0)
random1_alg1_error_val = step_error.astype('float32')
random1_alg1_std_error_val = std_error.astype('float32')

std_error = np.std(step_error_test, 0)
step_error = np.mean(step_error_test, 0)
random1_step_error_test = step_error.astype('float32')
random1_std_erro_test = std_error.astype('float32')
std_error = np.std(alg_error_test, 0)
step_error = np.mean(alg_error_test, 0)
random1_alg_error_test = step_error.astype('float32')
random1_alg_std_error_test = std_error.astype('float32')
std_error = np.std(alg1_error_test, 0)
step_error = np.mean(alg1_error_test, 0)
random1_alg1_error_test = step_error.astype('float32')
random1_alg1_std_error_test = std_error.astype('float32')

std_error = np.std(step_error_tests, 0)
step_error = np.mean(step_error_tests, 0)
random1_step_error_tests = step_error.astype('float32')
random1_std_erro_tests = std_error.astype('float32')
std_error = np.std(alg_error_tests, 0)
step_error = np.mean(alg_error_tests, 0)
random1_alg_error_tests = step_error.astype('float32')
random1_alg_std_error_tests = std_error.astype('float32')
std_error = np.std(alg1_error_tests, 0)
step_error = np.mean(alg1_error_tests, 0)
random1_alg1_error_tests = step_error.astype('float32')
random1_alg1_std_error_tests = std_error.astype('float32')

std_error = np.std(step_knn, 0)
step_error = np.mean(step_knn, 0)
random1_step_knn = step_error.astype('float32')
random1_std_knn = std_error.astype('float32')
std_error = np.std(alg_knn, 0)
step_error = np.mean(alg_knn, 0)
random1_alg_knn = step_error.astype('float32')
random1_alg_std_knn = std_error.astype('float32')
std_error = np.std(alg1_knn, 0)
step_error = np.mean(alg1_knn, 0)
random1_alg1_knn = step_error.astype('float32')
random1_alg1_std_knn = std_error.astype('float32')

# TODO: Properly debug the code
# TODO: Add EP formulation on test data
# TODO: Run Random (HPO) for EP on algorithms /  try EP again on random CASH.
# TODO: Fix the validation data plots
# TODO: Do KNN with EP.
# TODO: Plot the validation EQ properly
# TODO: Do correlation with KNN and EQ
# TODO: Do correlation with KNN and EP
# TODO: Compute EQ for all validation samples and do correlation on grid-search with Random, Bayesian for HPO and CASH
# TODO: Theoretical and deep underpinnings of EQ, EP (marginal costs) and SS (optimization)