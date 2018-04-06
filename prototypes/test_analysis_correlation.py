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

home = os.path.expanduser('~')
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']
lenp = len(pipeline['all'])
datasets = ['matsc_dataset1', 'matsc_dataset2']
classes = [2, 2]
hypers = ['haralick_distance', 'pca_whiten', 'n_components', 'n_neighbors', 'n_estimators', 'max_features', 'svm_gamma', 'svm_C']

for c1, data_name in enumerate(datasets):

	# # Bayesian1
	# type1 = 'bayesian_MCMC'
	# start = 1
	# stop = 6
	# data_home1 = data_home + 'datasets/' + data_name + '/'
	# l1 = os.listdir(data_home1)
	# y = []
	# names = []
	# for z in range(len(l1)):
	# 	if l1[z][0] == '.':
	# 		continue
	# 	l = os.listdir(data_home1 + l1[z] + '/')
	# 	y += [z] * len(l)
	# 	for i in range(len(l)):
	# 		names.append(data_home1 + l1[z] + '/' + l[i])
	# X = np.empty((len(y), 1))
	# indices = np.arange(len(y))
	# _, _, _, y_test, _, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	# step_error_tests = np.zeros((stop - 1, len(y_test), 3))
	# alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
	# alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))
	# hypers = ['haralick_distance', 'pca_whiten', 'n_components', 'n_neighbors', 'n_estimators', 'max_features',
	# 		  'svm_gamma', 'svm_C']
	#
	# for run in range(start, stop):
	# 	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
	# 						   str(run) + '_full.pkl', 'rb'), encoding='latin1')
	#
	# 	min_err_val = 100000000
	# 	min_fe_val = [1000000] * len(pipeline['feature_extraction'])
	# 	min_fe_pi = [None] * len(pipeline['feature_extraction'])
	# 	min_dr_val = [1000000] * len(pipeline['dimensionality_reduction'])
	# 	min_dr_pi = [None] * len(pipeline['dimensionality_reduction'])
	# 	min_la_val = [1000000] * len(pipeline['learning_algorithm'])
	# 	min_la_pi = [None] * len(pipeline['learning_algorithm'])
	#
	# 	min_all_val = [100000] * len(pipeline['all'])
	# 	min_all_pi = [None] * len(pipeline['all'])
	# 	min_all1_val = [100000] * len(pipeline['all'])
	# 	min_all1_pi = [None] * len(pipeline['all'])
	# 	min_alls_val = [0] * len(pipeline['all'])
	# 	min_alls_test = [0] * len(pipeline['all'])
	# 	min_alls_tests = np.zeros((len(y_test), len(pipeline['all'])))
	# 	s = [0] * len(pipeline['all'])
	# 	best_pi = None
	# 	pipelines = obj.all_incumbents
	# 	pipeline_errors = obj.error_curves[0]
	# 	for i in range(len(pipelines)):
	# 		pi = pipelines[i]
	# 		pie = pipeline_errors[i]
	# 		if pie < min_err_val:
	# 			min_err_val = pie
	# 			best_pi = pi
	#
	# 		pi1 = pi._values
	# 		p = pipeline['feature_extraction']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if pi1['feature_extraction'] == alg:
	# 				if pie < min_fe_val[j]:
	# 					min_fe_val[j] = pie
	# 					min_fe_pi[j] = pi
	#
	# 		p = pipeline['dimensionality_reduction']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if pi1['dimensionality_reduction'] == alg:
	# 				if pie < min_dr_val[j]:
	# 					min_dr_val[j] = pie
	# 					min_dr_pi[j] = pi
	#
	# 		p = pipeline['learning_algorithm']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if pi1['learning_algorithm'] == alg:
	# 				if pie < min_la_val[j]:
	# 					min_la_val[j] = pie
	# 					min_la_pi[j] = pi
	#
	# 		p = pipeline['all']
	# 		for j in range(len(p)):
	# 			alg = p[j]
	# 			if pi1['learning_algorithm'] == alg or pi1['feature_extraction'] == alg or pi1['dimensionality_reduction'] == alg:
	# 				if pie < min_all1_val[j]:
	# 					min_all1_val[j] = pie
	# 					min_all1_pi[j] = pi
	#
	# 				min_alls_val[j] += pie
	# 				pipeline1 = pi._values
	# 				hyper = {}
	# 				for v in pipeline1.keys():
	# 					for h in hypers:
	# 						s1 = v.find(h)
	# 						if s1 != -1:
	# 							hyper[h] = pipeline1[v]
	# 				g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 												  data_loc=data_home, type1='bayesian1',
	# 												  fe=pipeline1['feature_extraction'],
	# 												  dr=pipeline1['dimensionality_reduction'],
	# 												  la=pipeline1['learning_algorithm'],
	# 												  val_splits=3, test_size=0.2)
	# 				g.run()
	#
	# 				min_alls_test[j] += g.get_error()
	# 				clf = g.model
	# 				fe_test = g.ft_test
	# 				y_pred = g.y_pred
	# 				y_test = g.y_test
	# 				p_pred = g.p_pred
	# 				for k in range(len(y_test)):
	# 					f_val = fe_test[k, :]
	# 					y_val = y_test[k]
	# 					f_val = np.expand_dims(f_val, axis=0)
	# 					p_pred = clf.predict_proba(f_val)
	# 					y1 = np.zeros((1, classes[c1]))
	# 					y1[0, y_val] = 1
	# 					min_alls_tests[k, j] += metrics.log_loss(y_true=y1, y_pred=p_pred)
	# 				s[j] += 1
	# 			else:
	# 				if pie < min_all_val[j]:
	# 					min_all_val[j] = pie
	# 					min_all_pi[j] = pi
	#
	# 	## Test results for all test samples
	# 	pipeline1 = best_pi._values
	# 	hyper = {}
	# 	for v in pipeline1.keys():
	# 		for h in hypers:
	# 			s1 = v.find(h)
	# 			if s1 != -1:
	# 				hyper[h] = pipeline1[v]
	# 	g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 									  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 									  dr=pipeline1['dimensionality_reduction'],
	# 									  la=pipeline1['learning_algorithm'],
	# 									  val_splits=3, test_size=0.2)
	# 	g.run()
	# 	clf = g.model
	# 	fe_test = g.ft_test
	# 	y_test = g.y_test
	# 	p_pred = g.p_pred
	# 	min_err_tests = [0] * len(y_test)
	# 	for k in range(len(y_test)):
	# 		f_val = fe_test[k, :]
	# 		y_val = y_test[k]
	# 		f_val = np.expand_dims(f_val, axis=0)
	# 		p_pred = clf.predict_proba(f_val)
	# 		y1 = np.zeros((1, classes[c1]))
	# 		y1[0, y_val] = 1
	# 		min_err_tests[k] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	# 	min_err_tests = np.asarray(min_err_tests)
	#
	# 	min_fe_test = [0] * len(pipeline['feature_extraction'])
	# 	min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
	# 	for j in range(len(min_fe_pi)):
	# 		pipeline1 = min_fe_pi[j]._values
	# 		hyper = {}
	# 		for v in pipeline1.keys():
	# 			for h in hypers:
	# 				s1 = v.find(h)
	# 				if s1 != -1:
	# 					hyper[h] = pipeline1[v]
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 										  dr=pipeline1['dimensionality_reduction'],
	# 										  la=pipeline1['learning_algorithm'],
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_fe_test[j] = g.get_error()
	#
	# 		clf = g.model
	# 		fe_test = g.ft_test
	# 		y_test = g.y_test
	# 		p_pred = g.p_pred
	# 		for k in range(len(y_test)):
	# 			f_val = fe_test[k, :]
	# 			y_val = y_test[k]
	# 			f_val = np.expand_dims(f_val, axis=0)
	# 			p_pred = clf.predict_proba(f_val)
	# 			y1 = np.zeros((1, classes[c1]))
	# 			y1[0, y_val] = 1
	# 			min_fe_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	#
	# 	min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
	# 	min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))
	#
	# 	for j in range(len(min_dr_pi)):
	# 		pipeline1 = min_dr_pi[j]._values
	# 		hyper = {}
	# 		for v in pipeline1.keys():
	# 			for h in hypers:
	# 				s1 = v.find(h)
	# 				if s1 != -1:
	# 					hyper[h] = pipeline1[v]
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 										  dr=pipeline1['dimensionality_reduction'],
	# 										  la=pipeline1['learning_algorithm'],
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_dr_test[j] = g.get_error()
	#
	# 		clf = g.model
	# 		fe_test = g.ft_test
	# 		y_test = g.y_test
	# 		p_pred = g.p_pred
	# 		for k in range(len(y_test)):
	# 			f_val = fe_test[k, :]
	# 			y_val = y_test[k]
	# 			f_val = np.expand_dims(f_val, axis=0)
	# 			p_pred = clf.predict_proba(f_val)
	# 			y1 = np.zeros((1, classes[c1]))
	# 			y1[0, y_val] = 1
	# 			min_dr_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	#
	# 	min_la_test = [0] * len(pipeline['learning_algorithm'])
	# 	min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
	# 	for j in range(len(min_la_pi)):
	# 		pipeline1 = min_la_pi[j]._values
	# 		hyper = {}
	# 		for v in pipeline1.keys():
	# 			for h in hypers:
	# 				s1 = v.find(h)
	# 				if s1 != -1:
	# 					hyper[h] = pipeline1[v]
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 										  dr=pipeline1['dimensionality_reduction'],
	# 										  la=pipeline1['learning_algorithm'],
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_la_test[j] = g.get_error()
	#
	# 		clf = g.model
	# 		fe_test = g.ft_test
	# 		y_test = g.y_test
	# 		p_pred = g.p_pred
	# 		for k in range(len(y_test)):
	# 			f_val = fe_test[k, :]
	# 			y_val = y_test[k]
	# 			f_val = np.expand_dims(f_val, axis=0)
	# 			p_pred = clf.predict_proba(f_val)
	# 			y1 = np.zeros((1, classes[c1]))
	# 			y1[0, y_val] = 1
	# 			min_la_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	#
	# 	min_all_test = [0] * len(pipeline['all'])
	# 	min_all1_test = [0] * len(pipeline['all'])
	# 	min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
	# 	min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
	# 	for j in range(len(min_all_pi)):
	# 		pipeline1 = min_all_pi[j]._values
	# 		hyper = {}
	# 		for v in pipeline1.keys():
	# 			for h in hypers:
	# 				s1 = v.find(h)
	# 				if s1 != -1:
	# 					hyper[h] = pipeline1[v]
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 										  dr=pipeline1['dimensionality_reduction'],
	# 										  la=pipeline1['learning_algorithm'],
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_all_test[j] = g.get_error()
	# 		clf = g.model
	# 		fe_test = g.ft_test
	# 		y_test = g.y_test
	# 		p_pred = g.p_pred
	# 		for k in range(len(y_test)):
	# 			f_val = fe_test[k, :]
	# 			y_val = y_test[k]
	# 			f_val = np.expand_dims(f_val, axis=0)
	# 			p_pred = clf.predict_proba(f_val)
	# 			y1 = np.zeros((1, classes[c1]))
	# 			y1[0, y_val] = 1
	# 			min_all_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	# 		pipeline1 = min_all1_pi[j]._values
	# 		hyper = {}
	# 		for v in pipeline1.keys():
	# 			for h in hypers:
	# 				s1 = v.find(h)
	# 				if s1 != -1:
	# 					hyper[h] = pipeline1[v]
	# 		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
	# 										  data_loc=data_home, type1='bayesian1', fe=pipeline1['feature_extraction'],
	# 										  dr=pipeline1['dimensionality_reduction'],
	# 										  la=pipeline1['learning_algorithm'],
	# 										  val_splits=3, test_size=0.2)
	# 		g.run()
	# 		min_all1_test[j] = g.get_error()
	# 		clf = g.model
	# 		fe_test = g.fe_test
	# 		y_test = g.y_test
	# 		p_pred = g.p_pred
	# 		for k in range(len(y_test)):
	# 			f_val = fe_test[k, :]
	# 			y_val = y_test[k]
	# 			f_val = np.expand_dims(f_val, axis=0)
	# 			p_pred = clf.predict_proba(f_val)
	# 			y1 = np.zeros((1, classes[c1]))
	# 			y1[0, y_val] = 1
	# 			min_all1_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
	# 	for j in range(len(s)):
	# 		if s[j] == 0:
	# 			min_alls_test[j] = min_all1_val[j]
	# 			min_alls_val[j] = min_all1_test[j]
	# 			min_alls_tests[:, j] = min_all1_tests[:, j]
	# 		else:
	# 			min_alls_val[j] /= s[j]
	# 			min_alls_test[j] /= s[j]
	# 			min_alls_tests[:, j] /= s[j]
	#
	# 	errors = np.zeros((len(y_test), 3))
	# 	errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
	# 	errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
	# 	errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
	# 	step_error_tests[run - 1, :, :] = errors
	# 	a = np.tile(np.expand_dims(min_err_tests, 1), (1, 7))
	# 	alg_error_tests[run - 1, :, :] = min_all_tests - a
	# 	alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests
	#
	# std_error = np.std(step_error_tests, 0)
	# step_error = np.mean(step_error_tests, 0)
	# bayesian1_step_error_tests = step_error.astype('float32')
	# bayesian1_std_error_tests = std_error.astype('float32')
	# std_error = np.std(alg_error_tests, 0)
	# step_error = np.mean(alg_error_tests, 0)
	# bayesian1_alg_error_tests = step_error.astype('float32')
	# bayesian1_alg_std_error_tests = std_error.astype('float32')
	# std_error = np.std(alg1_error_tests, 0)
	# step_error = np.mean(alg1_error_tests, 0)
	# bayesian1_alg1_error_tests = step_error.astype('float32')
	# bayesian1_alg1_std_error_tests = std_error.astype('float32')
	#
	# test_results_bayesian1 = [bayesian1_step_error_tests, bayesian1_std_error_tests, bayesian1_alg_error_tests,
	# 						  bayesian1_alg_std_error_tests,
	# 						  bayesian1_alg1_error_tests, bayesian1_alg1_std_error_tests]

	# Bayesian
	type1 = 'bayesian_MCMC'
	start = 1
	stop = 6
	data_home1 = data_home + 'datasets/' + data_name + '/'
	l1 = os.listdir(data_home1)
	y = []
	names = []
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home1 + l1[z] + '/')
		y += [z] * len(l)
		for i in range(len(l)):
			names.append(data_home1 + l1[z] + '/' + l[i])
	X = np.empty((len(y), 1))
	indices = np.arange(len(y))
	_, _, _, y_test, _, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	step_error_tests = np.zeros((stop - 1, len(y_test), 3))
	alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
	alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))

	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '_parallel.pkl', 'rb'), encoding='latin1')

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
		s = [0] * len(pipeline['all'])
		best_pi = None
		pipelines = obj.best_pipelines
		paths = obj.paths
		pipeline_errors = obj.error_curves
		for i in range(len(pipelines)):
			err = pipeline_errors[i]
			pie = np.amin(err)
			path = paths[i]
			pi = pipelines[i]
			if pie < min_err_val:
				min_err_val = pie
				best_pi = (pi, path)

			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if path[0] == alg:
					if pie < min_fe_val[j]:
						min_fe_val[j] = pie
						min_fe_pi[j] = (pi, path)

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if path[1] == alg:
					if pie < min_dr_val[j]:
						min_dr_val[j] = pie
						min_dr_pi[j] = (pi, path)

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg:
					if pie < min_la_val[j]:
						min_la_val[j] = pie
						min_la_pi[j] = (pi, path)

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg or path[0] == alg or path[1] == alg:
					if pie < min_all1_val[j]:
						min_all1_val[j] = pie
						min_all1_pi[j] = (pi, path)

					min_alls_val[j] += pie
					pipeline1 = pi._values
					hyper = {}
					for v in pipeline1.keys():
						for h in hypers:
							s1 = v.find(h)
							if s1 != -1:
								hyper[h] = pipeline1[v]
					g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
													  data_loc=data_home, type1='bayesian',
													  fe=path[0],
													  dr=path[1],
													  la=path[2],
													  val_splits=3, test_size=0.2)
					g.run()

					min_alls_test[j] += g.get_error()
					clf = g.model
					fe_test = g.ft_test
					y_pred = g.y_pred
					y_test = g.y_test
					p_pred = g.p_pred
					for k in range(len(y_test)):
						f_val = fe_test[k, :]
						y_val = y_test[k]
						f_val = np.expand_dims(f_val, axis=0)
						p_pred = clf.predict_proba(f_val)
						y1 = np.zeros((1, classes[c1]))
						if y_val >= classes[c1]:
							y1[0, y_val-1] = 1
						else:
							y1[0, y_val] = 1
						min_alls_tests[k, j] += metrics.log_loss(y_true=y1, y_pred=p_pred)
					s[j] += 1
				else:
					if pie < min_all_val[j]:
						min_all_val[j] = pie
						min_all_pi[j] = (pi, path)

		## Test results for all test samples
		pipeline1 = best_pi[0]._values
		path = best_pi[1]
		hyper = {}
		for v in pipeline1.keys():
			for h in hypers:
				s1 = v.find(h)
				if s1 != -1:
					hyper[h] = pipeline1[v]
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='bayesian',
										  fe=path[0],
										  dr=path[1],
										  la=path[2],
										  val_splits=3, test_size=0.2)
		g.run()
		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		p_pred = g.p_pred
		min_err_tests = [0] * len(y_test)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			f_val = np.expand_dims(f_val, axis=0)
			p_pred = clf.predict_proba(f_val)
			y1 = np.zeros((1, classes[c1]))
			if y_val >= classes[c1]:
				y1[0, y_val - 1] = 1
			else:
				y1[0, y_val] = 1
			min_err_tests[k] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		min_err_tests = np.asarray(min_err_tests)

		min_fe_test = [0] * len(pipeline['feature_extraction'])
		min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
		for j in range(len(min_fe_pi)):
			pipeline1 = min_fe_pi[j][0]._values
			path = min_fe_pi[j][1]
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='bayesian',
											  fe=path[0],
											  dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			min_fe_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_fe_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
		min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))
		for j in range(len(min_dr_pi)):
			pipeline1 = min_dr_pi[j][0]._values
			path = min_dr_pi[j][1]
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='bayesian',
											  fe=path[0],
											  dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			min_dr_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_dr_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_la_test = [0] * len(pipeline['learning_algorithm'])
		min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
		for j in range(len(min_la_pi)):
			pipeline1 = min_la_pi[j][0]._values
			path = min_la_pi[j][1]
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='bayesian',
											  fe=path[0],
											  dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			min_la_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_la_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_all_test = [0] * len(pipeline['all'])
		min_all1_test = [0] * len(pipeline['all'])
		min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
		min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
		for j in range(len(min_all_pi)):
			pipeline1 = min_all_pi[j][0]._values
			path = min_all_pi[j][1]
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='bayesian',
											  fe=path[0],
											  dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			min_all_test[j] = g.get_error()
			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
			pipeline1 = min_all1_pi[j][0]._values
			path = min_all1_pi[j][1]
			hyper = {}
			for v in pipeline1.keys():
				for h in hypers:
					s1 = v.find(h)
					if s1 != -1:
						hyper[h] = pipeline1[v]
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='bayesian',
											  fe=path[0],
											  dr=path[1],
											  la=path[2],
											  val_splits=3, test_size=0.2)
			g.run()
			min_all1_test[j] = g.get_error()
			clf = g.model
			fe_test = g.fe_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all1_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		for j in range(len(s)):
			if s[j] == 0:
				min_alls_test[j] = min_all1_val[j]
				min_alls_val[j] = min_all1_test[j]
				min_alls_tests[:, j] = min_all1_tests[:, j]
			else:
				min_alls_val[j] /= s[j]
				min_alls_test[j] /= s[j]
				min_alls_tests[:, j] /= s[j]

		errors = np.zeros((len(y_test), 3))
		errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
		errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
		errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
		step_error_tests[run - 1, :, :] = errors
		a = np.tile(np.expand_dims(min_err_tests, 1), (1, 7))
		alg_error_tests[run - 1, :, :] = min_all_tests - a
		alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests

	std_error = np.std(step_error_tests, 0)
	step_error = np.mean(step_error_tests, 0)
	bayesian_step_error_tests = step_error.astype('float32')
	bayesian_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg_error_tests, 0)
	step_error = np.mean(alg_error_tests, 0)
	bayesian_alg_error_tests = step_error.astype('float32')
	bayesian_alg_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg1_error_tests, 0)
	step_error = np.mean(alg1_error_tests, 0)
	bayesian_alg1_error_tests = step_error.astype('float32')
	bayesian_alg1_std_error_tests = std_error.astype('float32')

	test_results_bayesian = [bayesian_step_error_tests, bayesian_std_error_tests, bayesian_alg_error_tests,
							 bayesian_alg_std_error_tests,
							 bayesian_alg1_error_tests, bayesian_alg1_std_error_tests]

	# Random
	type1 = 'random_MCMC'
	start = 1
	stop = 6
	data_home1 = data_home + 'datasets/' + data_name + '/'
	l1 = os.listdir(data_home1)
	y = []
	names = []
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home1 + l1[z] + '/')
		y += [z] * len(l)
		for i in range(len(l)):
			names.append(data_home1 + l1[z] + '/' + l[i])
	X = np.empty((len(y), 1))
	indices = np.arange(len(y))
	_, _, _, y_test, _, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	step_error_tests = np.zeros((stop - 1, len(y_test), 3))
	alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
	alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))

	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

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
		s = [0] * len(pipeline['all'])
		best_pi = None
		pipelines = obj.pipelines
		error_curves = []
		for i in range(len(pipelines)):
			p = pipelines[i]
			objects = []
			for j in range(len(p)):
				objects.append(p[j].get_error())
			error_curves.append(objects)
		paths = obj.paths
		for i in range(len(pipelines)):
			err = error_curves[i]
			pie = np.amin(err)
			pie_arg = np.argmin(err)
			path = paths[i]
			pi = pipelines[i]
			if pie < min_err_val:
				min_err_val = pie
				best_pi = pi[pie_arg]

			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if path[0] == alg:
					if pie < min_fe_val[j]:
						min_fe_val[j] = pie
						min_fe_pi[j] = pi[pie_arg]

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if path[1] == alg:
					if pie < min_dr_val[j]:
						min_dr_val[j] = pie
						min_dr_pi[j] = pi[pie_arg]

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg:
					if pie < min_la_val[j]:
						min_la_val[j] = pie
						min_la_pi[j] = pi[pie_arg]

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg or path[0] == alg or path[1] == alg:
					if pie < min_all1_val[j]:
						min_all1_val[j] = pie
						min_all1_pi[j] = pi[pie_arg]

					min_alls_val[j] += pie
					hyper = pi[pie_arg].kwargs
					g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
													  data_loc=data_home, type1='random',
													  fe=pi[pie_arg].feature_extraction,
													  dr=pi[pie_arg].dimensionality_reduction,
													  la=pi[pie_arg].learning_algorithm,
													  val_splits=3, test_size=0.2)
					g.run()

					min_alls_test[j] += g.get_error()
					clf = g.model
					fe_test = g.ft_test
					y_pred = g.y_pred
					y_test = g.y_test
					p_pred = g.p_pred
					for k in range(len(y_test)):
						f_val = fe_test[k, :]
						y_val = y_test[k]
						f_val = np.expand_dims(f_val, axis=0)
						p_pred = clf.predict_proba(f_val)
						y1 = np.zeros((1, classes[c1]))
						if y_val >= classes[c1]:
							y1[0, y_val - 1] = 1
						else:
							y1[0, y_val] = 1
						min_alls_tests[k, j] += metrics.log_loss(y_true=y1, y_pred=p_pred)
					s[j] += 1
				else:
					if pie < min_all_val[j]:
						min_all_val[j] = pie
						min_all_pi[j] = pi[pie_arg]

		## Test results for all test samples
		hyper = best_pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random',
										  fe=best_pi.feature_extraction,
										  dr=best_pi.dimensionality_reduction,
										  la=best_pi.learning_algorithm,
										  val_splits=3, test_size=0.2)
		g.run()
		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		p_pred = g.p_pred
		min_err_tests = [0] * len(y_test)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			f_val = np.expand_dims(f_val, axis=0)
			p_pred = clf.predict_proba(f_val)
			y1 = np.zeros((1, classes[c1]))
			if y_val >= classes[c1]:
				y1[0, y_val - 1] = 1
			else:
				y1[0, y_val] = 1
			min_err_tests[k] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		min_err_tests = np.asarray(min_err_tests)

		min_fe_test = [0] * len(pipeline['feature_extraction'])
		min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
		for j in range(len(min_fe_pi)):
			hyper = min_fe_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='grid',
											  fe=min_fe_pi[j].feature_extraction,
											  dr=min_fe_pi[j].dimensionality_reduction,
											  la=min_fe_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			g.run()
			min_fe_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_fe_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
		min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))

		for j in range(len(min_dr_pi)):
			hyper = min_dr_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random',
											  fe=min_dr_pi[j].feature_extraction,
											  dr=min_dr_pi[j].dimensionality_reduction,
											  la=min_dr_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_dr_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_dr_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_la_test = [0] * len(pipeline['learning_algorithm'])
		min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
		for j in range(len(min_la_pi)):
			hyper = min_la_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random',
											  fe=min_la_pi[j].feature_extraction,
											  dr=min_la_pi[j].dimensionality_reduction,
											  la=min_la_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_la_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_la_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_all_test = [0] * len(pipeline['all'])
		min_all1_test = [0] * len(pipeline['all'])
		min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
		min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
		for j in range(len(min_all_pi)):
			hyper = min_all_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random',
											  fe=min_all_pi[j].feature_extraction,
											  dr=min_all_pi[j].dimensionality_reduction,
											  la=min_all_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_all_test[j] = g.get_error()
			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
			hyper = min_all1_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random',
											  fe=min_all1_pi[j].feature_extraction,
											  dr=min_all1_pi[j].dimensionality_reduction,
											  la=min_all1_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_all1_test[j] = g.get_error()
			clf = g.model
			fe_test = g.fe_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all1_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		for j in range(len(s)):
			if s[j] == 0:
				min_alls_test[j] = min_all1_val[j]
				min_alls_val[j] = min_all1_test[j]
				min_alls_tests[:, j] = min_all1_tests[:, j]
			else:
				min_alls_val[j] /= s[j]
				min_alls_test[j] /= s[j]
				min_alls_tests[:, j] /= s[j]

		errors = np.zeros((len(y_test), 3))
		errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
		errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
		errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
		step_error_tests[run - 1, :, :] = errors
		a = np.tile(np.expand_dims(min_err_tests, 1), (1, 7))
		alg_error_tests[run - 1, :, :] = min_all_tests - a
		alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests

	std_error = np.std(step_error_tests, 0)
	step_error = np.mean(step_error_tests, 0)
	random_step_error_tests = step_error.astype('float32')
	random_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg_error_tests, 0)
	step_error = np.mean(alg_error_tests, 0)
	random_alg_error_tests = step_error.astype('float32')
	random_alg_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg1_error_tests, 0)
	step_error = np.mean(alg1_error_tests, 0)
	random_alg1_error_tests = step_error.astype('float32')
	random_alg1_std_error_tests = std_error.astype('float32')
	test_results_random = [random_step_error_tests, random_std_error_tests, random_alg_error_tests, random_alg_std_error_tests,
					random_alg1_error_tests, random_alg1_std_error_tests]

	# Grid
	type1 = 'grid_MCMC'
	start = 1
	stop = 2
	data_home1 = data_home + 'datasets/' + data_name + '/'
	l1 = os.listdir(data_home1)
	y = []
	names = []
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home1 + l1[z] + '/')
		y += [z] * len(l)
		for i in range(len(l)):
			names.append(data_home1 + l1[z] + '/' + l[i])
	X = np.empty((len(y), 1))
	indices = np.arange(len(y))
	_, _, _, y_test, _, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	step_error_tests = np.zeros((stop - 1, len(y_test), 3))
	alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
	alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))

	for run in range(start, stop):
		obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
							   str(run) + '.pkl', 'rb'), encoding='latin1')

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
		s = [0] * len(pipeline['all'])
		best_pi = None
		pipelines = obj.pipelines
		error_curves = []
		for i in range(len(pipelines)):
			p = pipelines[i]
			objects = []
			for j in range(len(p)):
				objects.append(p[j].get_error())
			error_curves.append(objects)
		paths = obj.paths
		for i in range(len(pipelines)):
			err = error_curves[i]
			pie = np.amin(err)
			pie_arg = np.argmin(err)
			path = paths[i]
			pi = pipelines[i]
			if pie < min_err_val:
				min_err_val = pie
				best_pi = pi[pie_arg]

			p = pipeline['feature_extraction']
			for j in range(len(p)):
				alg = p[j]
				if path[0] == alg:
					if pie < min_fe_val[j]:
						min_fe_val[j] = pie
						min_fe_pi[j] = pi[pie_arg]

			p = pipeline['dimensionality_reduction']
			for j in range(len(p)):
				alg = p[j]
				if path[1] == alg:
					if pie < min_dr_val[j]:
						min_dr_val[j] = pie
						min_dr_pi[j] = pi[pie_arg]

			p = pipeline['learning_algorithm']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg:
					if pie < min_la_val[j]:
						min_la_val[j] = pie
						min_la_pi[j] = pi[pie_arg]

			p = pipeline['all']
			for j in range(len(p)):
				alg = p[j]
				if path[2] == alg or path[0] == alg or path[1] == alg:
					if pie < min_all1_val[j]:
						min_all1_val[j] = pie
						min_all1_pi[j] = pi[pie_arg]

					min_alls_val[j] += pie
					hyper = pi[pie_arg].kwargs
					g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
													  data_loc=data_home, type1='grid',
													  fe=pi[pie_arg].feature_extraction,
													  dr=pi[pie_arg].dimensionality_reduction,
													  la=pi[pie_arg].learning_algorithm,
													  val_splits=3, test_size=0.2)
					g.run()

					min_alls_test[j] += g.get_error()
					clf = g.model
					fe_test = g.ft_test
					y_pred = g.y_pred
					y_test = g.y_test
					p_pred = g.p_pred
					for k in range(len(y_test)):
						f_val = fe_test[k, :]
						y_val = y_test[k]
						f_val = np.expand_dims(f_val, axis=0)
						p_pred = clf.predict_proba(f_val)
						y1 = np.zeros((1, classes[c1]))
						if y_val >= classes[c1]:
							y1[0, y_val - 1] = 1
						else:
							y1[0, y_val] = 1
						min_alls_tests[k, j] += metrics.log_loss(y_true=y1, y_pred=p_pred)
					s[j] += 1
				else:
					if pie < min_all_val[j]:
						min_all_val[j] = pie
						min_all_pi[j] = pi[pie_arg]

		## Test results for all test samples
		hyper = best_pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='grid',
										  fe=best_pi.feature_extraction,
										  dr=best_pi.dimensionality_reduction,
										  la=best_pi.learning_algorithm,
										  val_splits=3, test_size=0.2)
		g.run()
		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		p_pred = g.p_pred
		min_err_tests = [0] * len(y_test)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			f_val = np.expand_dims(f_val, axis=0)
			p_pred = clf.predict_proba(f_val)
			y1 = np.zeros((1, classes[c1]))
			if y_val >= classes[c1]:
				y1[0, y_val - 1] = 1
			else:
				y1[0, y_val] = 1
			min_err_tests[k] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		min_err_tests = np.asarray(min_err_tests)

		min_fe_test = [0] * len(pipeline['feature_extraction'])
		min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
		for j in range(len(min_fe_pi)):
			hyper = min_fe_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='grid',
											  fe=min_fe_pi[j].feature_extraction,
											  dr=min_fe_pi[j].dimensionality_reduction,
											  la=min_fe_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_fe_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_fe_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
		min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))

		for j in range(len(min_dr_pi)):
			hyper = min_dr_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='grid',
											  fe=min_dr_pi[j].feature_extraction,
											  dr=min_dr_pi[j].dimensionality_reduction,
											  la=min_dr_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_dr_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_dr_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_la_test = [0] * len(pipeline['learning_algorithm'])
		min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
		for j in range(len(min_la_pi)):
			hyper = min_la_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
												  data_loc=data_home, type1='grid',
												  fe=min_la_pi[j].feature_extraction,
												  dr=min_la_pi[j].dimensionality_reduction,
												  la=min_la_pi[j].learning_algorithm,
												  val_splits=3, test_size=0.2)
			g.run()
			min_la_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_la_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_all_test = [0] * len(pipeline['all'])
		min_all1_test = [0] * len(pipeline['all'])
		min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
		min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
		for j in range(len(min_all_pi)):
			hyper = min_all_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='grid',
											  fe=min_all_pi[j].feature_extraction,
											  dr=min_all_pi[j].dimensionality_reduction,
											  la=min_all_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_all_test[j] = g.get_error()
			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
			hyper = min_all1_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='grid',
											  fe=min_all1_pi[j].feature_extraction,
											  dr=min_all1_pi[j].dimensionality_reduction,
											  la=min_all1_pi[j].learning_algorithm,
											  val_splits=3, test_size=0.2)
			g.run()
			min_all1_test[j] = g.get_error()
			clf = g.model
			fe_test = g.fe_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all1_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		for j in range(len(s)):
			if s[j] == 0:
				min_alls_test[j] = min_all1_val[j]
				min_alls_val[j] = min_all1_test[j]
				min_alls_tests[:, j] = min_all1_tests[:, j]
			else:
				min_alls_val[j] /= s[j]
				min_alls_test[j] /= s[j]
				min_alls_tests[:, j] /= s[j]

		errors = np.zeros((len(y_test), 3))
		errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
		errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
		errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
		step_error_tests[run - 1, :, :] = errors
		a = np.tile(np.expand_dims(min_err_tests, 1), (1, 7))
		alg_error_tests[run - 1, :, :] = min_all_tests - a
		alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests

	std_error = np.std(step_error_tests, 0)
	step_error = np.mean(step_error_tests, 0)
	grid_step_error_tests = step_error.astype('float32')
	grid_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg_error_tests, 0)
	step_error = np.mean(alg_error_tests, 0)
	grid_alg_error_tests = step_error.astype('float32')
	grid_alg_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg1_error_tests, 0)
	step_error = np.mean(alg1_error_tests, 0)
	grid_alg1_error_tests = step_error.astype('float32')
	grid_alg1_std_error_tests = std_error.astype('float32')

	test_results_grid = [grid_step_error_tests, grid_std_error_tests, grid_alg_error_tests, grid_alg_std_error_tests,
					grid_alg1_error_tests, grid_alg1_std_error_tests]



	# Random1
	type1 = 'random_MCMC'
	start = 1
	stop = 6
	data_home1 = data_home + 'datasets/' + data_name + '/'
	l1 = os.listdir(data_home1)
	y = []
	names = []
	for z in range(len(l1)):
		if l1[z][0] == '.':
			continue
		l = os.listdir(data_home1 + l1[z] + '/')
		y += [z] * len(l)
		for i in range(len(l)):
			names.append(data_home1 + l1[z] + '/' + l[i])
	X = np.empty((len(y), 1))
	indices = np.arange(len(y))
	_, _, _, y_test, _, _ = train_test_split(X, y, indices, test_size=0.2, random_state=42, shuffle=True)
	step_error_tests = np.zeros((stop - 1, len(y_test), 3))
	alg_error_tests = np.zeros((stop - 1, len(y_test), 7))
	alg1_error_tests = np.zeros((stop - 1, len(y_test), 7))

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
					for k in range(len(y_test)):
						f_val = fe_test[k, :]
						y_val = y_test[k]
						f_val = np.expand_dims(f_val, axis=0)
						p_pred = clf.predict_proba(f_val)
						y1 = np.zeros((1, classes[c1]))
						if y_val >= classes[c1]:
							y1[0, y_val - 1] = 1
						else:
							y1[0, y_val] = 1
						min_alls_tests[k, j] += metrics.log_loss(y_true=y1, y_pred=p_pred)
					s[j] += 1
				else:
					if pie < min_all_val[j]:
						min_all_val[j] = pie
						min_all_pi[j] = pi

		## Test results for all test samples
		hyper = best_pi.kwargs
		g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
										  data_loc=data_home, type1='random1',
										  fe=best_pi.feature_extraction.decode('latin1'),
										  dr=best_pi.dimensionality_reduction.decode('latin1'),
										  la=best_pi.learning_algorithm.decode('latin1'),
										  val_splits=3, test_size=0.2)
		g.run()
		clf = g.model
		fe_test = g.ft_test
		y_test = g.y_test
		p_pred = g.p_pred
		min_err_tests = [0] * len(y_test)
		for k in range(len(y_test)):
			f_val = fe_test[k, :]
			y_val = y_test[k]
			f_val = np.expand_dims(f_val, axis=0)
			p_pred = clf.predict_proba(f_val)
			y1 = np.zeros((1, classes[c1]))
			if y_val >= classes[c1]:
				y1[0, y_val - 1] = 1
			else:
				y1[0, y_val] = 1
			min_err_tests[k] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		min_err_tests = np.asarray(min_err_tests)

		min_fe_test = [0] * len(pipeline['feature_extraction'])
		min_fe_tests = np.zeros((len(y_test), len(pipeline['feature_extraction'])))
		for j in range(len(min_fe_pi)):
			hyper = min_fe_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_fe_pi[j].feature_extraction.decode('latin1'),
											  dr=min_fe_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_fe_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_fe_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			test = g.fe_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_fe_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_dr_test = [0] * len(pipeline['dimensionality_reduction'])
		min_dr_tests = np.zeros((len(y_test), len(pipeline['dimensionality_reduction'])))

		for j in range(len(min_dr_pi)):
			hyper = min_dr_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_dr_pi[j].feature_extraction.decode('latin1'),
											  dr=min_dr_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_dr_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_dr_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_dr_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_la_test = [0] * len(pipeline['learning_algorithm'])
		min_la_tests = np.zeros((len(y_test), len(pipeline['learning_algorithm'])))
		for j in range(len(min_la_pi)):
			hyper = min_la_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_la_pi[j].feature_extraction.decode('latin1'),
											  dr=min_la_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_la_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_la_test[j] = g.get_error()

			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_la_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)

		min_all_test = [0] * len(pipeline['all'])
		min_all1_test = [0] * len(pipeline['all'])
		min_all_tests = np.zeros((len(y_test), len(pipeline['all'])))
		min_all1_tests = np.zeros((len(y_test), len(pipeline['all'])))
		for j in range(len(min_all_pi)):
			hyper = min_all_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_all_pi[j].feature_extraction.decode('latin1'),
											  dr=min_all_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_all_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_all_test[j] = g.get_error()
			clf = g.model
			fe_test = g.ft_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
			hyper = min_all1_pi[j].kwargs
			g = image_classification_pipeline(hyper, ml_type='testing', data_name=data_name,
											  data_loc=data_home, type1='random1',
											  fe=min_all1_pi[j].feature_extraction.decode('latin1'),
											  dr=min_all1_pi[j].dimensionality_reduction.decode('latin1'),
											  la=min_all1_pi[j].learning_algorithm.decode('latin1'),
											  val_splits=3, test_size=0.2)
			g.run()
			min_all1_test[j] = g.get_error()
			clf = g.model
			fe_test = g.fe_test
			y_test = g.y_test
			p_pred = g.p_pred
			for k in range(len(y_test)):
				f_val = fe_test[k, :]
				y_val = y_test[k]
				f_val = np.expand_dims(f_val, axis=0)
				p_pred = clf.predict_proba(f_val)
				y1 = np.zeros((1, classes[c1]))
				if y_val >= classes[c1]:
					y1[0, y_val - 1] = 1
				else:
					y1[0, y_val] = 1
				min_all1_tests[k, j] = metrics.log_loss(y_true=y1, y_pred=p_pred)
		for j in range(len(s)):
			if s[j] == 0:
				min_alls_test[j] = min_all1_val[j]
				min_alls_val[j] = min_all1_test[j]
				min_alls_tests[:, j] = min_all1_tests[:, j]
			else:
				min_alls_val[j] /= s[j]
				min_alls_test[j] /= s[j]
				min_alls_tests[:, j] /= s[j]

		errors = np.zeros((len(y_test), 3))
		errors[:, 0] = np.mean(min_fe_tests, 1) - min_err_tests
		errors[:, 1] = np.mean(min_dr_tests, 1) - min_err_tests
		errors[:, 2] = np.mean(min_la_tests, 1) - min_err_tests
		step_error_tests[run - 1, :, :] = errors
		a = np.tile(np.expand_dims(min_err_tests, 1), (1, 7))
		alg_error_tests[run - 1, :, :] = min_all_tests - a
		alg1_error_tests[run - 1, :, :] = min_alls_tests - min_all1_tests

	std_error = np.std(step_error_tests, 0)
	step_error = np.mean(step_error_tests, 0)
	random1_step_error_tests = step_error.astype('float32')
	random1_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg_error_tests, 0)
	step_error = np.mean(alg_error_tests, 0)
	random1_alg_error_tests = step_error.astype('float32')
	random1_alg_std_error_tests = std_error.astype('float32')
	std_error = np.std(alg1_error_tests, 0)
	step_error = np.mean(alg1_error_tests, 0)
	random1_alg1_error_tests = step_error.astype('float32')
	random1_alg1_std_error_tests = std_error.astype('float32')

	test_results_random1 = [random1_step_error_tests, random1_std_error_tests, random1_alg_error_tests, random1_alg_std_error_tests,
					random1_alg1_error_tests, random1_alg1_std_error_tests]
	pickle.dump([test_results_grid, test_results_random, test_results_random1, test_results_bayesian], open(results_home + 'intermediate/' + data_name + '_test_error_correlation.pkl', 'wb'))
