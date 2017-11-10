import pickle
import os
import numpy as np

home = os.path.expanduser('~')
dataset = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
g = pickle.load(open(results_home + 'intermediate/random_search_' + dataset + '.pkl', 'rb'))

trials = g.trials
errors = g.error
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]

min_err = np.argmin(errors)
t0 = trials[min_err]
e0 = errors[min_err]
ef = [np.Inf] * len(pipeline['feature_extraction'])
ed = [np.Inf] * len(pipeline['dimensionality_reduction'])
el = [np.Inf] * len(pipeline['learning_algorithm'])
for i in range(len(errors)):
	t = trials[i]
	p = pipeline['feature_extraction']
	for j in range(len(p)):
		f = p[j]
		if f in t:
			if errors[i] < ef[j]:
				ef[j] = errors[i]
	p = pipeline['dimensionality_reduction']
	for j in range(len(p)):
		f = p[j]
		if f in t:
			if errors[i] < ef[j]:
				ed[j] = errors[i]
	p = pipeline['learning_algorithm']
	for j in range(len(p)):
		f = p[j]
		if f in t:
			if errors[i] < ef[j]:
				el[j] = errors[i]

f = open(results_home + 'experiments/random_search_errors_' + dataset + '.txt', 'w')
f.write("Error from feature extraction :" + str(np.mean(ef) - e0) + '\n')
f.write("Error from dimensionality reduction :" + str(np.mean(ed) - e0) + '\n')
f.write("Error from learning algorithms :" + str(np.mean(el) - e0) + '\n')

