import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

home = os.path.expanduser('~')
data_name = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]

# Bayesian
type1 = 'bayesian_MCMC'
error = np.zeros((1, 3))
for run in range(1, 6):
	if run == 2:
		continue
	obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '.pkl','rb'))

	min_err = 100000000
	min_fe = [1000000] * len(pipeline['feature_extraction'])
	min_dr = [1000000] * len(pipeline['dimensionality_reduction'])
	min_la = [1000000] * len(pipeline['learning_algorithm'])
	for i in range(len(obj.error_curves)):
		err = obj.error_curves[i]
		if err[-1] < min_err:
			min_err = err[-1]
		p = pipeline['feature_extraction']
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][0] == alg:
				if err[-1] < min_fe[j]:
					min_fe[j] = err[-1]

		p = pipeline['dimensionality_reduction']
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][1] == alg:
				if err[-1] < min_dr[j]:
					min_dr[j] = err[-1]

		p = pipeline['learning_algorithm']
		for j in range(len(p)):
			alg = p[j]
			if obj.paths[i][2] == alg:
				if err[-1] < min_la[j]:
					min_la[j] = err[-1]

	errors = [np.mean(min_fe) - min_err, np.mean(min_dr) - min_err, np.mean(min_la) - min_err]
	errors = np.asarray(errors)
	error = np.vstack((error, errors))

error = np.delete(error, 0, 0)
std_error = np.std(error, 0)
error = np.mean(error, 0)
error = error.astype('float32')
std_error = std_error.astype('float32')
x = ['FE', 'DR', 'LA']
plt.figure(1, figsize=(12, 4))

# plt.subplot(131)
# plt.errorbar(range(1, 4), error[0, :], std_error[:, :], linestyle='None', marker='^', capsize=3)
# plt.title('Random')
# plt.xlabel('Agnostic Step')
# plt.ylabel('Errors')
# plt.xticks(range(1, 4), x)

plt.subplot(111)
plt.errorbar(range(1, 4), error, std_error, linestyle='None', marker='^', capsize=3)
plt.title('Bayesian')
plt.xlabel('Agnostic Step')
plt.ylabel('Errors')
plt.xticks(range(1, 4), x)
# plt.show()
plt.savefig(results_home + 'figures/agnostic_errors_bayesian_' + data_name + '.jpg')
plt.close()

