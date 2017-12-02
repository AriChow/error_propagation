import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

home = os.path.expanduser('~')
dataset = 'matsc_dataset2'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# # RL analysis
#
# f = open(results_home + 'experiments/mcmc_error_analysis.txt', 'a')
# f.write('RL ERROR ANALYSIS:\n')
#
# # CONTROL
# [_, best_pipeline, err0] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC.pkl', 'rb'))
# f.write('CONTROL \n')
# f.write('Best pipeline error:' + str(err0) + '\n')
# f.write('Feature extraction: ' + best_pipeline.feature_extraction + ', Dimensionality reduction: ' + best_pipeline.dimensionality_reduction + ', Learning Algorithm: ' + best_pipeline.learning_algorithm + '\n')
# f.write('Hyper-parameters: ')
# hypers = best_pipeline.kwargs
# for key, val in hypers.items():
# 	f.write(key + ': ' + str(val) + ' ')
#
# # FEATURE EXTRACTION AGNOSTIC
# [_, _, err1] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_VGG.pkl', 'rb'))
# [_, _, err2] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_inception.pkl', 'rb'))
# [_, _, err3] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_haralick.pkl', 'rb'))
# f.write('\nFEATURE EXTRACTION AGNOSTIC\n')
# err = (err1 + err2 + err3) / 3
# f.write('Error:' + str(err) + '\n')
# f.write('Error contributed by feature extraction: ' + str(err-err0))
#
# # DIMENSIONALITY REDUCTION AGNOSTIC
# [_, _, err1] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_PCA.pkl', 'rb'))
# [_, _, err2] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_ISOMAP.pkl', 'rb'))
# f.write('\nDIMENSIONALITY REDUCTION AGNOSTIC\n')
# err = (err1 + err2) / 2
# f.write('Error:' + str(err) + '\n')
# f.write('Error contributed by dimensionality reduction: ' + str(err-err0))
#
# # LEARNING ALGORITHM AGNOSTIC
# [_, _, err1] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_SVM.pkl', 'rb'))
# [_, _, err2] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_RF.pkl', 'rb'))
# f.write('\nLEARNING ALGORITHM AGNOSTIC\n')
# err = (err1 + err2) / 2
# f.write('Error:' + str(err) + '\n')
# f.write('Error contributed by learning algorithms: ' + str(err-err0) + '\n\n')


# PLOTS
import matplotlib.pyplot as plt
max_iters = 21
errors = []
errors1 = []
errors2 = []
for i in range(max_iters):
	[_, _, err, _] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	errors.append(err)
	[_, _, err, _] = pickle.load(open(results_home + 'intermediate/random_MCMC/random_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	errors1.append(err)
	[_, _, err, _] = pickle.load(open(results_home + 'intermediate/bayesian_MCMC/bayesian_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	errors2.append(err)

plt.plot(range(max_iters), errors, label='RL')
plt.plot(range(max_iters), errors1, 'r', label='random')
plt.plot(range(max_iters), errors2, 'g', label='bayesian')
plt.xlabel('Iterations')
plt.ylabel('Errors')
plt.legend()
plt.savefig(results_home + 'figures/RL_random_MCMC_bayesian_error_' + dataset +'.jpg')
plt.close()

# ERROR BARS
errors = pickle.load(open(results_home + 'intermediate/RL_MCMC/errors' + dataset +'.pkl', 'rb'))
err = errors[-1, :]

error = []
for i in range(len(err)-1):
	error.append(err[i] - err[-1])
m = min(error)
error /= m
error = np.asarray(error)
error = error.astype('int8')
error = error.tolist()
x = ['FE', 'DR', 'LA']
plt.figure(1, figsize=(12, 4))

# fig, ax = plt.subplots()
plt.subplot(131)
plt.bar(range(1, 4), error)
plt.title('RL')
plt.xlabel('Agnostic Step')
plt.ylabel('Relative errors')
plt.xticks(range(1, 4), x)
errors = pickle.load(open(results_home + 'intermediate/random_MCMC/errors' + dataset + '.pkl', 'rb'))
err = errors[-1, :]
error = []
for i in range(len(err)-1):
	error.append(err[i] - err[-1])
m = min(error)
error /= m
error = np.asarray(error)
error = error.astype('int8')
error = error.tolist()
plt.subplot(132)
plt.bar(range(1, 4), error)
plt.title('Random')
plt.xlabel('Agnostic Step')
plt.ylabel('Relative errors')
plt.xticks(range(1, 4), x)

errors = pickle.load(open(results_home + 'intermediate/bayesian_MCMC/errors' + dataset + '.pkl', 'rb'))
err = errors[-1, :]
error = []
for i in range(len(err)-1):
	error.append(err[i] - err[-1])
m = min(error)
error /= m
error = np.asarray(error)
error = error.astype('int8')
error = error.tolist()
plt.subplot(133)
plt.bar(range(1, 4), error)
plt.title('Bayesian')
plt.xlabel('Agnostic Step')
plt.ylabel('Relative errors')
plt.xticks(range(1, 4), x)
plt.savefig(results_home + 'figures/RL_random_bayesian_MCMC_error_bars_' + dataset + '.jpg')
plt.close()
