import pickle
import os

home = os.path.expanduser('~')
dataset = 'breast'
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
for i in range(2, max_iters):
	[_, _, err] = pickle.load(open(results_home + 'intermediate/RL_MCMC/RL_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	errors.append(err)
	[_, _, err] = pickle.load(open(results_home + 'intermediate/random_MCMC/random_MCMC_' + dataset + '_iter_' + str(i) + '.pkl', 'rb'))
	errors1.append(err)

plt.plot(range(1, max_iters-1), errors)
plt.plot(range(1, max_iters-1), errors1, 'r')
plt.xlabel('Iterations')
plt.ylabel('Errors')
plt.savefig(results_home + 'figures/MCMC_error.jpg')



