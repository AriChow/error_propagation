import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

home = os.path.expanduser('~')
data_name = sys.argv[1]
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'


# Errors
random_errors = []
bayesian_errors = []
mcmc_errors = []

random_times = []
bayesian_times = []
mcmc_times = []
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
for run in range(1, 6):
	# type1 = 'bayesian_MCMC'
	# bayesian_obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
	# 					   str(run) + '_full.pkl', 'rb'))
	# bayesian_errors.append(bayesian_obj.error_curves[0][-1])
	# bayesian_times.append(bayesian_obj.times[-1])

	type1 = 'random_MCMC'
	random_obj = pickle.load(open(results_home + 'intermediate_CCNI/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'), encoding='latin1')

	type1 = 'RL_MCMC'
	mcmc_obj1 = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full_multivariate.pkl', 'rb'), encoding='latin1')
	type1 = 'RL_MCMC'
	mcmc_obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
								 str(run) + '_full_univariate.pkl', 'rb'), encoding='latin1')

	axs[(run - 1)].plot(mcmc_obj.times, mcmc_obj.error_curve[:len(mcmc_obj.times)], label='Stochastic based search (univariate)')
	axs[(run - 1)].plot(random_obj.times, random_obj.error_curve, 'r', label='Random search')
	axs[(run - 1)].plot(mcmc_obj1.times, mcmc_obj1.error_curve[:len(mcmc_obj1.times)], 'g', label='Stochastic based search (multivariate)')

	# axs[(run - 1)].plot(range(len(bayesian_obj.error_curves[0])), bayesian_obj.error_curves[0], 'g', label='bayesian')
	axs[(run - 1)].set_xlabel('Run-time (s)')
	axs[(run - 1)].set_ylabel('Cross-validation error')
	axs[(run - 1)].legend()
	axs[(run - 1)].set_title('Run ' + str(run))

plt.savefig(results_home + 'figures/error_curves_' + data_name + '.jpg')
plt.close()

# mean_errors = [np.mean(random_errors), np.mean(bayesian_errors), np.mean(mcmc_errors)]
# std_errors = [np.std(random_errors), np.std(bayesian_errors), np.std(mcmc_errors)]
# mean_errors = np.asarray(mean_errors)
# std_errors = np.asarray((std_errors))
#
# mean_times = [np.mean(random_times), np.mean(bayesian_times), np.mean(mcmc_times)]
# std_times = [np.std(random_times), np.std(bayesian_times), np.std(mcmc_times)]
# mean_times = np.asarray(mean_times)
# std_times = np.asarray(std_times)
#
# x = ['Random', 'Bayesian', 'MCMC']
# plt.figure(1, figsize=(12, 4))
#
# plt.subplot(121)
# plt.errorbar(range(1, 4), mean_errors, std_errors, linestyle='None', marker='^', capsize=3)
# plt.title('Errors')
# plt.xlabel('Algorithm')
# plt.ylabel('Errors (log-loss)')
# plt.xticks(range(1, 4), x)
#
# plt.subplot(122)
# plt.errorbar(range(1, 4), mean_times, std_times, linestyle='None', marker='^', capsize=3)
# plt.title('Times')
# plt.xlabel('Algorithm')
# plt.ylabel('Runtime(s)')
# plt.xticks(range(1, 4), x)
# plt.savefig(results_home + 'figures/error_time_multivariate' + data_name + '.jpg')
# plt.close()
