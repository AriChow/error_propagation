import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

home = os.path.expanduser('~')
data_name = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'


# Errors
bayesian_errors = []
bayesian_times = []
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for run in range(1, 6):
	type1 = 'bayesian_MCMC'
	bayesian_obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
						   str(run) + '_full.pkl', 'rb'))
	bayesian_errors.append(bayesian_obj.error_curves[0][-1])
	bayesian_times.append(bayesian_obj.times[-1])


	axs[(run - 1) // 3, (run - 1) % 3].plot(range(len(bayesian_obj.error_curves[0])), bayesian_obj.error_curves[0], 'g', label='bayesian')
	axs[(run - 1) // 3, (run - 1) % 3].set_xlabel('Iterations')
	axs[(run - 1) // 3, (run - 1) % 3].set_ylabel('Errors')
	axs[(run - 1) // 3, (run - 1) % 3].legend()
	axs[(run - 1) // 3, (run - 1) % 3].set_title('Run ' + str(run))

plt.savefig(results_home + 'figures/error_curves_bayesian_' + data_name + '.jpg')
plt.close()

mean_errors = [np.mean(bayesian_errors)]
std_errors = [np.std(bayesian_errors)]
mean_errors = np.asarray(mean_errors)
std_errors = np.asarray((std_errors))

mean_times = [np.mean(bayesian_times)]
std_times = [np.std(bayesian_times)]
mean_times = np.asarray(mean_times)
std_times = np.asarray(std_times)

x = ['Bayesian']
plt.figure(1, figsize=(12, 4))

plt.subplot(121)
plt.errorbar(range(1, 2), mean_errors, std_errors, linestyle='None', marker='^', capsize=3)
plt.title('Errors')
plt.xlabel('Algorithm')
plt.ylabel('Errors (log-loss)')
plt.xticks(range(1, 4), x)

plt.subplot(122)
plt.errorbar(range(1, 2), mean_times, std_times, linestyle='None', marker='^', capsize=3)
plt.title('Times')
plt.xlabel('Algorithm')
plt.ylabel('Runtime(s)')
plt.xticks(range(1, ), x)
plt.savefig(results_home + 'figures/error_time_bayesian_' + data_name + '.jpg')
plt.close()
