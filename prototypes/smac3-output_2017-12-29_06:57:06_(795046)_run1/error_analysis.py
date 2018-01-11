import pickle
import os

home = os.path.expanduser('~')
data_name = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
type1 = 'bayesian_MCMC'
run = 2
obj = pickle.load(open(results_home + 'intermediate/bayesian_MCMC/' + type1 + '_' + data_name + '_run_' +
					   str(run) + '.pkl','rb'))

print()
