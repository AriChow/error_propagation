import pickle
import os

home = os.path.expanduser('~')
dataset = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'
g = pickle.load(open(results_home + 'intermediate/smac_control_' + dataset + '.pkl', 'rb'))

gfe = pickle.load(open(results_home + 'intermediate/smac_feature_extraction_' + dataset + '.pkl', 'rb'))

gdr = pickle.load(open(results_home + 'intermediate/smac_learning_algorithm_' + dataset + '.pkl', 'rb'))
