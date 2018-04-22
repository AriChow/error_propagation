
import numpy as np
import os
import sys
import pickle
from prototypes.data_analytic_pipeline import image_classification_pipeline

home = os.path.expanduser('~')
data_name = 'breast'
data_home = home + '/Documents/research/EP_project/data/'
results_home = home + '/Documents/research/EP_project/results/'

# Specification of pipeline
pipeline = {}
pipeline['feature_extraction'] = ["VGG", "haralick", "inception"]
pipeline['dimensionality_reduction'] = ["PCA", "ISOMAP"]
pipeline['learning_algorithm'] = ["SVM", "RF"]
pipeline['all'] = pipeline['feature_extraction'] + pipeline['dimensionality_reduction'] + pipeline['learning_algorithm']



# Grid search
type1 = 'grid_MCMC'
obj = pickle.load(open(results_home + 'intermediate/' + type1 + '/' + type1 + '_' + data_name + '_run_' +
					str(1) + '.pkl', 'rb'), encoding='latin1')

print
