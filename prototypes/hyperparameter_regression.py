import numpy as np
import os
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

rng = np.random.RandomState(0)

def gp_regression(X, y):
	gp = RBF()
	gpr = GaussianProcessRegressor(kernel=gp)
	gpr.fit(X, y)
	return gpr

def sv_regression(X, y):
	svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=3, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
																	   "gamma": np.logspace(-2, 2, 5),
																	   "kernel": ['linear', 'rbf']})
	svr.fit(X, y)
	return svr

def kr_regression(X, y):
	krr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3, param_grid={"gamma": np.logspace(-2, 2, 5),
																   "kernel": ['linear', 'rbf']})
	krr.fit(X, y)
	return krr

def regress(X, y):
	gpr = gp_regression(X, y)
	krr = kr_regression(X, y)
	svr = sv_regression(X, y)
	return (X, y, gpr, krr, svr)

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
					   str(1) + '.pkl', 'rb'))

paths = obj.paths
pipelines = obj.pipelines
models = []
for i in range(len(paths)):
	path = paths[i]
	pipeline = pipelines[i]
	X = []
	y = []
	for j in range(len(pipeline)):
		p = pipeline[j]
		x = []
		for k in p.kwargs:
			x.append(float(p.kwargs[k]))
		x = np.asarray(x)
		x = np.expand_dims(x, 0)
		if j == 0:
			X = x
		else:
			X = np.vstack((X, x))
		y.append(p.get_error())
	y = np.asarray(y)
	model = regress(X, y)
	models.append(model)
pickle.dump(models, open(results_home + 'intermediate/regression_models.pkl', 'wb'))
