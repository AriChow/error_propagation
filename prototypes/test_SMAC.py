import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

cs = ConfigurationSpace()

kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default="poly")
cs.add_hyperparameter(kernel)

C = UniformFloatHyperparameter("C", 0.001, 1000.0, default=1.0)
shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default="true")
cs.add_hyperparameters([C, shrinking])

degree = UniformIntegerHyperparameter("degree", 1, 5, default=3)
coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default=0.0)
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=coef0, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])

gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default="auto")
gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default=1)
cs.add_hyperparameters([gamma, gamma_value])
cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))



iris = load_digits()
def svm_from_cfg(cfg):
	cfg = {k : cfg[k] for k in cfg if cfg[k]}
	cfg["shrinking"] = True if cfg["shrinking"] == True else False
	if "gamma" in cfg:
		cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
		cfg.pop("gamma_value", None)
	clf = svm.SVC(**cfg, random_state=42)
	scores = cross_val_score(clf, iris.data, iris.target, cv=5)
	return 1-np.mean(scores)

scenario = Scenario({"run_obj":"quality",
					 "run_count-limit": 200,
					 "cs": cs,
					 "deterministic": "true"})


smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=svm_from_cfg)
incumbent = smac.optimize()
inc_value = svm_from_cfg(incumbent)
print(inc_value)



