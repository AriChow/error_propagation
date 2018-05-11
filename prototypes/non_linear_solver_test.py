import numpy as np
from scipy.optimize import fsolve

def func_beta(p):
	alpha1, alpha2, alpha3, gamma1, gamma2, gamma3, beta1, beta2, beta3 = p
	errs = np.random.random(9)
	f1 = alpha1 + gamma1 * (alpha1 * beta1) - errs[0]
	f2 = alpha2 + gamma1 * (alpha2 + beta1) - errs[1]
	f3 = alpha1 + gamma2 * (alpha1 + beta2) - errs[2]
	f4 = alpha2 + gamma2 * (alpha2 + beta2) - errs[3]
	f5 = alpha3 + gamma2 * (alpha3 + beta2) - errs[4]
	f6 = alpha3 + gamma1 * (alpha3 + beta1) - errs[5]
	f7 = alpha3 + gamma3 * (alpha3 + beta3) - errs[6]
	f8 = alpha2 + gamma3 * (alpha2 + beta3) - errs[7]
	f9 = alpha1 + gamma3 * (alpha1 + beta3) - errs[8]
	return (f1, f2, f3, f4, f5, f6, f7, f8, f9)

p = np.random.rand(9)
parameters = np.zeros((1, 9))
error = np.zeros((1, 9))
i = 0
params = fsolve(func_beta, p)
err = func_beta(params)
error[i, :] = np.expand_dims(np.asarray(err), 0)
parameters[i, :] = np.expand_dims(np.asarray(params), 0)

print('Parameters:')
print(parameters)

print('Errors:')
print(error)


# import numpy as np
# 
# def null(A, eps=1e-15):
#     u, s, vh = np.linalg.svd(A)
#     null_space = np.compress(s <= eps, vh, axis=0)
#     return null_space.T
# 
# e = -1 + 2 * np.random.rand(3, 3)
# alpha = null(e).T
# 
# print(alpha)
# print(e * alpha).T


