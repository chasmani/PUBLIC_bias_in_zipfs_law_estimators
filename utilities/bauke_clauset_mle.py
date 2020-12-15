import numpy as np
from scipy.special import zeta
from scipy.optimize import bisect 
from scipy import optimize




################################################
# VERSION 1 - MAXIMISING LIKELIHOOD


def likelihood_bauke(lamb, n):
	"""
	log L = -lamb* SUM(n(i) * ln(i)) - N * ln(zeta(lamb))
	"""

	sum_part = 0
	for j in range(len(n)):
		rank = j+1
		sum_part += n[j]*np.log(rank)

	z = zeta(lamb)
	log_l = -lamb * sum_part - sum(n) * np.log(z)
	return log_l

def mle_bauke_l(n):

	# If we do not know W, assume it's the same as the length of n
	f = likelihood_bauke
	args = (n, )
	x0 = 1 # initial guess

	mle = optimize.fmin(lambda x, n: -f(x, n), x0, args=args)
	return mle[0]


#################################################
# VERSION 2 - ROOT OF DIFF LIKELIHOOD

def D_l(alpha, t):
	return log_deriv_zeta(alpha) + t	

def get_t(n,x_min=1):
	sum_part = 0
	for j in range(len(n)):
		rank = j+1
		sum_part += n[j]*np.log(rank)
	return sum_part/sum(n)

def log_zeta(alpha):
    return np.log(zeta(alpha, 1))

def log_deriv_zeta(alpha):
    h = 1e-5
    return (log_zeta(alpha+h) - log_zeta(alpha-h))/(2*h)

def mle_bauke_diff(ns):
	t = get_t(ns)
	a,b = 1.01, 10
	alpha_hat = bisect(D_l, a, b, args=t, xtol=1e-6)
	return alpha_hat

################################################
# VERSION 3 - python's powerlaw package


def mle_powerlaw_package(x):

	lib_fit = powerlaw.Fit(x, discrete=True, xmin=1, estimate_discrete=False)
	return lib_fit.power_law.alpha


#################################################
# TESTS

def test_all_the_same_result():

	n = [80,50,20,1,1,0]

	# Check 2 versions, plus version from python packages, at least one
	print(mle_bauke_l(n))

	# Convert to x vector to use powerlar library
	x = []
	for i in range(len(n)):
		rank = i+1
		x += [rank]*n[i]

	print(mle_powerlaw_package(x))


	print(mle_bauke_diff(n))

if __name__=="__main__":
	test_all_the_same_result()