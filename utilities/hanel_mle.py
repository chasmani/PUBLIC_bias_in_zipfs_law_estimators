
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from scipy import optimize
from scipy.optimize import bisect
import numpy as np

from utilities.probability_distributions import get_probabilities_power_law_finite_event_set


################################################
# VERSION 1A - MAXIMISING LIKELIHOOD WITH NORMALISATION CONSTANT


def get_z(lamb, W):
	"""
	Dumb version - jsut add it in normal space, no log space
	"""
	z = 0
	for j in range(1, W+1):
		z+= j**(-1*lamb)
	return z


def log_likelihood_with_z(lamb, n):
	"""
	log L = -lamb* SUM(n(i) * ln(i)) - N * ln(z)
	"""
	W = len(n)
	sum_part = 0
	for j in range(W):
		rank = j+1
		sum_part += n[j]*np.log(rank)

	z = get_z(lamb, W)
	log_l = -lamb * sum_part - sum(n) * np.log(z)
	return log_l


def hanel_mle_1a(n, W=None):

	# If we know W
	if W:
		# Add zero counts
		n_full = np.zeros(W)
		n_full[:len(n)] += n
		n = n_full

	# If we do not know W, assume it's the same as the length of n
	f = log_likelihood_with_z
	args = (n, )
	x0 = 1 # initial guess

	mle = optimize.fmin(lambda x, n: -f(x, n), x0, args=args)
	return mle[0]


################################################
# VERSION 1B - MAXIMISING LIKELIHOOD WITH GENERATED PROBABILITIES
def get_log_identity_prob(p,n,W):

	log_prob = 0
	for j in range(W):
		log_prob += n[j] * np.log(p[j])
	return log_prob

def get_log_likelihood_from_ps(lamb, n):

	W = len(n)
	p = get_probabilities_power_law_finite_event_set(lamb, W)
	return get_log_identity_prob(p, n, W)

def hanel_mle_1b(n, W=None):

	# If we know W
	if W:
		# Add zero counts
		n_full = np.zeros(W)
		n_full[:len(n)] += n
		n = n_full

	# If we do not know W, assume it's the same as the length of n
	f = get_log_likelihood_from_ps
	args = (n, )
	x0 = 1 # initial guess

	mle = optimize.fmin(lambda x, n: -f(x, n), x0, args=args)
	return mle[0]



#################################################
# VERSION 2A - ROOT OF DIFF LIKELIHOOD WITH NORMALISATION CONSTANT

def get_D_z(lamb, W):
	"""
	Dumb version - jsut add it in normal space, no log space
	D_Z = SUM(-j^lamb * ln(j))
	"""
	D_z = 0
	for j in range(1, W+1):
		D_z += -1*j**(-lamb)*np.log(j)
	return D_z


def D_likelihood(lamb, n):
	"""
	D_l = N Z'/Z + SUM(n(j)ln(j))
	"""


	W = len(n)
	sum_part = 0
	for j in range(W):
		rank = j+1
		sum_part += n[j]*np.log(rank)

	D_z = get_D_z(lamb, W)
	z = get_z(lamb, W)
	N = sum(n)

	D_l = N*D_z/z + sum_part
	return D_l


def hanel_mle_2a(n, W=None):

	a,b = 0.1, 10
	lamb_hat = bisect(D_likelihood, a, b, args=n, xtol=1e-6)
	return lamb_hat




#################################################
# TESTS

def test_all_equal_same_value():

	n = [80,50,20,10,5,1,1]

	print(hanel_mle_1b(n))
	print(hanel_mle_1a(n))
	print(hanel_mle_2a(n))


if __name__=="__main__":
	test_all_equal_same_value()
