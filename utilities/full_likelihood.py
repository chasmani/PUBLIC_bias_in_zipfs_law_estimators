

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from scipy import optimize

from utilities.probability_distributions import get_probabilities_power_law_finite_event_set


def get_symmetric_group(W):
	"""
	Return all permutations of the numbers 1 to W
	"""
	perms = [[]]

	# Iterate W times, add one lement to the permutations at each iteration
	for j in range(W):
		
		new_perms = []
		for perm in perms:
			# Branch out through permutation tree 
			for i in range(W):
				# Only traverse valid branches
				if i not in perm:
					new_perm = perm.copy()
					new_perm.append(i)
				
					new_perms.append(new_perm)
		perms = new_perms
	return perms


def get_unique_permutations(n):
	"""
	Get all the unique permutations, taking into account tied empirical events
	Permutations are unique in F(W, n) - unique sorting of events into counts
	Permutations are in a dictionary format
	The permutations only list deviations from the identity permutation
		{ j: n(s(j)) },
		{ event_rank: n(permutation(event_rank)) },
		{ event_rank: number of observations of this event given the permutation },
	Each permutation in this format is a unique element in F(W,n) by definition
	"""
	W = len(n)
	symmetric_group = get_symmetric_group(W)

	unique_perms = [{}]

	for s in symmetric_group:
		new_perm = {s_i: n[index] for index, s_i in enumerate(s) if n[s_i]!=n[index]}
		if new_perm not in unique_perms:
			unique_perms.append(new_perm)
	return unique_perms


def get_likelihood(lamb, n):
	"""
	Get the likelihood of the data given labmda, considering only unique permutations of probabilty ranks into empirical counts
	This likelihood is proportional to the likelihood of all permutations, and saves orders of magnitude of comutation time
	"""
	permutations = get_unique_permutations(n)
	W = len(n)
	p = get_probabilities_power_law_finite_event_set(lamb, W)
	identity_prob = get_identity_prob(p,n,W)
	relative_probs = []
	for perm in permutations:
		relative_probs.append(get_relative_prob(lamb, p, n, perm))
	likelihood = sum(relative_probs) * identity_prob
	return likelihood


def get_relative_prob(lamb, p, n, perm):
	"""
	Get the probability of a pemrutation relative to the identity permutation
	"""
	relative_prob = 1
	for j in perm.keys():
		relative_prob *= p[j]**(perm[j]-n[j])
	return relative_prob


def get_identity_prob(p,n,W):
	"""
	Probabilty of the identity permutation
	"""
	prob = 1
	for j in range(W):
		prob *= p[j]**n[j]
	return prob

def get_full_mle(n):
	"""
	Maximise the likelihood function
	"""

	f = get_likelihood
	args = (n,)
	x0 = 1 # initial guess

	mle = optimize.fmin(lambda x, n: -f(x, n), x0, args=args)
	return mle[0]


def get_z(lamb, W):
	"""
	Dumb version - jsut add it in normal space, no log space
	"""
	z = 0
	for j in range(1, W+1):
		z+= j**(-1*lamb)
	return z


def get_d_z(lamb, W):
	"""
	Dumb version - jsut add it in normal space, no log space
	D_Z = SUM(-j^lamb * ln(j))
	"""
	D_z = 0
	for j in range(1, W+1):
		D_z += -1*j**(-lamb)*np.log(j)
	return D_z


def get_permutation_mean_log(n, perm, W):

	mean_log = 0

	for j in range(W):
		rank = j+1
		if j in perm.keys():
			mean_log += perm[j] * np.log(rank)
		else:
			mean_log += n[j] * np.log(rank)
	return mean_log

def get_permutation_likelihood(n, perm, W, lamb, Z):

	likelihood = 1

	for j in range(W):
		rank = j+1
		if j in perm.keys():
			likelihood *= (rank**(-1*lamb)/Z)**perm[j] 
		else:
			likelihood *= (rank**(-1*lamb)/Z)**n[j]
	return likelihood



def get_full_d_likelihood(lamb, n):

	permutations = get_unique_permutations(n)
	W = len(n)
	
	N = sum(n)
	Z = get_z(lamb, W)
	d_Z = get_d_z(lamb, W)

	total_sum = 0
	for perm in permutations:
		perm_mean_log = get_permutation_mean_log(n,perm, W)
		total_sum += (N*d_Z/Z + perm_mean_log) * get_permutation_likelihood(n, perm, W, lamb, Z)
	return total_sum



if __name__=="__main__":

	n = [4,1]
	lamb = 1
	W = len(n)

	print(get_z(lamb, W))

	print(get_d_z(lamb, W))


	print(get_full_d_likelihood(2.0, [4,1]))
	print(get_full_d_likelihood(1.8, [4,1]))
	print(get_full_mle(n))