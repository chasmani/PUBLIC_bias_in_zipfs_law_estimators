
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from utilities.probability_distributions import get_probabilities_power_law_finite_event_set


def convert_observations_into_ranked_empirical_counts(samples):

	counts = Counter(samples)
	n = [v for k,v in counts.most_common()]
	return n

#########################
# Fininte event set

def generate_samples_from_finite_power_law(exponent, W, N):
	"""
	generate random samples
	"""
	probs = get_probabilities_power_law_finite_event_set(exponent, W)
	return np.random.choice(W, size=N, p=probs)


def get_ranked_empirical_counts_from_finite_power_law(exponent, W, N):

	samples = generate_samples_from_finite_power_law(exponent, W, N)
	n = convert_observations_into_ranked_empirical_counts(samples)
	return n

######################
# Infinite event set

def generate_samples_from_infinite_power_law(exponent, N):

	xs_np = np.random.zipf(a=exponent, size = N)
	return xs_np


def get_ranked_empirical_counts_from_infinite_power_law(exponent, N):

	xs = generate_samples_from_infinite_power_law(exponent, N)
	n = convert_observations_into_ranked_empirical_counts(xs)
	return n

##################
# Generate Data from Zipf-Mandelbrot law

def generate_samples_from_zipf_mandelbrot_law(exponent=1.2, q=10, N=100):
	"""
	Easiest way to do this is simply shift the samples from a zipf law by q
	Thne throw away any samples below 1. 
	"""

	loc = -q

	# This will generate the correct dsitribution
	# But includes some negative numbers, which we do not want 
	x = scipy.stats.zipf.rvs(exponent, loc=loc, size=N)
	# get only positive numbers
	
	x = x[x>0]



	# We need to genreate more numbers, to get to N. 
	# Generate a few more than we expect to need, in case we don't get enough
	N_extra_needed_plus_few_more = ((N/(len(x)+1) - 1) * N)*1.2
	while (len(x) < N):
		x_extra = scipy.stats.zipf.rvs(exponent, loc=loc, size=int(N_extra_needed_plus_few_more))
		x_extra = x_extra[x_extra>0]
		x = np.concatenate((x, x_extra))

	x = x[:N]
	return x

def get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law(exponent, q, N):

	xs = generate_samples_from_zipf_mandelbrot_law(exponent, q, N)
	n = convert_observations_into_ranked_empirical_counts(xs)
	return n


if __name__=="__main__":
	d = generate_samples_from_infinite_power_law(1.04, 200000)
	print(d)