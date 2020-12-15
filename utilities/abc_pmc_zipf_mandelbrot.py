import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math

import scipy.stats
import numpy as np
import pandas as pd

from utilities.data_generators import get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law


def abc_pmc_zipf_mandelbrot(n, exponent_min=1.01, exponent_max=3, q_min=0, q_max=10, survival_fraction=0.4, n_particles=256, n_generations=10):
	"""
	As described in Pilgrim and Hills, "Bias in Zipf's Law Estimators" (2021)
	"""	
	prior_dist = scipy.stats.uniform(loc=exponent_min, scale=exponent_max-exponent_min)
	n_data = sum(n)
	tolerance = math.inf
	for g in range(n_generations):
		print("Running ABC PMC generation {} of {}".format(g, n_generations))
		thetas = []
		ds = []
		weights = []
		for i in range(n_particles):
			hit = False
			while not hit:
				if g == 0:
					# First generation - uniform dist is a different type of object in scipy to kde
					exponent = prior_dist.rvs(1)[0]
					q = np.random.choice(q_max)
					theta = [exponent, q]
				else:
					sample = proposal_dist.resample(1)
					lamb = sample[0][0]
					q = int(round(sample[1][0]))
					theta = [lamb, q]

				if theta[0] > exponent_min and theta[0] < exponent_max and theta[1] > q_min:
					z = get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law(theta[0], theta[1], N=n_data)
					d = scipy.stats.wasserstein_distance(z, n)
					if d <= tolerance:
						thetas.append(theta)
						ds.append(d)
						if g == 0:
							# First generation - uniform proposal so equal weights
							weight = 1
						else:
							# Uniform dist has equal values at each vaue of theta
							# Relative values of weights is needed - not absolute values
							weight = 1/proposal_dist.evaluate(theta)[0]
						weights.append(weight)
						hit=True

		tolerance = get_tolerance(ds, survival_fraction)
		# bandwidth with 2 * data weighted variance works well
		proposal_dist = scipy.stats.gaussian_kde(np.transpose(thetas), weights=np.array(weights), bw_method=np.sqrt(2))

		# Export results to a pickle to analyse later
		pickle_filename = "../data/simulated/zipf_mandelbrot_N_{}_generation_{}.pkl".format(n_data, g)
		export_results_to_pickle(thetas, weights, pickle_filename)

	posterior = scipy.stats.gaussian_kde(np.transpose(thetas), weights=np.array(weights), bw_method=1)

	exponents_num = 10000
	qs_num = 10
	exponents = np.linspace(1,3, num=exponents_num)
	qs = np.arange(qs_num)

	results = np.zeros((exponents_num,qs_num))
	for i in range(exponents_num):
		for j in range(qs_num):
			p = posterior.evaluate([exponents[i], qs[j]])
			results[i,j] = p

	i, j = np.unravel_index(np.nanargmax(results), results.shape)
	mle = exponents[i],qs[j]
	return mle


def export_results_to_pickle(thetas, ws, pickle_name="data/zipf_mandelbrot_one_run.pkl"):

	df = pd.DataFrame(columns=['exponent','q', 'w'])
	for i in range(len(thetas)):
		df = df.append({'exponent':thetas[i][0], 'q':thetas[i][1], 'w':ws[i]}, ignore_index=True)
	df.to_pickle(pickle_name)


def get_tolerance(distances, survival_fraction):

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	return new_tolerance


def test_with_basic_data():

	actual_lamb = 1.2
	actual_q = 4
	N = 100000
	n = get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law(actual_lamb, q=actual_q, N=N)
	estimated_theta = abc_pmc_zipf_mandelbrot(n, n_generations=20)
	print("Actual parameters were {}, q={}, predicted lamb was {}".format(actual_lamb, actual_q, estimated_theta))


if __name__=="__main__":
	test_with_basic_data()