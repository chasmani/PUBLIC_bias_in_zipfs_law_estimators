import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math

import scipy.stats
import numpy as np

from utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law


def abc_pmc_zipf(n, theta_min=1.01, theta_max=3, survival_fraction=0.4, n_particles=256, n_generations=10):
	"""
	As described in Pilgrim and Hills, "Bias in Zipf's Law Estimators" (2021)
	"""	
	prior_dist = scipy.stats.uniform(loc=theta_min, scale=theta_max-theta_min)
	n_data = sum(n)
	tolerance = math.inf
	proposal_dist = prior_dist
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
					theta = proposal_dist.rvs(1)[0]
				else:
					theta = proposal_dist.resample(1)[0][0]

				if theta > theta_min and theta < theta_max:
					z = get_ranked_empirical_counts_from_infinite_power_law(theta, N=n_data)
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
		var = 2*np.cov(thetas, aweights=weights)
		# bandwidth with 2 * data weighted variance works well
		proposal_dist = scipy.stats.gaussian_kde(thetas, weights=weights, bw_method=np.sqrt(2))

	posterior = scipy.stats.gaussian_kde(thetas, weights=weights)
	xs = np.linspace(theta_min,theta_max, 100000)
	posterior_values = posterior.evaluate(xs)
		
	mle = xs[np.argmax(posterior_values)]
	return mle
	

def get_tolerance(distances, survival_fraction):

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	return new_tolerance


def test_with_basic_data():

	actual_lamb = 1.2
	n = get_ranked_empirical_counts_from_infinite_power_law(1.2, N=10000)
	estimated_lamb = abc_pmc_zipf(n)
	print("Actual lamb was {}, predicted lamb was {}".format(actual_lamb, estimated_lamb))


if __name__=="__main__":
	test_with_basic_data()