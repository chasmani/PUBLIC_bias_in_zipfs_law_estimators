import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


import time

import numpy as np

from utilities.abc_pmc_zipf import abc_pmc_zipf
from utilities.general_utilities import append_to_csv
from utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law


def run_sims_changing_lambda_one_seed(seed=100, results_filename="../data/simulated/zipf_beaumont_results_cow_test.csv"):

	n_data = 10000
	n_particles = 256
	survival_fraction = 0.4
	n_generations = 10

	for i in range(1):	
		for exponent in np.linspace(1.01, 2, 100):
			print(exponent)
			
			np.random.seed(seed)
			print("Seed {} exponent {}".format(seed, exponent))

			ns = get_ranked_empirical_counts_from_infinite_power_law(exponent, N=n_data)
			
			try:
				start=time.time()
				mle = abc_pmc_zipf(ns, n_particles=n_particles, survival_fraction=survival_fraction, n_generations=n_generations)
				print(mle)
				end=time.time()
				csv_row = ["Beaumont 2007 Basic", seed, exponent, n_particles, survival_fraction, 
					n_data, n_generations, mle, end-start]
				print(csv_row)
			except Exception as e:
				print("EXCEPTION ", str(e))
				csv_row = ["Beaumont 2007 Basic", seed, exponent, n_particles, survival_fraction, 
					n_data, n_generations, str(e)]
			append_to_csv(csv_row, results_filename)


def run_sims_changing_N_one_seed(seed=100, results_filename="../data/simulated/zipf_beaumont_results_cow_test.csv"):

	# Experiment variables - match ones chosen for Clauset
	N_exponents = range(6,21)
	Ns = [2**a for a in N_exponents]
	exponent = 1.1

	# WABC variables
	n_particles = 256
	survival_fraction = 0.4
	n_generations = 10

	print(Ns)

	for i in range(1):	
		for N in Ns:
			n_data = N
			print(N)
			
			np.random.seed(seed)
			print("Seed {} N {}".format(seed, N))

			ns = get_ranked_empirical_counts_from_infinite_power_law(exponent, N=N)
			
			try:
				start=time.time()
				mle = abc_pmc_zipf(ns, n_particles=n_particles, survival_fraction=survival_fraction, n_generations=n_generations)
				print(mle)
				end=time.time()
				csv_row = ["Beaumont 2007 Basic", seed, exponent, n_particles, survival_fraction, 
					n_data, n_generations, mle, end-start]
				print(csv_row)
			except Exception as e:
				print("EXCEPTION ", str(e))
				csv_row = ["Beaumont 2007 Basic", seed, exponent, n_particles, survival_fraction, 
					n_data, n_generations, str(e)]
			append_to_csv(csv_row, results_filename)

def simulate_lots():

	for seed in range(1):
		#run_sims_changing_lambda_one_seed(seed)
		run_sims_changing_N_one_seed(seed)


if __name__=="__main__":
	simulate_lots()