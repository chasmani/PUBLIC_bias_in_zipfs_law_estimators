import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from utilities.data_generators import get_ranked_empirical_counts_from_finite_power_law
from utilities.full_likelihood import get_likelihood, get_full_mle, get_full_d_likelihood
from utilities.hanel_mle import hanel_mle_1a, log_likelihood_with_z, D_likelihood
from utilities.general_utilities import append_to_csv

from design_scheme import PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, POINT_SIZE, LINEWIDTH


def plot_full_likelihood_vs_hanel():

	csv_filename = "data/likelihood_values_b.csv"

	seed = 12
	np.random.seed(seed)

	lamb = 0.4

	W = 6
	N = 100

	n = get_ranked_empirical_counts_from_finite_power_law(lamb, W, N)

	n = [10,3,3,2,1,1]

	print(n)

	lambs = np.linspace(0.1, 2.5, 50)


	plt.subplot(211)

	full_likelihoods = [] 
	hanel_likelihoods = []
	for lamb in lambs:
		likelihood = get_likelihood(lamb, n)
		full_likelihoods.append(likelihood)

		hanel_likelihood = np.exp(log_likelihood_with_z(lamb, n))
		hanel_likelihoods.append(hanel_likelihood)
	
	plt.plot(lambs, full_likelihoods, label="Full Likelihood", color=PRIMARY_COLOR, linewidth=LINEWIDTH)
	plt.plot(lambs, hanel_likelihoods, label="Hanel et al", color=SECONDARY_COLOR, linewidth=LINEWIDTH)

	full_mle = get_full_mle(n)
	plt.axvline(full_mle, color=PRIMARY_COLOR, linewidth=LINEWIDTH, linestyle="dashed")

	hanel_mle = hanel_mle_1a(n)
	plt.axvline(hanel_mle, color=SECONDARY_COLOR, linewidth=LINEWIDTH, linestyle="dashed")

	plt.yscale("log")

	plt.ylabel("$\mathcal{L}(\lambda|n)$")

	plt.subplot(212)


	full_d_likelihoods = []
	full_d_hanels = []

	for lamb in lambs:
		full_d_likelihood = get_full_d_likelihood(lamb, n)
		full_d_likelihoods.append(full_d_likelihood*10**12.5)



		hanel_d_likelihood = D_likelihood(lamb, n)
		full_d_hanels.append(hanel_d_likelihood)

	print("Full d likelihoods: ", full_d_likelihoods)


	plt.plot(lambs, full_d_likelihoods, label="Full Likelihood", color=PRIMARY_COLOR, linewidth=LINEWIDTH)
	plt.plot(lambs, full_d_hanels, label="Hanel et al", color=SECONDARY_COLOR, linewidth=LINEWIDTH)

	plt.axhline(0, linewidth=LINEWIDTH, color="lightgray")

	plt.axvline(full_mle, color=PRIMARY_COLOR, linewidth=LINEWIDTH, linestyle="dashed")

	plt.axvline(hanel_mle, color=SECONDARY_COLOR, linewidth=LINEWIDTH, linestyle="dashed")

	print(full_mle)

	plt.xlabel("$\lambda$")

	plt.ylabel("$D\mathcal{L}(\lambda|n)$")
	plt.legend()


	plt.savefig("images/likelihood_function_and_differential_full_vs_hanel_b.png", dpi=300)

	plt.show()





"""
def plot_likelihood_function_at_different_depths():

	csv_filename = "data/likelihood_values_b.csv"

	df = pd.read_csv(csv_filename, delimiter=";", names=["seed", "n", "depth", "lamb", "likelihood"])
	
	HANEL_COLOR = "orange"
	FULL_L_COLOR = "blue"
	linewidth=2

	hanel_df = df.loc[df['depth']==0]
	sns.lineplot(x="lamb", y="likelihood", data=hanel_df, label="Hanel et al", color=HANEL_COLOR, linewidth=linewidth)

	full_df = df.loc[df['depth']==20]
	sns.lineplot(x="lamb", y="likelihood", data=full_df, label="Full likelihood", color=FULL_L_COLOR, linewidth=linewidth)


	plt.xlabel("exponent, $\lambda$")
	plt.ylabel("Likelihood of data")

	plt.yscale("log")

	n = [10,3,3,2,1,1]
	hanel_lamb_hat = get_mle(n, 0)
	print("Hanel estimator is ", hanel_lamb_hat)

	plt.axvline(hanel_lamb_hat, linestyle="dashed", color=HANEL_COLOR, linewidth=linewidth)

	full_lamb_hat = get_mle(n, 10)
	plt.axvline(full_lamb_hat, linestyle="dashed", color=FULL_L_COLOR, linewidth=linewidth)	
	print("Full estimator is ", full_lamb_hat)


	plt.legend()

	plt.savefig("images/likelihood_function_hanel_and_full.png")

	plt.show()

"""








if __name__=="__main__":
	plot_full_likelihood_vs_hanel()


