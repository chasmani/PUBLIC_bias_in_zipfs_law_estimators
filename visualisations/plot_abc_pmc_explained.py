
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law


def get_new_tolerance(distances, survival_fraction):

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	return new_tolerance


def plot_wasserstein_vs_lamb_simple():

	np.random.seed(1)

	N = 1000
	actual_lamb = 1.2
	n_lambs = 2000
	lambs = np.random.uniform(1.15, 1.25, n_lambs)

	ns_obs = get_ranked_empirical_counts_from_infinite_power_law(actual_lamb, N)

	wass_ds = []
	for lamb in lambs:
		ns = get_ranked_empirical_counts_from_infinite_power_law(lamb, N)
		w_d = scipy.stats.wasserstein_distance(ns_obs, ns)
		wass_ds.append(w_d)

	tolerance = get_new_tolerance(wass_ds, 0.5)

	top_lambs = []
	top_ws = []

	for i in range(n_lambs):
		if wass_ds[i] < tolerance:
			top_lambs.append(lambs[i])
			top_ws.append(wass_ds[i])

	sns.scatterplot(top_lambs, top_ws, color="blue")
	plt.axhline(tolerance, color="blue", linestyle="--", label=r"$\epsilon$")
	

	plt.ylim(0.06, tolerance*1.3)

	plt.xlabel("$\lambda$")
	plt.ylabel("$W(n_i, n_{obs})$")

	plt.legend()

	plt.savefig("images/abc-pmc-wasserstein_distance_vs_lamb_N_{}-simple.png".format(N))

	plt.show()


def plot_kde_of_result_simple():

	np.random.seed(1)

	N = 1000
	actual_lamb = 1.2
	n_lambs = 2000
	lambs = np.linspace(1.15, 1.25, n_lambs)

	ns_obs = get_ranked_empirical_counts_from_infinite_power_law(actual_lamb, N)

	wass_ds = []
	for lamb in lambs:
		ns = get_ranked_empirical_counts_from_infinite_power_law(lamb, N)
		w_d = scipy.stats.wasserstein_distance(ns_obs, ns)
		wass_ds.append(w_d)

	top_tolerance = get_new_tolerance(wass_ds, 0.5)

	top_lambs = []
	top_ws = []

	for i in range(n_lambs):
		if wass_ds[i] < top_tolerance:
			top_lambs.append(lambs[i])
			top_ws.append(wass_ds[i])

	var = np.var(top_lambs)

	sns.kdeplot(top_lambs, bw_method=np.sqrt(2))



	plt.xlabel("$\lambda$")
	plt.ylabel("proposal distribution, $g(\lambda)$")

	plt.savefig("images/abc-pmc-proposal-distribution-simple.png")

	plt.show()



def plot_data_and_sims():

	plt.rcParams["figure.figsize"] = (20,4)

	np.random.seed(5)

	actual_lamb = 1.2
	N = 1000

	ns_observed = get_ranked_empirical_counts_from_infinite_power_law(actual_lamb, N)
	ns_ranks = range(1, len(ns_observed)+1)


	fig, (ax1, axgap, ax2, ax3, ax4) = plt.subplots(1, 5, sharey=True, sharex=True)


	ax1.scatter(ns_ranks, ns_observed)

	plt.xscale("log")
	plt.yscale("log")

	ax1.set_xlabel("$n_{obs}$")
	
	ax1.set_title("Observed data, $\lambda={}$".format(actual_lamb))

	axgap.set_title('$\lambda_i \sim g^{t-1}(\lambda)$')

	lamb_1 = 1.15
	ns_2 = get_ranked_empirical_counts_from_infinite_power_law(lamb_1, N)
	ns_2_ranks = range(1, len(ns_2)+1)

	ax2.scatter(ns_2_ranks, ns_2)

	ax2.set_title("$\lambda_1 = {}$".format(lamb_1))

	ax2.set_xlabel("$n_1$")



	lamb_2 = 1.21
	ns_3 = get_ranked_empirical_counts_from_infinite_power_law(lamb_2, N)
	ns_3_ranks = range(1, len(ns_3)+1)

	ax3.scatter(ns_3_ranks, ns_3)

	ax3.set_title("$\lambda_2 = {}$".format(lamb_2))
	ax3.set_xlabel("$n_2$")


	lamb_3 = 1.23
	ns_4 = get_ranked_empirical_counts_from_infinite_power_law(lamb_3, N)
	ns_4_ranks = range(1, len(ns_4)+1)

	ax4.scatter(ns_4_ranks, ns_4)

	ax4.set_title("$\lambda_3 = {}$".format(lamb_3))
	ax4.set_xlabel("$n_3$")

	for ax in [ax1, ax2, ax3, ax4]:

		ax.tick_params(
	    	axis='x',          # changes apply to the x-axis
	    	which='both',      # both major and minor ticks are affected
	    	bottom=False,      # ticks along the bottom edge are off
	    	top=False,         # ticks along the top edge are off
	    	labelbottom=False)
		ax.tick_params(
	    	axis='y',          # changes apply to the x-axis
	    	which='both',      # both major and minor ticks are affected
	    	left=False,      # ticks along the bottom edge are off
	    	right=False,         # ticks along the top edge are off
	    	labelleft=False)


	
	plt.savefig("images/abc-pmc-top-part-data-and-sims.png")
	plt.show()


def plot_all_for_figure():


	plot_wasserstein_vs_lamb_simple()
	plot_kde_of_result_simple()
	plot_data_and_sims()


if __name__=="__main__":
	plot_all_for_figure()