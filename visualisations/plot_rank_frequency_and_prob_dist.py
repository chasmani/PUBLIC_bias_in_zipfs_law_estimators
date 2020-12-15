
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from utilities.probability_distributions import get_probabilities_power_law_finite_event_set


def get_raw_data_with_ranks(exponent, W, N):

	probs = get_probabilities_power_law_finite_event_set(exponent, W)
	ds = np.random.choice(W, size=N, p=probs)
	return ds


def convert_ds_to_hist(ds, W, norm=True):

	ds = np.array(ds)

	hists = []
	for i in range(W):
		m = (ds == i).sum()
		hists.append(m)
	hists = np.array(hists)
	if norm==True:
		hists = hists/sum(hists)
	return hists	


def convert_ds_to_rank_freq_ns(ds, W, norm=True):

	counts = Counter(ds)
	ns = [v for k,v in counts.most_common()]
	
	# Add zero counts
	n_full = np.zeros(W)
	n_full[:len(ns)] += ns

	ns = np.array(n_full)
	if norm == True:
		ns = ns/sum(ns)

	return ns


def plot_rank_frequency_and_prob_dist_and_ranked_data():

	np.random.seed(3)

	lamb = 1
	W = 60
	N = 200


	ds = get_raw_data_with_ranks(lamb, W, N)

	ms = convert_ds_to_hist(ds, W, norm=True)
	
	ns = convert_ds_to_rank_freq_ns(ds, W, norm=True)

	# Plot prob distribution
	ps = get_probabilities_power_law_finite_event_set(lamb, W)
	plt.plot(range(1, W+1), ps, linestyle="dashed", linewidth=3, label="Probability Distribution")


	# Plot data with known ranks
	plt.scatter(range(1, W+1), ms, s=50, alpha=0.5, label="A priori ranks")

	# Plot rank-frequency data
	plt.scatter(range(1,W+1), ns, s=50, marker="x", color="red", label="Empirical ranks")

	plt.xscale("log")
	plt.yscale("log")

	plt.xlabel("rank")
	plt.ylabel("probability/frequency")

	plt.legend()

	plt.savefig("images/rank_frequency_vs_prob_dist.png")
	plt.show()

if __name__=="__main__":
	plot_rank_frequency_and_prob_dist_and_ranked_data()