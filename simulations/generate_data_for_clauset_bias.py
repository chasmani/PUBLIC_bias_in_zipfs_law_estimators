
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



from utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law
from utilities.bauke_clauset_mle import mle_bauke_diff
from utilities.general_utilities import append_to_csv

def generate_data_for_clauset_bias():

	data_filename = ("../data/simulated/clauset_bias_changing_lamb.csv")

	lambs = np.linspace(1.01,2,100)
	N = 10000

	seeds = range(100)

	for seed in seeds:
		print(seed)
		for lamb in lambs:
			np.random.seed(seed)
			n = get_ranked_empirical_counts_from_infinite_power_law(lamb, N)
			lamb_hat = mle_bauke_diff(n)
			csv_list = [seed, lamb, N, "clauset", lamb_hat]
			append_to_csv(csv_list, data_filename)

if __name__=="__main__":
	generate_data_for_clauset_bias()