

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_clauset_bias():

	data_filename = ("../data/simulated/clauset_bias_changing_lamb.csv")

	N = 100000

	df = pd.read_csv(data_filename, delimiter=";", names=["seed", "actual_lambda", "N", "method", "mle"])

	# Check we are only getting data we want
	df = df[(df['N'] == N) & (df['method'] == "clauset")]

	# Remove duplicated seeds
	df = df.drop_duplicates(['seed','actual_lambda'])

	print(df)

	# Get mean across seeds
	df = df.groupby(["actual_lambda"]).mean().reset_index()

	print(df)

	fig, ax = plt.subplots()

	plt.plot(df["actual_lambda"], df["actual_lambda"], label="y=x", linewidth=3, linestyle="dashed", color="r")
	plt.plot(df["actual_lambda"], df["mle"], label="MLE", linewidth=3, color="b")


	plt.xlabel("$\lambda$")
	plt.ylabel("$\hat{\lambda}$")

	plt.title("Bias in MLE for Zeta Distributed Rank-Frequency Data\nN={}".format(N))

	plt.legend()

	
	df["bias"] = df["mle"] - df["actual_lambda"]


	axins = inset_axes(ax,  "30%", "40%" ,loc="lower right", borderpad=3)


	axins.plot(df["actual_lambda"], df["bias"], linewidth=3, color="b")
	#sns.lineplot(df["actual_lambda"], df["bias"], ax=axins, size=3)
	
	axins.set_xlabel('')
	axins.set_ylabel("Bias")


	plt.savefig("images/bias_in_mle_across_lambda_N_{}.png".format(N))

	plt.show()




if __name__=="__main__":
	#generate_data_for_clauset_bias()
	plot_clauset_bias()