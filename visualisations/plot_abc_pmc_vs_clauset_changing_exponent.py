import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,4)

from design_scheme import PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, POINT_SIZE, LINEWIDTH

def plot_wabc_vs_clauset():
	"""
	Combine results from csv files and plot them
	"""

	# Concatenate all dataframes together - clauset
	frames = []
	for seed_start in ["_100", "_120", "_140","_160", "_180", "_200", "_220", "_20", "_40", "_60", "_80"]:
		data_filename = ("../data/simulated/abc_linear_regression_mean_log_observations_and_clauset_changing_lambda{}.csv".format(seed_start))
		df = pd.read_csv(data_filename, delimiter=";", names=["seed", "actual_lambda", "N", "Prop", "method", "mle"])		
		frames.append(df)
	df = pd.concat(frames)
	N = 10000
	# Check we are only getting data we want
	df = df[(df['N'] == N)]
	df = df[(df['actual_lambda'] < 2)]
	# Remove duplicated seeds
	df = df.drop_duplicates(['seed','actual_lambda', 'method'])
	# Get only the 100 runs of each lambda and method
	df = df.groupby(['actual_lambda', 'method']).head(100).reset_index()
	# clauset only
	clauset_df = df[(df['method'] == "clauset")]

	#############
	# WABC
	input_filename = "../data/simulated/zipf_beaumont_results_cow_b.csv"
	names = ["method", "seed", "actual_lambda", "n_particles", "survival_fraction", 
					"n_data", "generations", "mle", "time"]

	df = pd.read_csv(input_filename, names=names, sep=";")

	df = df[(df['n_data'] == N)]
	df = df[(df['actual_lambda'] < 2)]

	df = df.drop_duplicates(['seed','actual_lambda', 'method'])

	counts = df.groupby(['actual_lambda', 'method']).count()


	df = df.groupby(['actual_lambda', 'method']).head(100).reset_index()
	


	df = df[(~df["time"].isnull())]

	df = df[(df["generations"] == 10)]


	df["mle"] = df["mle"].astype(float)

	df = df[(df["method"] == "Beaumont 2007 Basic")]
	df["bias"] = df["mle"] - df["actual_lambda"]
	wabc_df = df

	counts = df.groupby(["actual_lambda"]).count()
	print(counts)





	# Plots
	plt.subplot(1,3,1)
	plt.plot(df["actual_lambda"], df["actual_lambda"], label="y=x", linewidth=LINEWIDTH*2, linestyle="dashed", color=TERTIARY_COLOR, alpha=0.6)

	sns.lineplot(x="actual_lambda", y="mle", data=wabc_df, ci=68, label="ABC-PMC", color=PRIMARY_COLOR, linewidth=LINEWIDTH)

	sns.lineplot(x="actual_lambda", y="mle", data=clauset_df, ci=68, label="Clauset et al", color=SECONDARY_COLOR, linewidth=LINEWIDTH, linestyle="dashdot")

	plt.xlabel("$\lambda$")
	plt.ylabel("$E(\hat{\lambda})$")



	plt.subplot(1,3,2)

	sns.lineplot(x="actual_lambda", y="bias", data=wabc_df, ci=68, color=PRIMARY_COLOR, linewidth=LINEWIDTH)
	clauset_df["bias"] = clauset_df["mle"] - clauset_df["actual_lambda"]
	sns.lineplot(x="actual_lambda", y="bias", data=clauset_df, ci=68, color=SECONDARY_COLOR, linewidth=LINEWIDTH, linestyle="dashdot")

	plt.xlabel("$\lambda$")
	plt.ylabel("Bias")


	plt.subplot(1,3,3)

	wabc_grouped = wabc_df.groupby('actual_lambda').agg(variance=("mle", "var")).reset_index()
	sns.lineplot(x="actual_lambda", y="variance", data=wabc_grouped, color=PRIMARY_COLOR, linewidth=LINEWIDTH)

	clauset_grouped = clauset_df.groupby('actual_lambda').agg(variance=("mle", "var")).reset_index()
	sns.lineplot(x="actual_lambda", y="variance", data=clauset_grouped, color=SECONDARY_COLOR, linewidth=LINEWIDTH, linestyle="dashdot")

	plt.xlabel("$\lambda$")
	plt.ylabel("Variance")

#	plt.suptitle("Clauset vs WABC PMC Mean MLEs for Unbounded Power Law\nN={}".format(N))

	plt.tight_layout()
	

	plt.savefig("images/wabc_linear_vs_clauset_exponents.png", dpi=300)

	plt.show()

if __name__=="__main__":
	plot_wabc_vs_clauset()