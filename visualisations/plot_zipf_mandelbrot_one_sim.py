
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def plot_from_pickle(pickle_name="../data/simulated/overnight_results_gen_10.pkl"):

	df = pd.read_pickle(pickle_name)
	sns.kdeplot(data=df, x="exponent", y="q", weights="w", shade=True)


	exponents = df['exponent'].to_numpy()
	qs = df['q'].to_numpy()
	thetas = np.array([exponents, qs])

	ws = df['w'].to_numpy()

	kde = scipy.stats.gaussian_kde(thetas, weights=ws)

	# Find mle
	exponents_num = 1000
	qs_num = 10
	exponents = np.linspace(-2,2, num=exponents_num)
	qs = np.arange(qs_num)

	results = np.zeros((exponents_num,qs_num))

	results_df = pd.DataFrame(columns=['exponent','q', 'prob'])


	for i in range(exponents_num):
		for j in range(qs_num):
			p = kde.evaluate([exponents[i], qs[j]])
			results[i,j] = p

	i, j = np.unravel_index(np.nanargmax(results), results.shape)
	print("Mle is ", exponents[i],qs[j])

	plt.scatter(1.2,4, color="orange", marker="x", s=100, linewidth=3, label="actual")

	plt.scatter(exponents[i],qs[j], color="yellow", s=100, linewidth=3, label="mle")

	plt.tight_layout()
	
	plt.legend()
	plt.savefig("images/zipf_mandelbrot_posterior_from_wabc.png")
	
	plt.show()

plot_from_pickle()