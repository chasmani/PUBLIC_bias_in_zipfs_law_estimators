
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from design_scheme import PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, POINT_SIZE, LINEWIDTH


def plot_from_pickle(pickle_name="../data/simulated/overnight_results_gen_10.pkl"):

	df = pd.read_pickle(pickle_name)
	sns.kdeplot(data=df, x="exponent", y="q", weights="w", shade=True, color=PRIMARY_COLOR)

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

	sns.scatterplot(x=[1.2],y=[4], color="white", marker="x", s=POINT_SIZE*2, linewidth=LINEWIDTH+1)
	sns.scatterplot(x=[1.2],y=[4], color=SECONDARY_COLOR, marker="x", s=POINT_SIZE*1.5, linewidth=LINEWIDTH, label="Actual")

	sns.scatterplot(x=[exponents[i]],y=[qs[j]], color=TERTIARY_COLOR, s=POINT_SIZE*2, label="Mle")

	plt.xlabel("$\lambda$")
	plt.ylabel("$q$")

	plt.tight_layout()
	
	plt.legend()
	plt.savefig("images/zipf_mandelbrot_posterior_from_wabc.png", dpi=300)
	
	plt.show()

plot_from_pickle()