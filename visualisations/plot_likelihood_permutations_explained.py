
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import string

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns

from utilities.probability_distributions import get_probabilities_power_law_finite_event_set

from design_scheme import PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, POINT_SIZE, LINEWIDTH


NX_KWARGS = {
	"node_size":800,
	"width":5,
	"arrowsize":25
}



def plot_prob_dist_mapping_onto_empirical_ranks():
	"""
	A figure showing the empirical ranking, the probabilty dsitribution and the bipartite matching
	"""

	W = 6
	lamb = 1.1
	N = 100


	colors = [PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR]
	PALETTE = sns.color_palette(colors)

	plt.rcParams.update({'font.size': 22})
	f, axs = plt.subplots(3,1, figsize=(10,10))
	plt.subplot(3,1,1)
	
	ns = [8, 6, 3, 2, 1, 1]

	sns.barplot(x=list(range(1,W+1)), y=ns, palette=PALETTE)
	
	plt.xlabel("Empirical rank, i")
	plt.ylabel("Frequency, n(i)")

	ax = plt.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	bars = [r for r in ax.get_children() if type(r)==Rectangle]
	colors = [c.get_facecolor() for c in bars[:-1]]
	
	plt.subplot(3,1,2)

	plot_bipartite_graph_to_seperate_file(W, colors)

	


	############################################################
	# Probability Ranks 

	plt.subplot(3,1,3)

	
	ps = get_probabilities_power_law_finite_event_set(lamb, W)
	#plt.scatter(x=list(range(1,W+1)), y=ps)
	sns.barplot(x=list(range(1,W+1)), y=ps, palette=PALETTE)
	#plt.yscale("log")


	ax = plt.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)


	plt.xlabel("Probability rank, j")
	plt.ylabel("p(j)")

	

	
	

	plt.tight_layout()


	plt.savefig("images/emprirical_vs_prob_ranking_and_bipartite_graph_w_6.png", dpi=300)
	plt.show()


def plot_bipartite_graph_to_seperate_file(W=6, colors=["grey"]*6):


	plt.rcParams["figure.figsize"] = (10,4)

	B=nx.DiGraph()
	tops = ["r{}".format(index) for index in range(1, W+1)]
	bottoms = ["z{}".format(index) for index in range(1, W+1)]

	B.add_nodes_from(tops)
	B.add_nodes_from(bottoms)
	
	layout = nx.spring_layout(B, scale=0.1)

	s = [2,1,5,3,4,6]

	for index in range(len(tops)):

		B.add_edge(bottoms[index], tops[s[index]-1])

		right_offset = 5*index		
		
		layout[tops[index]][0] = right_offset
		layout[tops[index]][1] = 2

		layout[bottoms[index]][0] = right_offset		
		layout[bottoms[index]][1] = -2

	nx.draw_networkx(B, pos=layout, node_color=colors*2, with_labels=False, **NX_KWARGS)
	plt.box(False)

	axis = plt.gca()
	axis.set_xlim([1.1*x for x in axis.get_xlim()])
	axis.set_ylim([1.1*y for y in axis.get_ylim()])




if __name__=="__main__":
	plot_prob_dist_mapping_onto_empirical_ranks()