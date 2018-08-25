# Sample from a given grammar
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import argparse
import numpy.random
import numpy as np
from numpy.random import RandomState
import math






import pcfgfactory
import pcfg
import utility
import inside

parser = argparse.ArgumentParser(description='Compute the lexical distribution of a given PCFG')

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("--plot", help="filename for pdf to be saved if you want it to be plotted.")
## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
te = mypcfg.terminal_expectations()

probs = [ te[a] for a in te]

probs.sort(key = lambda x : -x)
for p in probs:
	print(p)

if args.plot:
	# Plot it and save it in a pdf

	# A Zipf plot
	ranks = np.arange(1, len(probs)+1)
	plt.plot(ranks, probs, rasterized=True,marker=".")
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Rank')
	plt.ylabel('Frequency')

	plt.title("lexical rank frequency plot")
	plt.savefig(args.plot)
	plt.gcf().clear()  
## compute the lexical expectations and then sort them into 

