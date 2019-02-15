# Make a zipf plot of the true and sampled rank-frequency plots.
import matplotlib
matplotlib.use('Agg')

import argparse
import pcfg
from collections import Counter
from numpy.random import RandomState
import utility


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import numpy as np

alpha = 1.0

def sample1(sampler, samples):
	mcte = Counter()
	for i in range(samples):
		tree = sampler.sample_tree()
		# defatul is string.
		s = utility.collect_yield(tree)
		for a in s:
			mcte[a] += 1
	return mcte

def plot2(sampler, samples):
	m1 = sample1(sampler,samples)
	m2 = sample1(sampler,samples)
	joint = [ a for a in m1 if m2[a] > 0]
	rank_counts = np.array( [ m1[a] for a in joint])
	frequency_counts = np.array( [ m2[a] for a in joint])
	total = np.sum(rank_counts)
	xindices = np.argsort(-rank_counts)
	frequencies = frequency_counts[xindices]/total
	ranks = np.arange(1, len(joint) + 1)
	return ranks,frequencies

def plot_te(te):
	ranks = np.arange(1, len(te)+1)
	counts = np.array(list(te.values()))
	#print(counts)
	total = np.sum(counts)
	xindices = np.argsort(-counts)
	frequencies = counts[xindices]/ float(total)
	return ranks, frequencies

def make_rank_frequency(mypcfg, prng, samples, filename):
	print("Making true lexical rank frequency plots")
	te = mypcfg.terminal_expectations()
	ranks,frequencies = plot_te(te)
	plt.plot(ranks, frequencies, "b", rasterized=True,label="Exact")
	# mcte = Counter()
	mysampler = pcfg.Sampler(mypcfg, random=prng)
	# for i in range(samples):
	# 	tree = mysampler.sample_tree()
	# 	# defatul is string.
	# 	s = utility.collect_yield(tree)
	# 	for a in s:
	# 		mcte[a] += 1
	ranks,frequencies = plot2(mysampler, samples)
	plt.plot(ranks, frequencies, ".r", alpha = alpha, rasterized=True,label="Sampled")
	
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Rank')
	plt.ylabel('Frequency')

	plt.title("lexical rank frequency plot")
	plt.savefig(filename)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make unigram rank frequency plot for a  PCFG')
	parser.add_argument("inputfilename", help="File where the given PCFG is.")
	parser.add_argument("outputfilename", help="File where the PDF will be saved.")
	parser.add_argument("--seed",help="Choose random seed",type=int)
	parser.add_argument("--alpha",help="Transparency",default=0.1,type=float)

	parser.add_argument("--samples", type=int, default=100000,help="Number of samples, (default 10000)")
	args = parser.parse_args()

	mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
	if args.seed:
		print("Setting seed to ",args.seed)
		prng = RandomState(args.seed)
	else:
		prng = RandomState()
	alpha = args.alpha
	make_rank_frequency(mypcfg, prng, args.samples, args.outputfilename)

