# Make a zipf plot of the true and sampled rank-frequency plots.
import matplotlib
matplotlib.use('Agg')

import math

import argparse
import pcfg
from collections import Counter
from numpy.random import RandomState
import utility
import numpy.random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import numpy as np

#alpha = 1.0

def sample1(sampler, samples):
	mcte = Counter()
	for i in range(samples):
		tree = sampler.sample_tree()
		# defatul is string.
		s = utility.collect_yield(tree)
		for a in s:
			mcte[a] += 1
	return mcte

def plotsplit(m1,m2):
	joint = [ a for a in m1 if m2[a] > 0]
	rank_counts = np.array( [ m1[a] for a in joint])
	frequency_counts = np.array( [ m2[a] for a in joint])
	total = np.sum(rank_counts)
	xindices = np.argsort(-rank_counts)
	frequencies = frequency_counts[xindices]/total
	ranks = np.arange(1, len(joint) + 1)
	return ranks,frequencies

def plot2(sampler, samples):
	m1 = sample1(sampler,samples)
	m2 = sample1(sampler,samples)
	return plotsplit(m1,m2)


def plotcorpus(filename):
	m1 = Counter()
	m2 = Counter()
	with open(filename) as inf:
		for line in inf:
			if len(line) > 0 and not line[0] == '#':
				s = line.strip().split()
				for a in s:
					if numpy.random.randint(0,2) == 0:
						m1[a] += 1
					else:
						m2[a] += 1
	return plotsplit(m1,m2)

def plot_te(te):
	ranks = np.arange(1, len(te)+1)
	counts = np.array(list(te.values()))
	#print(counts)
	total = np.sum(counts)
	xindices = np.argsort(-counts)
	frequencies = counts[xindices]/ float(total)
	return ranks, frequencies

def make_rank_frequency(mypcfgs, prng, samples, filename, corpus):
	
	alpha = 1.0/math.sqrt(len(mypcfgs))
	for mypcfg in mypcfgs:
		te = mypcfg.terminal_expectations()
		ranks,frequencies = plot_te(te)
		plt.plot(ranks, frequencies, "b", alpha = alpha,rasterized=True)
		# mcte = Counter()
	if args.corpus:
		ranks,frequencies = plotcorpus(args.corpus)
	# else:
	# 	mysampler = pcfg.Sampler(mypcfg, random=prng)
		# for i in range(samples):
		# 	tree = mysampler.sample_tree()
		# 	# defatul is string.
		# 	s = utility.collect_yield(tree)
		# 	for a in s:
		# 		mcte[a] += 1
		# ranks,frequencies = plot2(mysampler, samples)
		plt.plot(ranks, frequencies, ".r",  alpha=0.3, rasterized=True)
	red_patch = mpatches.Patch(color='red', label='Corpus')
	blue_patch = mpatches.Patch(color='blue', label='Grammar')
	plt.legend(handles=[ blue_patch, red_patch])

	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Rank')
	plt.ylabel('Frequency')

	plt.title("lexical rank frequency plot")
	plt.savefig(filename)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make unigram rank frequency plot for a group of PCFGs, and optionally plot the same for a sampled corpus.')
	parser.add_argument("inputfilenames", help="File where the given PCFG is.", nargs='+',)
	parser.add_argument("--outputpdf", help="File where the PDF will be saved.", default="zipf.pdf")
	parser.add_argument("--seed",help="Choose random seed",type=int)
	parser.add_argument("--alpha",help="Transparency",default=0.1,type=float)
	
	parser.add_argument("--corpus",help="File containing some samples")
	
	parser.add_argument("--samples", type=int, default=100000,help="Number of samples, (default 10000)")
	args = parser.parse_args()
	print(args.inputfilenames)
	mypcfgs = [ pcfg.load_pcfg_from_file(f) for f in args.inputfilenames]
#	mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
	if args.seed:
		print("Setting seed to ",args.seed)
		prng = RandomState(args.seed)
	else:
		prng = RandomState()
	alpha = args.alpha
	make_rank_frequency(mypcfgs, prng, args.samples, args.outputpdf,args.corpus)

