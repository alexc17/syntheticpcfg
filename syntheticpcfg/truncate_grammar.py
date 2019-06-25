#truncate_grammar.py

import argparse
import random
import numpy.random

import pcfg
import utility

from collections import Counter

parser = argparse.ArgumentParser(description='Sample N trees from a PCFG and return the ML estimate based on those trees.')

parser.add_argument("inputfilename", help="File where the original PCFG is.")
parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored.")

parser.add_argument("--samples", help="Number of samples (default 10000)", default=10000,type=int)
parser.add_argument("--seed", help="Random seed", default=None,type=int)



args = parser.parse_args()


if args.seed:
	print("Setting seed to ",args.seed)
	prng = RandomState(args.seed)
else:
	prng = RandomState()




original = pcfg.load_pcfg_from_file(args.inputfilename)

mysampler = pcfg.Sampler(original, random=prng)
counter = Counter()

for _ in range(args.samples):
	tree = mysampler.sample_tree()
	utility.count_productions(tree, counter)

ml = pcfg.PCFG()

for production,n in counter.items():
	ml.productions.append(production)
	ml.nonterminals.add(production[0])
	if len(production) == 2:
		ml.terminals.add(production[1])
	ml.parameters[production] = float(n)

ml.normalise()
ml.store(args.outputfilename,header=[  "Resampled from %d samples" % args.samples])




