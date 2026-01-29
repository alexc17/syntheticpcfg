#sample_grammar.py

import argparse
import random
import numpy.random

from . import pcfgfactory
from . import pcfg
from . import utility


parser = argparse.ArgumentParser(description='Replace low probability tokens with an UNK token for a given PCFG')

parser.add_argument("inputfilename", help="File where the original PCFG is.")
parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored.")

parser.add_argument("--threshold", help="Probability (not expectation) default 1e-5", default=1e-5,type=float)

parser.add_argument("--unk", help="Symbol to use, default UNK", default="UNK")
args = parser.parse_args()

original = pcfg.load_pcfg_from_file(args.inputfilename)

newterminals = set()
te = original.terminal_expectations()
L = original.expected_length()

for a,e in te.items():
	if e * L > args.threshold:
		newterminals.add(a)

unk_probs = { nt:0 for nt in original.nonterminals }

newproductions  = []
newparameters = {}
for prod,e in original.parameters.items():
	if len(prod) == 3:
		newproductions.append(prod)
		newparameters[prod] = e
	else:
		nt,a = prod
		if a in newterminals:
			newproductions.append(prod)
			newparameters[prod] = e
		else:
			unk_probs[nt] += e

for nt,p in unk_probs.items():
	if p > 0:
		prod = (nt,args.unk)
		newproductions.append(prod)
		newparameters[prod] = p

original.terminals = newterminals
original.productions = newproductions
original.parameters = newparameters

original.store(args.outputfilename,header=[  "Unkified with symbol %s, threshold %e" % (args.unk,args.threshold)])




