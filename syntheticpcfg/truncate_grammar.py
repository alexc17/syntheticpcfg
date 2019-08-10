#truncate_grammar.py

import argparse
import random
import numpy.random

import pcfg
import utility

from collections import Counter

parser = argparse.ArgumentParser(description='remove all low expectation productions and renormalise.')

parser.add_argument("inputfilename", help="File where the original PCFG is.")
parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored.")

parser.add_argument("--eproduction", help="threshold of expectation for a production (default 1e-5)", default=1e-5,type=float)
parser.add_argument("--enonterminal", help="threshold of expectation for a Nonterminal (default 1e-4)", default=1e-4,type=float)


args = parser.parse_args()

enonterminal = args.enonterminal
eproduction = args.eproduction

assert enonterminal < 1.0
assert eproduction < enonterminal


original = pcfg.load_pcfg_from_file(args.inputfilename)


ml = pcfg.PCFG()
pe = original.production_expectations()
nte = original.nonterminal_expectations()
ml.nonterminals = set(nt for nt,e in nte.items() if e > enonterminal)
ml.start = original.start
assert ml.start in ml.nonterminals

ml.productions = [prod for prod,e in pe.items() if e > eproduction and prod[0] in ml.nonterminals]
finalnts = set( prod[0] for prod in ml.productions)
assert len(finalnts) == len(ml.nonterminals)
rhs = set()
for prod in ml.productions:
	if len(prod) == 3:
		rhs.add(prod[1])
		rhs.add(prod[2])
assert len(rhs) + 1 == len(ml.nonterminals)

ml.terminals = set(prod[1] for prod in ml.productions if len(prod) == 2 )
ml.parameters = { prod : original.parameters[prod] for prod in ml.productions }
ml.normalise()
ml.store(args.outputfilename,header=[  "Pruned with  %e, %e threshold " % (enonterminal,eproduction)])




