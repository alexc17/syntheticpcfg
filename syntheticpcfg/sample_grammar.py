#sample_grammar.py

import argparse

import pcfgfactory
import pcfg

parser = argparse.ArgumentParser(description='Create a randomly generated PCFG')

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")

parser.add_argument("--terminals", help="Number of terminals", default=100,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)
parser.add_argument("--binaryproductions", help="Number of binary productions", default=100,type=int)
parser.add_argument("--lexicalproductions", help="Number of lexical productions", default=100,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)


args = parser.parse_args()

nts = args.nonterminals

if args.binaryproductions >  nts * nts * nts:
	raise ValueError("Impossibly high number of binary rules.")


factory = pcfgfactory.PCFGFactory()

## Configure factory

pcfg = factory.sample()

pcfg.store(args.outputfilename)



