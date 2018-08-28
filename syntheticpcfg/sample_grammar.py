#sample_grammar.py

import argparse

import pcfgfactory
import pcfg

parser = argparse.ArgumentParser(description='Create a randomly generated PCFG')

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")

parser.add_argument("--terminals", help="Number of terminals", default=100,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)
parser.add_argument("--binaryproductions", help="Number of binary productions", default=200,type=int)
parser.add_argument("--lexicalproductions", help="Number of lexical productions", default=200,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)

parser.add_argument("--pitmanyor",help="Use Pitman Yor process for lexical probs",action="store_true")

args = parser.parse_args()

nts = args.nonterminals

if args.binaryproductions >  nts * nts * nts:
	raise ValueError("Impossibly high number of binary rules.")


factory = pcfgfactory.PCFGFactory()

## Configure factory

factory.cfgfactory.number_nonterminals = args.nonterminals
factory.cfgfactory.number_terminals = args.terminals
factory.cfgfactory.binary_rules = args.binaryproductions
factory.cfgfactory.lexical_rules = args.lexicalproductions


if args.pitmanyor:
	factory.lexical_distribution = pcfgfactory.LexicalPitmanYor()




# sample and save
pcfg = factory.sample()

pcfg.store(args.outputfilename)



