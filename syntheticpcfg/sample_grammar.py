#sample_grammar.py

import argparse
import random
import numpy.random
import pcfgfactory
import pcfg

parser = argparse.ArgumentParser(description='Create a randomly generated PCFG')

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")
parser.add_argument("--numbergrammars", help="Number of grammars to generate", default=1,type=int)

parser.add_argument("--terminals", help="Number of terminals", default=100,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)
parser.add_argument("--binaryproductions", help="Number of binary productions", default=200,type=int)
parser.add_argument("--lexicalproductions", help="Number of lexical productions", default=200,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)

parser.add_argument("--pitmanyor",help="Use Pitman Yor process for lexical probs",action="store_true")
parser.add_argument("--dirichletparam",help="Set Dirichlet for lexical probs",type=int,default=1.0)

parser.add_argument("--cdslength", help="Use distribution of lengths from corpus of English child directed speech.", action="store_true")
parser.add_argument("--nonstrictcnf", help="Allow start symbol to occur on right hand side of production.", action="store_true")

parser.add_argument("--corpuslength", help="Use distribution of lengths from a chosen corpus. Filename of corpus.")


args = parser.parse_args()

nts = args.nonterminals

if args.binaryproductions >  0.5 * nts * nts * nts:
	raise ValueError("High number of binary rules.")

if args.binaryproductions < nts:
	raise ValueError("Low number of binary rules.")

factory = pcfgfactory.PCFGFactory()

## Configure factory

factory.cfgfactory.number_nonterminals = args.nonterminals
factory.cfgfactory.number_terminals = args.terminals
factory.cfgfactory.binary_rules = args.binaryproductions
factory.cfgfactory.lexical_rules = args.lexicalproductions

if args.seed:
	random.seed(args.seed)
	numpy.random(args.seed)
if args.nonstrictcnf:
	factory.cfgfactory.strict_cnf = False
if args.pitmanyor:
	factory.lexical_distribution = pcfgfactory.LexicalPitmanYor()
else:
	factory.lexical_distribution = pcfgfactory.LexicalDirichlet(dirichlet=args.dirichlet)

if args.cdslength:
	factory.length_distribution.cds()
elif args.corpuslength:
	factory.length_distribution.from_corpus(args.corpuslength)


if args.numbergrammars > 1:
	assert '%d' in args.outputfilename
	
# sample and save
for i in args.numbergrammars:
	pcfg = factory.sample()
	if args.numbergrammars > 1:
		pcfg.store(args.outputfilename % (i+1))
	else:
		pcfg.store(args.outputfilename)



