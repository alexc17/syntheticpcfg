#sample_grammar.py

import argparse
import random
import numpy.random
import pcfgfactory
import pcfg
import utility

parser = argparse.ArgumentParser(description='Create a randomly generated PCFG')

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")
parser.add_argument("--numbergrammars", help="Number of grammars to generate", default=1,type=int)

parser.add_argument("--terminals", help="Number of terminals", default=100,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)
parser.add_argument("--binaryproductions", help="Number of binary productions", default=200,type=int)
parser.add_argument("--lexicalproductions", help="Number of lexical productions", default=200,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)

parser.add_argument("--pitmanyor",help="Use Pitman Yor process for lexical probs",action="store_true")
parser.add_argument("--dirichletparam",help="Set Dirichlet for lexical probs",type=float,default=1.0)

parser.add_argument("--cdslength", help="Use distribution of lengths from corpus of English child directed speech.", action="store_true")
parser.add_argument("--wsjlength", help="Use distribution of lengths from corpus of WSJ text.", action="store_true")

parser.add_argument("--nonstrictcnf", help="Allow start symbol to occur on right hand side of production.", action="store_true")

parser.add_argument("--corpuslength", help="Use distribution of lengths from a chosen corpus. Filename of corpus.")

# Parameters for training length distribution
LENGTH_EM_ITERATIONS = 10
LENGTH_EM_MAX_LENGTH = 20

# if we get within this ratio of the best possible length distribution then we terminate. 
TERMINATION_KLD = 0.05

parser.add_argument("--length_em_iterations",help="Maximum number of iterations to train legth distribution. (default 10)",type=int,default=10)
parser.add_argument("--length_em_max_length",help="Maximum length of strings to consider when training length distribution (default 20)",type=int,default=20)
parser.add_argument("--length_em_termination_kld",help="Threshold for terminating training of length distribution; default (0.05)",type=float,default=0.5)



args = parser.parse_args()
pcfgfactory.LENGTH_EM_ITERATIONS = args.length_em_iterations
pcfgfactory.LENGTH_EM_MAX_LENGTH = args.length_em_max_length
pcfgfactory.TERMINATION_KLD = args.length_em_termination_kld

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
	factory.lexical_distribution = pcfgfactory.LexicalDirichlet(dirichlet=args.dirichletparam)

if args.cdslength:
	factory.length_distribution.cds()
elif args.wsjlength:
	factory.length_distribution.wsj()
elif args.corpuslength:
	factory.length_distribution.from_corpus(args.corpuslength)


if args.numbergrammars > 1:
	assert '%d' in args.outputfilename
	
# sample and save
n = 0
while n < args.numbergrammars:
	try:
		pcfg = factory.sample()
		n += 1
		if args.numbergrammars > 1:
			fn = args.outputfilename % (n)
			pcfg.store(fn)
			print("Stored",fn)
		else:
			pcfg.store(args.outputfilename)
	except utility.ParseFailureException:
		print("Skipping a degenerate grammar.")



