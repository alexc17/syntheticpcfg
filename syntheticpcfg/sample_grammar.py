#sample_grammar.py

import argparse
import random
import logging
import numpy.random

from . import pcfgfactory
from . import pcfg
from . import utility

parser = argparse.ArgumentParser(description='Create a randomly generated PCFG')

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")
parser.add_argument("--numbergrammars", help="Number of grammars to generate", default=1,type=int)

parser.add_argument("--terminals", help="Number of terminals", default=10000,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)
parser.add_argument("--binaryproductions", help="Number of binary productions", default=40,type=int)
parser.add_argument("--lexicalproductions", help="Number of lexical productions", default=10000,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)

parser.add_argument("--pitmanyor",help="Use Pitman Yor process for lexical probs",action="store_true")
parser.add_argument("--pitmanyora",help="PitmanYor a parameter",type=float,default=0.5)
parser.add_argument("--pitmanyorb",help="PitmanYor b parameter",type=float,default=0.5)
parser.add_argument("--sigma",help="Standard deviation for LogNormal prior (default = 3.0)",type=float,default=3.0)



parser.add_argument("--dirichletparam",help="Set Dirichlet for lexical probs (must be > 0)",type=float,default=-1.0)

parser.add_argument("--cdslength", help="Use distribution of lengths from corpus of English child directed speech.", action="store_true")
parser.add_argument("--wsjlength", help="Use distribution of lengths from corpus of WSJ text.", action="store_true")
parser.add_argument("--poisson", type=float, default=5.0, help="Use Zero truncated poisson.")

parser.add_argument("--nonstrictcnf", help="Allow start symbol to occur on right hand side of production.", action="store_true")

parser.add_argument("--corpuslength", help="Use distribution of lengths from a chosen corpus. Filename of corpus.")
parser.add_argument("--verbose",help="Print useful information",action="store_true")

# Parameters for training length distribution
LENGTH_EM_ITERATIONS = 10
LENGTH_EM_MAX_LENGTH = 20

# if we get within this ratio of the best possible length distribution then we terminate. 
TERMINATION_KLD = 0.05

parser.add_argument("--length_em_iterations",help="Maximum number of iterations to train legth distribution. (default 10)",type=int,default=10)
parser.add_argument("--length_em_max_length",help="Maximum length of strings to consider when training length distribution (default 20)",type=int,default=20)
parser.add_argument("--length_em_termination_kld",help="Threshold for terminating training of length distribution; default (0.05)",type=float,default=0.5)


header = []
args = parser.parse_args()

if args.verbose:
	logging.basicConfig(level=logging.INFO)

if args.seed:
	random.seed(args.seed)
	numpy.random.seed(args.seed)
	header.append("Random seed is %d " % args.seed)

pcfgfactory.LENGTH_EM_ITERATIONS = args.length_em_iterations
pcfgfactory.LENGTH_EM_MAX_LENGTH = args.length_em_max_length
pcfgfactory.TERMINATION_KLD = args.length_em_termination_kld

nts = args.nonterminals

if args.binaryproductions >  0.5 * nts * nts * nts:
	raise ValueError("High number of binary rules.")

if args.binaryproductions < nts:
	raise ValueError("Low number of binary rules.")

factory = pcfgfactory.PCFGFactory()

#print(utility.generateRandomString(10))
## Configure factory

factory.cfgfactory.number_nonterminals = args.nonterminals
factory.cfgfactory.number_terminals = args.terminals
factory.cfgfactory.binary_rules = args.binaryproductions
factory.cfgfactory.lexical_rules = args.lexicalproductions

header.append( "Nonterminals %d terminals %d binary_rules %d lexical_rules % d" % (args.nonterminals,args.terminals, args.binaryproductions, args.lexicalproductions) )

if args.nonstrictcnf:
	factory.cfgfactory.strict_cnf = False
	header.append("S allowed on RHS of productions.")

if args.pitmanyor:
	factory.lexical_distribution = pcfgfactory.StickBreakingPrior(a=args.pitmanyora,b=args.pitmanyorb)
	header.append("Lexical distribution: Pitman Yor  prior with discount %f, concentration %f " % (args.pitmanyora,args.pitmanyorb))
elif args.dirichletparam > 0:
	header.append("Lexical distribution: Dirichlet alpha =  %f " % args.dirichletparam)
	factory.lexical_distribution = pcfgfactory.LexicalDirichlet(dirichlet=args.dirichletparam)
else:
	header.append("Lexical distribution (default): Log normal , sigma =  %f " % args.sigma)
	factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=args.sigma)

if args.cdslength:
	header.append("Length distribution: Childes")
	factory.length_distribution.cds()
elif args.wsjlength:
	header.append("Length distribution: WSJ")
	factory.length_distribution.wsj()
elif args.corpuslength:
	header.append("Length distribution: sampled from corpus")
	factory.length_distribution.from_corpus(args.corpuslength)
else:
	header.append("Length distribution: Poisson %f" % args.poisson)
	factory.length_distribution.set_poisson(args.poisson, int(4 * args.poisson))




if args.numbergrammars > 1:
	assert '%d' in args.outputfilename
	
# sample and save
n = 0
while n < args.numbergrammars:
	try:
		pcfg = factory.sample()
		n += 1
		#header.extend(pcfg.useful_information())
		if args.numbergrammars > 1:
			fn = args.outputfilename % (n)
			if n == 1:
				header.append( "slice %d" % n)
			else:
				header[-1]  = "slice %d" % n
			pcfg.store(fn,header = header)
			print("Stored",fn, header)
		else:
			pcfg.store(args.outputfilename, header = header)
	except utility.ParseFailureException:
		print("Skipping a degenerate grammar.")



