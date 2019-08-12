#sample_fullgrammar.py

import argparse
import random
import numpy.random
import pcfgfactory
import pcfg
import utility
import logging

# set from command line flag



parser = argparse.ArgumentParser(description="""Create a randomly generated PCFG, 
	with a trivial CFG backbone -- i.e. all  possible rules in CNF.

	Defaults should give something that looks roughly like Child directed speech.
	Length distribution : zero truncated poisson with mean length 5.
	Binary rules samples from Dirichlet with parameter set to keep entropy reasonable.
	Lexical rules sampled from a Log normal to get long tails.
	10 nonterminals and 10^4 terminals.

	If you want to generate multiple grammars use a %d in the outputfilename which will be filled with
	a number.
	""")

parser.add_argument("outputfilename", help="File where the resulting PCFG will be stored")
parser.add_argument("--numbergrammars", help="Number of grammars to generate", default=1,type=int)

parser.add_argument("--terminals", help="Number of terminals.", default=20000,type=int)
parser.add_argument("--nonterminals", help="Number of nonterminals", default=10,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)
parser.add_argument("--verbose",help="Print useful information",action="store_true")

## Lexical options
parser.add_argument("--pitmanyor",help="Use Pitman Yor process for lexical probs",action="store_true")
parser.add_argument("--pitmanyora",help="PitmanYor a parameter",type=float,default=0.5)
parser.add_argument("--pitmanyorb",help="PitmanYor b parameter",type=float,default=0.5)
parser.add_argument("--lognormal",help="Use LogNormal prior for lexical probs",action="store_true")
parser.add_argument("--sigma",help="Standard deviation for LogNormal prior",type=float,default=4.0)
parser.add_argument("--dirichletlexical",help="Set Dirichlet for lexical probs",type=float,default=-1.0)

parser.add_argument("--dirichletbinary",help="Set Dirichlet for binary probs (default is (|V-1|^2 + 1)^-1 )",type=float,default=-1.0)

## Length options
parser.add_argument("--cdslength", help="Use distribution of lengths from corpus of English child directed speech.", action="store_true")
parser.add_argument("--wsjlength", help="Use distribution of lengths from corpus of WSJ text.", action="store_true")
parser.add_argument("--poisson", type=float, default=5.0, help="Use Zero truncated poisson.")
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

if args.verbose:
	logging.basicConfig(level=logging.INFO)
pcfgfactory.LENGTH_EM_ITERATIONS = args.length_em_iterations
pcfgfactory.LENGTH_EM_MAX_LENGTH = args.length_em_max_length
pcfgfactory.TERMINATION_KLD = args.length_em_termination_kld


header = [ "Nonterminals %d Terminals %d " % (args.nonterminals,args.terminals)]
if args.seed:
	random.seed(args.seed)
	numpy.random.seed(args.seed)
	logging.info("setting seed to %d" % args.seed)
	header.append("Random seed %d" % args.seed)

factory = pcfgfactory.FullPCFGFactory(nonterminals = args.nonterminals, terminals = args.terminals)

if args.pitmanyor:
	factory.lexical_distribution = pcfgfactory.StickBreakingPrior(a=args.pitmanyora,b=args.pitmanyorb)
	header.append("Lexical distribution: Pitman Yor  prior with discount %f, concentration %f " % (args.pitmanyora,args.pitmanyorb))
elif args.dirichletlexical > 0:
	header.append("Lexical distribution: Dirichlet prior with alpha %f " % args.dirichletlexical)
	factory.lexical_distribution = pcfgfactory.LexicalDirichlet(dirichlet=args.dirichletlexical)	
else:
	header.append("Lexical distribution: log normal prior with standard deviation %f " % args.sigma)
	factory.lexical_distribution = pcfgfactory.LogNormalPrior(sigma=args.sigma)

if args.dirichletbinary > 0:
	alpha = args.dirichletbinary
else:
	alpha = 1.0 / ( (args.nonterminals -1) **2   + 1 )

header.append("Binary distribution: Dirichlet with parameter %f " % alpha)
factory.binary_distribution = pcfgfactory.LexicalDirichlet(dirichlet = alpha)

if args.cdslength:
	header.append("Length distribution: Childes")
	factory.length_distribution.cds()
elif args.wsjlength:
	header.append("Length distribution: WSJ")
	factory.length_distribution.wsj()
elif args.corpuslength:
	header.append("Length distribution: estimated from corpus ")
	factory.length_distribution.from_corpus(args.corpuslength)
else:
	header.append("Length distribution: Poisson with parameter %f " % args.poisson)
	factory.length_distribution.set_poisson(args.poisson, int(4 * args.poisson))




if args.numbergrammars > 1:
	assert '%d' in args.outputfilename
	
# sample and save
n = 0
while n < args.numbergrammars:
	try:
		pcfg = factory.sample_smart()
		n += 1
		if args.numbergrammars > 1:
			fn = args.outputfilename % (n)
			header.append("Slice %d" % n)
			pcfg.store(fn,header=header)
			logging.info("Stored",fn)
		else:
			pcfg.store(args.outputfilename,header=header)
	except utility.ParseFailureException:
		logging.warning("Skipping a degenerate grammar.")



