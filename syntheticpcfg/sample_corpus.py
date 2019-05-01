# Sample from a given grammar


import argparse
import numpy.random
from numpy.random import RandomState

import pcfgfactory
import pcfg
import utility
import inside

import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")
    
parser = argparse.ArgumentParser(description="""Sample from a given PCFG.

 default gives three log probabilities and a tree. 
 the derivation
 all derivations with the same bracketed tree and yield
 all derivations with the same yield.

The last one is expensive to compute if the string is long and the grammar large.


 """)

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("outputfilename", help="File where the resulting corpus will be stored")

parser.add_argument("--n", help="Number of samples (default 1000)", default=1000,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)
parser.add_argument("--maxlength", help="limit samples to this length",type=int)
parser.add_argument("--omitprobs", help="don't compute any probabilities",action="store_true")
parser.add_argument("--omitinside", help="don't compute the inside probabilities",action="store_true")
parser.add_argument("--yieldonly", help="just output the yield",action="store_true")


## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)


if args.seed:
	print("Setting seed to ",args.seed)
	prng = RandomState(args.seed)
else:
	prng = RandomState()


mysampler = pcfg.Sampler(mypcfg, random=prng)
insider = inside.InsideComputation(mypcfg)

with open(args.outputfilename,'w') as outf:
	i = 0
	while i < args.n:
		tree = mysampler.sample_tree()
		# defatul is string.
		s = utility.collect_yield(tree)
		if not args.maxlength or len(s) <= args.maxlength:
			if not args.omitprobs:
				
				lpt = mypcfg.log_probability_derivation(tree)
				lpb = insider._bracketed_log_probability(tree)[mypcfg.start]
				if args.omitinside:
					outf.write( "%e %e " % ( lpt, lpb))
				else:
					lps = insider.inside_log_probability(s)
					outf.write( "%e %e %e " % ( lpt, lpb, lps))
			if args.yieldonly:
				outf.write(" ".join(s) + "\n")
			else:
				outf.write(utility.tree_to_string(tree) + "\n")
			i += 1




