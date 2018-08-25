# Sample from a given grammar


import argparse
import numpy.random
from numpy.random import RandomState
import math

import pcfgfactory
import pcfg
import utility
import inside

parser = argparse.ArgumentParser(description='Compute the expected length distribution of a given PCFG')

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("--maxlength", help="limit samples to this length",type=int,default=30)

## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
upcfg = mypcfg.make_unary()
insider = inside.InsideComputation(upcfg)
res = 1.0
for length in range(1,args.maxlength + 1):
	s = (pcfg.UNARY_SYMBOL,) * length
	lp = insider.inside_log_probability(s)
	print(length,lp,math.exp(lp))
	res -= math.exp(lp)
print(">" + str(args.maxlength),math.log(res),res)


