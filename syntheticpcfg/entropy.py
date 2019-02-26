import argparse
import pcfg

import numpy.random
from numpy.random import RandomState




parser = argparse.ArgumentParser(description='Estimate information theoretic quantities of a PCFG')
parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("--seed",help="Choose random seed",type=int)
parser.add_argument("--maxlength",help="Max length of strings",type=int,default=50)

parser.add_argument("--alpha",help="Transparency",default=0.1,type=float)

parser.add_argument("--samples", type=int, default=10000,help="Number of samples, (default 10000)")
args = parser.parse_args()

if args.seed:
	print("Setting seed to ",args.seed)
	prng = RandomState(args.seed)
else:
	prng = RandomState()
mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
mysampler = pcfg.Sampler(mypcfg,random=prng)

mysampler.estimate_string_entropy(args.samples,max_length=args.maxlength)

