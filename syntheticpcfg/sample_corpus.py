# Sample from a given grammar


import argparse

import pcfgfactory
import pcfg
import utility

parser = argparse.ArgumentParser(description='Sample from a given PCFG')

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("outputfilename", help="File where the resulting corpus will be stored")

parser.add_argument("--n", help="Number of samples", default=100,type=int)

parser.add_argument("--seed",help="Choose random seed",type=int)


## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)

mysampler = pcfg.Sampler(mypcfg)

with open(args.outputfilename,'w') as outf:
	for i in range(args.n):	
		tree = mysampler.sample_tree()
		# defatul is string.
		s = utility.collect_yield(tree)
		outf.write(" ".join(s) + "\n")



