# Sample from a given grammar


import argparse
import numpy.random
from numpy.random import RandomState
import math

import pcfgfactory
import pcfg
import utility
import inside

parser = argparse.ArgumentParser(description='Compute the lexical distribution of a given PCFG')

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("--plot", help="filename for pdf to be saved if you want it to be plotted.")
## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)


## compute the lexical expectations and then sort them into 

