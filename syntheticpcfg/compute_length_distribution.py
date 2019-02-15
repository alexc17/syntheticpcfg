# Sample from a given grammar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import argparse
import numpy.random
from numpy.random import RandomState
import math
import numpy as np
import pcfgfactory
import pcfg
import utility
import inside

parser = argparse.ArgumentParser(description='Compute the expected length distribution of a given PCFG')

parser.add_argument("inputfilename", help="File where the given PCFG is.")
parser.add_argument("--maxlength", help="limit samples to this length",type=int,default=30)
parser.add_argument("--pdf", help="Save a diagram as a pdf")
## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = pcfg.load_pcfg_from_file(args.inputfilename)
upcfg = mypcfg.make_unary()
insider = inside.InsideComputation(upcfg)
res = 1.0
x = np.arange(1, args.maxlength + 1)
y = np.zeros(args.maxlength)

for length in range(1,args.maxlength + 1):
	s = (pcfg.UNARY_SYMBOL,) * length
	lp = insider.inside_log_probability(s)
	p = math.exp(lp)
	y[length-1] = p
	print(length,lp,p)
	res -= math.exp(lp)
print(">" + str(args.maxlength),math.log(res),res)
#print(x,y)
if args.pdf:

	plt.plot(x, y, label="Grammar")
	
	plt.legend()
	plt.xlabel('Length')
	plt.ylabel('Probability')

	plt.title("Length distribution")
	plt.savefig(args.pdf)
	plt.close()
