# Sample from a given grammar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import argparse
import math
import collections

import numpy as np
import numpy.random
from numpy.random import RandomState

from . import pcfgfactory
from . import pcfg
from . import utility
from . import inside

parser = argparse.ArgumentParser(description='Compute the expected length distribution of a given PCFG')

parser.add_argument("inputfilenames", help="Files where the given PCFGs are.",nargs="+")
parser.add_argument("--corpus", help="Plot empirical length distribution of a corpus: filename")
parser.add_argument("--maxlength", help="limit samples to this length",type=int,default=20)
parser.add_argument("--pdf", help="Location of output file pdf, default length.pdf", default="length.pdf")
parser.add_argument("--logscale", help="Y axis in log scale",action="store_true")
## Other options: control output format, what probs are calculated.


args = parser.parse_args()
l = args.maxlength + 1
x = np.arange(1, l)
ys = []
for f in args.inputfilenames:
	mypcfg = pcfg.load_pcfg_from_file(f)
	print(f)
	upcfg = mypcfg.make_unary()

	insider = inside.UnaryInside(upcfg)
	table = insider.compute_inside_smart(l)
	start = insider.start
	y = np.zeros(args.maxlength)

	for length in range(1,l):
			
		p = table[length,start]
		print(length,p)
		y[length-1] = p
	ys.append(y)

alpha = 1.0/math.sqrt(len(ys))
for y in ys:
	plt.plot(x, y, 'b',alpha=alpha)

if args.corpus:
	y = np.zeros(args.maxlength)
	N = 0
	with open(args.corpus,'r') as inc:
		for line in inc:
			ll = len(line.split())
			if ll < l and ll > 0:
				y[ll-1] +=1
				N += 1
	print(y,N)
	plt.plot(x, y/N, 'r')

if args.corpus:
	red_patch = mpatches.Patch(color='red', label='Corpus')
	blue_patch = mpatches.Patch(color='blue', label='Grammar')
	plt.legend(handles=[ blue_patch, red_patch])
if args.logscale:
	plt.yscale('log')
plt.xlabel('Length')
plt.ylabel('Probability')

plt.title("Length distribution")
plt.savefig(args.pdf)
plt.close()
