

import argparse

import pcfg

parser = argparse.ArgumentParser(description='Produce some useful information about a given PCFG and/or corpus')

parser.add_argument("inputfilename", help="File where the given PCFG is.")


## Other options: control output format, what probs are calculated.


args = parser.parse_args()

mypcfg = 
## Compute useful stuff and output it.
# Connectivity
# expectations of each production, terminal, nonterminal, 
# expecetd length from each string.
# true distribution of lengths.
# Ambiguity
# thickness and width. 


