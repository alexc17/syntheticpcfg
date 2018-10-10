#oracle_learner.py

import math
import utility
import inside
import pcfg

from utility import ParseFailureException
from collections import Counter

max_length = 6
max_ratios = 1000

def load_sentences(file):
		sentences = []
		with open(file) as inf:
			for line in inf:
				s = line.split()
				if len(s) > 0:
					sentences.append(tuple(s))
					
		return sentences

class OracleLearner:
	def __init__(self, insider, sentences):
		self.insider = insider
		self.sentences = sentences
		self.start = None
		self.kernels = None
		self.te = None
		self.cache = {}
		self.terminals = self.gather_terminals()

	def find_predictive_context(self, a, max_tests = 100):
		## find context that maximises probability of a
		best = None
		bestcp = 0.0
		n = 0
		for s in self.sentences:
			for i,b in enumerate(s):
				if b == a:
					p = self.probability(s)

					l = s[:i]
					r = s[i+1:]
					q = self.probability(l + ("",) + r)
					n += 1
					cp = p/q
					if cp > bestcp:
						bestcp = cp
						best = (l,r)
			if n > max_tests:
				break
		return best, bestcp

	def find_most_predicted_string(self,l,r):
		bestp = 0
		best = None
		for a in self.terminals:
			p = self.probability( l + (a,) + r)/self.te[a]
			if p > bestp:
				bestp =p
				best = a
		return best

	def pick_kernels(self):
		F = {}
		K = set()
		for a in self.terminals:
			f, cp = self.find_predictive_context(a)
			F[a] = f
			a = self.find_most_predicted_string(*f)
			K.add(a)
		self.kernels = list(K)
		self.features = F

	def learn(self):
		self.estimate_terminal_expectations()
		self.pick_kernels()
		self.start = self.pick_start()
		ntmap = {self.start: "S"}
		for a in self.kernels:
			if a != self.start:
				ntmap[a] = "NT" + a
		
		if not self.te:
			self.te = estimate_terminal_expectations(sentences)
		parameters = {}
		for a in self.kernels:
			for b in self.terminals:
				prod = (ntmap[a],b)
				xi = min(self.test_unary(a,b).values())
				if xi > 0:
					parameters[prod] =xi
		for a in self.kernels:
			for b in self.kernels:
				for c in self.kernels:
					prod = (ntmap[a],ntmap[b],ntmap[c])
					xi = min(self.test_binary(a,b,c).values())
					if xi > 0:
						parameters[prod] =xi
		lpcfg = pcfg.PCFG()
		lpcfg.terminals =self.terminals
		lpcfg.nonterminals = ntmap.values()
		lpcfg.start = "S"
		lpcfg.parameters = parameters
		lpcfg.productions = list(parameters)
		lpcfg.log_parameters = { prod: math.log(parameters[prod]) for prod in parameters}
		return lpcfg

	def reestimate(learned_grammar):
		insidel = inside.InsideComputation(learned_grammar)
		counter = Counter()
		for s in self.sentences:
			tree  = insidel.viterbi_parse(s)
			utility.count_productions(tree, counter)
		return counter

	def pick_start(self):
	
		return max(self.kernels, key = lambda x : self.probability((x,)))

	def probability(self,s):
		if s in self.cache:
			return self.cache[s]
		try:
			lp = self.insider.inside_log_probability(s)
			p = math.exp(lp)
		except utility.ParseFailureException:
			p = 0
		self.cache[s] = p
		return p
	
	def gather_terminals(self):
		terminals = set()
		for s in self.sentences:
			for a in s:
				terminals.add(a)
		return terminals

	def estimate_terminal_expectations(self):
		lcounter = Counter()
		n = len(self.sentences)
		for s in self.sentences:
			for a in s:
				lcounter[a] += 1
		self.te =  { a: lcounter[a]/n for a in lcounter}

	def test_binary(self, a,b,c):
		ratios = {}
		te = self.te
		for s in self.sentences:
			if len(s) < max_length:
				for i,aa in enumerate(s):
					if aa == a:
						l = s[:i]
						r = s[i+1:]
						if not (l,r) in ratios:
							denom = self.insider.inside_log_probability(s)
							try:
								num = self.insider.inside_log_probability(l + (b,c) + r)
								scale =  te[a] / (te[b] * te[c])
								ratios[(l,r)] = math.exp(num - denom) * scale
								if len(ratios) >= max_ratios:
									return ratios
							except utility.ParseFailureException:
								return { (l,r): 0.0 }
		return ratios

								
	def test_unary(self, a,b, scale_terminal = False):
		ratios = {}
		te = self.te
		for s in self.sentences:
			if len(s) < max_length:
				for i,aa in enumerate(s):
					if aa == a:
						l = s[:i]
						r = s[i+1:]
						if not (l,r) in ratios:
							denom = self.insider.inside_log_probability(s)
							try:
								num = self.insider.inside_log_probability(l + (b,) + r)
								if scale_terminal:
									scale =  te[a] / te[b] 
								else:
									scale =  te[a]
								ratios[(l,r)] = math.exp(num - denom) * scale
								if len(ratios) >= max_ratios:
									return ratios
							except utility.ParseFailureException:
								#print("Failed",l,r, b)
								return { (l,r): 0.0 }
		return ratios


	def convert_parameters(pcfg1):
		"""
		Assume pcfg1 has parameters in xi format.
		Convert these to pi
		"""
		expectations = pcfg1.compute_partition_function_fast()
		for prod in pcfg1.productions:
			param = pcfg1.parameters[prod]
			if len(prod) == 2:
				nt,a = prod
				newparam = param/expectations[nt]
			else:
				a,b,c = prod
				newparam = param * expectations[b] * expectations[c] / expectations[a]
			pcfg1.parameters[prod] = newparam
		return pcfg1



