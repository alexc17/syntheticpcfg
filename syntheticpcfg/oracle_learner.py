#oracle_learner.py

import math
import utility
import inside
import pcfg
import logging

from utility import ParseFailureException
from collections import Counter
from collections import defaultdict

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
		# map from kernels to a list of contexts.
		self.features = defaultdict(list)
		## map from productions to the contexts that attain the minima
		self.xi_minima = {}

	def compute_total_expectation(self):
		""" 2 |w| - 1 """
		e = sum([len(s) for s in self.sentences])/len(self.sentences)
		return 2 * e - 1


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
					assert q > 0
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

	def pick_kernels(self, verbose=False):
		F = defaultdict(list)
		K = set()
		for i,a in enumerate(self.terminals):
			f, cp = self.find_predictive_context(a)
			if verbose:
				print(i,a,f,cp)
			logging.info("Most predictive context %d %s %s %f", i,a,f,cp)
			a = self.find_most_predicted_string(*f)
			F[a].append(f)
			K.add(a)
		self.kernels = list(K)
		self.features = F

	def learn(self):
		print("Starting Oracle Learner")
		if  self.te:
			print("Using oracle terminal expectations")
		else:
			print("Estimating terminal expectations")
			self.estimate_terminal_expectations()
		if not self.kernels:
			self.pick_kernels()
		self.start = self.pick_start()
		ntmap = {self.start: "S"}
		for a in self.kernels:
			if a != self.start:
				ntmap[a] = "NT" + a
		
		# if not self.te:
		# 	self.te = estimate_terminal_expectations(sentences)
		parameters = {}
		print("Starting Lexical rules")
		for a in self.kernels:
			for b in self.terminals:
				prod = (ntmap[a],b)
				#xi = min(self.test_unary(a,b).values())
				rr = self.test_unary(a,b)
				xi = min(rr.values())
				for context in rr:
					if rr[context] == xi:
						self.xi_minima[prod] = context
				if xi > 0:
					parameters[prod] =xi
		print("Starting Binary rules")
		for a in self.kernels:
			for b in self.kernels:
				for c in self.kernels:
					prod = (ntmap[a],ntmap[b],ntmap[c])
					rr = self.test_binary(a,b,c)
					xi = min(rr.values())
					for context in rr:
						if rr[context] == xi:
							self.xi_minima[prod] = context
					if xi > 0:
						parameters[prod] =xi

		lpcfg = pcfg.PCFG()
		lpcfg.terminals =self.terminals
		lpcfg.nonterminals = ntmap.values()
		lpcfg.start = "S"
		lpcfg.parameters = parameters
		lpcfg.productions = list(parameters)
		lpcfg.log_parameters = { prod: math.log(parameters[prod]) for prod in parameters}
		self.kernel2nt = ntmap
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
		# cache the calls to the inside algorithm as they can be expensive.
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
		raise ValueError()
		lcounter = Counter()
		n = len(self.sentences)
		for s in self.sentences:
			for a in s:
				lcounter[a] += 1
		self.te =  { a: lcounter[a]/n for a in lcounter}


	def test_unary_f(self, a,b, scale_terminal = False):
		ratios = {}
		if scale_terminal:
			scale =  self.te[a] / self.te[b] 
		else:
			scale =  self.te[a]
		for l,r in self.features[a]:
			denom = self.probability(l + (a,) + r)
			assert denom > 0
			num = self.probability(l + (b,) + r)
			
			ratios[(l,r)] = (num/denom) * scale
			if num == 0:
				break
		return ratios

	def test_binary_f(self, a,b,c):
		ratios = {}
		scale =  self.te[a] / (self.te[b] * self.te[c])
		for l,r in self.features[a]:
			denom = self.probability(l + (a,) + r)
			assert denom > 0
			num = self.probability(l + (b,c) + r)
			ratios[(l,r)] = (num/denom) * scale
			if num == 0:
				break
		return ratios

	def test_binary(self, a,b,c):
		ratios = {}
		te = self.te

		## Probably only need to use the selected 
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
		# We return the map rather than just the minimum as it might be desirable not to take the absolute minimum
		# for reasons of robustness.
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


	



