# pcfgfactory.py
from collections import Counter
from collections import defaultdict
import numpy.random
import numpy as np
import scipy.misc
from scipy.stats import poisson
import logging

import inside
import cfg
import cfgfactory
import pcfg

LENGTH_EM_ITERATIONS = 5
LENGTH_EM_MAX_LENGTH = 20


class LengthDistribution:

	def __init__(self):
		self.weights = [0.0, 9.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]


	def cds(self):
		"""
		Kind of weird the hump at n =1, but it is what it is.
		"""
		self.weights = [0, 33457, 18113,  26082,  29916,  25065, 19806,  14680, 10414, 7144, 
			4720,  3049, 2225,  1664, 1217,  819,  414,  185,  84,  54,  32 ]
		

 

	def ztnb(self,r=5,p=0.5,max_length=20):
		"""
		Weights are sampled from a zero truncated negative binomial.
		"""
		probs = [0.0] 
		for k in range(1,max_length+1):
			probability = scipy.misc.comb(k + r -1,k) * (p ** k) * ((1.0  -p) ** r)
			probs.append(probability)
		self.weights = probs

	def poisson(self, alpha):
		"""
		Sampled from a poisson
		"""
		probs = [0.0] 
		for k in range(1,max_length+1):
			probability = poisson(k,alpha)
			probs.append(probability)
		self.weights = probs


	def max_length(self):
		return len(scores)

	def from_corpus(self, filename):
		"""
		read the desired dsirtibution of lengths from a corpus.
		"""
		length_counter = Counter()
		with open(filename) as corpus:
			for line in corpus:
				l = len(line.split())
				if l > 0:
					length_counter[l] += 1
		self.max_length = max(list(length_counter))
		self.weights = [ length_counter[i] for i in  range(self.max_length+1)]


class LexicalDirichlet:

	def __init__(self,dirichlet=1.0):
		self.dirichlet = dirichlet

	def sample(self, k):
		#print(k)
		alphs = np.ones(k) * self.dirichlet
		dprobs = numpy.random.dirichlet(alphs, 1)
		return [ dprobs[0,i] for i in range(k)]

import pitmanyor

class LexicalPitmanYor:

	def __init__(self,d=0.9,alpha=1.0):
		"""
		d is discount
		alpha is concentration parameter: alpha close to zero means all concentrated in a single one, 
		alpha greater than 1 means more spread out.
		d is zero gives something close to Dirichlet 
		d = 0.95 or so gives you something close to 
		"""
		self.d = d
		self.alpha = alpha

	def sample(self, k):
		# k,d,alpga
		return pitmanyor.sample2(k, self.d,self.alpha)

class LexicalUniform:

	def sample(self, k):
		return [1.0/k for i in range(k)]

class PCFGFactory:

	def __init__(self):

		self.cfgfactory = cfgfactory.CFGFactory()
		self.length_distribution = LengthDistribution()
		self.lexical_distribution = LexicalDirichlet()

	def train_unary_once(self, my_pcfg, a, max_length):
		posteriors = defaultdict(float)
		max_length = min(max_length, len(self.length_distribution.weights)-1)

		insidec = inside.InsideComputation(my_pcfg)
		for l in range(1 , max_length+1):
			s = tuple([ a for i in range(l)])
			w = self.length_distribution.weights[l]
			if w > 0:
				insidec.add_posteriors(s, posteriors, w)
			#print("Posteriors",l,posteriors)
		my_pcfg.parameters = posteriors
		my_pcfg.normalise()
		return my_pcfg

	def sample_naive(self):
		"""
		Naive approach.
		Sample grammar; sample parameters from uniform Dirichlet.
		renormalise.
		"""
		
		cfg = self.cfgfactory.sample_trim()
		lhsindex = defaultdict(list)
		for prod in cfg.productions:
			lhsindex[prod[0]].append(prod)
		parameters = {}
		ld = LexicalDirichlet()
		for nt in cfg.nonterminals:
			prods = lhsindex[nt]
			k = len(prods)
			params = ld.sample(k)
			for prod,alpha in zip(prods,params):
				parameters[prod] = alpha
		result = pcfg.PCFG(cfg=cfg)
		result.parameters = parameters
		result.renormalise()
		return result

	def sample(self):
		"""
		return a PCFG that is well behaved.
		"""
		cfg = self.cfgfactory.sample_trim()
		## 
		unary = "_unary_"
		# make the 
		unary_pcfg = pcfg.PCFG()
		unary_pcfg.start = cfg.start
		unary_pcfg.terminals.add(unary)
		unary_pcfg.nonterminals = set(cfg.nonterminals)
		productions = set()
		lexical_index = defaultdict(list)
		for prod in cfg.productions:
			if len(prod) == 3:
				productions.add(prod)
			else:
				productions.add( (prod[0], unary))
				lexical_index[prod[0]].append(prod[1])
		unary_pcfg.productions = list(productions)
		unary_pcfg.parameters = { prod : 1.0 for prod in productions}
		unary_pcfg.normalise()

		self.current = unary_pcfg

		#return unary_pcfg
		for i in range(LENGTH_EM_ITERATIONS):
			try:
				unary_pcfg = self.train_unary_once(unary_pcfg, unary, LENGTH_EM_MAX_LENGTH)
			except ValueError as e:
				print("error training lengths")
				print(unary_pcfg.productions)
				raise e
			

		final_pcfg = pcfg.PCFG()
		#print("nonterminals", unary_pcfg.nonterminals)
		for nt in lexical_index:
			totalp = unary_pcfg.parameters[(nt,unary)]
			k = len(lexical_index[nt])
			probs = self.lexical_distribution.sample(k)
			#print(nt,probs,k,totalp,unary)
			assert(len(probs) == k)
			for a,p  in zip(lexical_index[nt],probs):
				prod = (nt,a)

				final_pcfg.productions.append(prod)
				final_pcfg.parameters[prod] = p * totalp
		for prod in unary_pcfg.productions:
			if len(prod) == 3:
				final_pcfg.productions.append(prod)
				final_pcfg.parameters[prod] = unary_pcfg.parameters[prod]
		final_pcfg.start = unary_pcfg.start
		final_pcfg.nonterminals = unary_pcfg.nonterminals
		final_pcfg.terminals = cfg.terminals
		final_pcfg.normalise()
		return final_pcfg






