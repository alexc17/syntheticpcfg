# pcfgfactory.py
from collections import Counter
from collections import defaultdict
import numpy.random
import math
import numpy as np
import scipy.misc
from scipy.stats import poisson
import logging

import inside
import cfg
import cfgfactory
import pcfg
import utility

LENGTH_EM_ITERATIONS = 5
LENGTH_EM_MAX_LENGTH = 20

# if we get within this ratio of the best possible length distribution then we terminate. 
TERMINATION_KLD = 0.05

class LengthDistribution:

	def __init__(self):
		# Do not use these default weights.
		self.weights = [0.0, 1.0,1.0,1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0]

	def length_kld(self, mypcfg):
		"""
		Compute KLD between the distribution given by the weights and the true distribution.
		"""
		ui = inside.UnaryInside(mypcfg)
		table = ui.compute_inside_smart(len(self.weights))

		kld = 0.0
		total = sum(self.weights)
		for length, w in enumerate(self.weights):
			if w > 0:
				p = w/total
				q = table[length, ui.start]
				kld += p * math.log(p/q)
		return kld

	def cds(self):
		"""
		Kind of weird the hump at n =1, but it is what it is.
		"""
		self.weights = [0, 33457, 18113,  26082,  29916,  25065, 19806,  14680, 10414, 7144, 
			4720,  3049, 2225,  1664, 1217,  819,  414,  185,  84,  54,  32 ]
		
	def wsj(self):
		"""
		Set distribution of lengths to that of the WSJ corpus.
		Only taking the first 50 for efficiency reasons.
		"""
		#self.weights = [0, 137, 282, 309, 434, 504, 630, 739, 929, 992, 1101, 1226, 1508, 1457, 1540, 1529, 1640, 1616, 1714, 1580, 1666, 1599, 1566, 1566, 1392, 1452, 1365, 1245, 1146, 1102, 1031, 881, 768, 696, 581, 548, 483, 441, 384, 352, 263]
		self.weights = [0, 137, 282, 309, 434, 504, 630, 739, 929, 992, 1101, 1226, 1508, 1457, 1540, 1529, 1640, 1616, 1714, 1580, 1666, 1599, 1566, 1566, 1392, 1452, 1365, 1245, 1146, 1102, 1031, 881, 768, 696, 581, 548, 483, 441, 384, 352, 263, 252, 237, 184, 159, 121, 99, 89, 74, 58, 56]
		#[0, 137, 282, 309, 434, 504, 630, 739, 929, 992, 1101, 1226, 1508, 1457, 1540, 1529, 1640, 1616, 1714, 1580, 1666, 1599, 1566, 1566, 1392, 1452, 1365, 1245, 1146, 1102, 1031, 881, 768, 696, 581, 548, 483, 441, 384, 352, 263, 252, 237, 184, 159, 121, 99, 89, 74, 58, 56, 62, 40, 27, 29, 27, 19, 28, 9, 14, 17, 13, 9, 8, 2, 1, 2, 2, 5, 6, 2, 1, 1, 6, 1, 3, 1, 3, 2, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	def ml_lp(self, max_length):
		s = sum(self.weights[:max_length+1])
		lp = 0.0
		for w in self.weights[:max_length+1]:
			if w > 0:
				p = float(w)/s
				lp += w * math.log(p)
		return lp 

	def ml_lp_general(self, max_length):
		s = sum(self.weights)
		lp = 0.0
		for w in self.weights[:max_length+1]:
			if w > 0:
				p = float(w)/s
				lp += w * math.log(p)
		return lp 

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
		return len(self.weights)

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
		total = 0
		for l in range(1 , max_length+1):
			s = tuple([ a for i in range(l)])
			w = self.length_distribution.weights[l]
			#print(l,w)
			if w > 0:
				lp = w * insidec.add_posteriors(s, posteriors, w)
				total += lp
			#print("Posteriors",l,posteriors)
		logging.info("UNARY LP %f", total)
		my_pcfg.parameters = posteriors
		my_pcfg.normalise()
		return my_pcfg,total

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

	def sample(self, ignore_errors=False, viterbi=False):
		"""
		return a PCFG that is well behaved.
		This may not generate strings of all lengths.
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


		# compute best possible LP.
		logging.info("Training with LENGTH_EM_MAX_LENGTH %d ", LENGTH_EM_MAX_LENGTH)
		#return unary_pcfg
		targetlp = self.length_distribution.ml_lp(LENGTH_EM_MAX_LENGTH)
		valid_target = self.length_distribution.ml_lp_general(LENGTH_EM_MAX_LENGTH)
		logging.info("Target LP = %f, %f", targetlp, valid_target)
		ui = inside.UnaryInside(unary_pcfg)
		
		for i in range(LENGTH_EM_ITERATIONS):

			logging.info("Starting EM iteration %d, target = %f", i,targetlp)
			lp,kld = ui.train_once_smart(self.length_distribution.weights, LENGTH_EM_MAX_LENGTH)
			delta = abs(lp - targetlp)
			logging.info("KLD from target %f", kld)
			if kld< TERMINATION_KLD:
				logging.info("Converged enough.  %f < %f ",  kld , TERMINATION_KLD)
				break
		else:
			logging.warning("Reached maximum number of iterations without reaching convergence threshold. Final KLD is %f.",kld)
			
		
		unary_pcfg.parameters = ui.get_params()
		unary_pcfg.trim_zeros()
		final_pcfg = pcfg.PCFG()
		#print("nonterminals", unary_pcfg.nonterminals)
		for nt in lexical_index:

			if (nt,unary) in unary_pcfg.parameters:
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






