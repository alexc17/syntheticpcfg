
## Python 3 


import math
import numpy as np
import numpy.linalg
import numpy.random
from collections import defaultdict
## A PCFG in CNF ish
## S can appear on rhs of productions.
## May not be normalised or consistent.

from utility import *
import inside


RIGHT_ARROW = "->"
START_SYMBOL = "S"

class PCFG:

	def __init__(self, cfg=None):
		if cfg:
			self.nonterminals = set(cfg.nonterminals)
			self.terminals = set(cfg.terminals)
			self.start = cfg.start
			self.productions = list(cfg.productions)
		else:
			self.nonterminals = set()
			self.terminals = set()
			## Productions are tuples (A, a) or (A, B,C)
			self.start = None
			self.productions = []
		self.parameters = {}
		self.log_parameters = {}
		
	def store(self, filename):
		"""
		Store this to a file.
		"""
		with open(filename,'w') as fhandle:
			for prod in self.productions:
				p = self.parameters[prod]
				if len(prod) == 2:	
					fhandle.write( "%0.12f %s %s %s \n" % ( p, prod[0], RIGHT_ARROW,  prod[1] ))
				else:
					fhandle.write( "%0.12f %s %s %s %s \n" % ( p, prod[0], RIGHT_ARROW, prod[1],prod[2]))

	def store_mjio(self, filename):
		"""
		Store this in a format suitable for use by MJ IO code.
		So introduce preterminals for all nonterminals. 
		Assume in CNF.
		"""
		with open(filename,'w') as fhandle:
			fhandle.write("1.0 S1 --> S \n")
			preterminals = { nt : ("PRE" + nt) for nt in self.nonterminals}
			## now add up probs of preterminals
			preterminal_probs = { nt : 0.0 for nt in self.nonterminals}
			for prod in self.productions:
				if len(prod) == 2:
					preterminal_probs[prod[0]] += self.parameters[prod]
			for nt in self.nonterminals:
				fhandle.write( "%0.12f %s --> %s \n" % ( preterminal_probs[nt], nt, preterminals[nt] ))
			for prod in self.productions:
				if len(prod) == 2:
					# preterminal
					p = self.parameters[prod] / preterminal_probs[prod[0]]
					fhandle.write( "%0.12f %s --> %s \n" % ( p, preterminals[prod[0]], prod[1]))
				else:
					# binary rule so 
					p = self.parameters[prod] / (1.0 - preterminal_probs[prod.lhs])
					fhandle.write( "%0.12f %s --> %s %s \n" % ( p, prod[0], prod[1],prod[2]))
			

	def normalise(self):
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		for prod in self.productions:
			p = self.parameters[prod]
			if p == 0.0:
				raise ValueError("zero parameter",prod)
			self.parameters[prod] = p/ totals[prod[0]]
			self.log_parameters[prod] = math.log(p/ totals[prod[0]])



	def log_probability_derivation(self, tree):
		"""
		Compute the log prob of a derivation.
		"""
		if len(tree) == 2:
			# lexical
			return self.log_parameters[tree]
		else:
			left_lp = self.log_probability_derivation(tree[1])
			right_lp = self.log_probability_derivation(tree[2])
			local_tree = (tree[0], tree[1][0], tree[2][0])
			local_lp = self.log_parameters[local_tree]
			return left_lp + right_lp + local_lp

	
	## Various properties of the PCFG that may be useful to compute.

	def entropy_nonterminals(self):
		# entropy of the productions.
		e = defaultdict(float)
		for prod in self.productions:
			p = self.parameters[prod]
			if p > 0:
				e[prod[0]] -= p * math.log(p)
		return e

	def derivational_entropy(self):
		"""
		entropy of the distribution over derivation trees.
		"""
		nt_entropies = self.entropy_nonterminals()
		nt_expectations = self.nonterminal_expectations()
		return sum([ nt_entropies[nt] * nt_expectations[nt] for nt in self.nonterminals])
 
	def monte_carlo_entropy(self, n):
		"""
		Use a Monte Carlo approximation; return string entropy, unlabeled entropy and derivation entropy.
		"""
		string_entropy=0
		unlabeled_tree_entropy = 0
		labeled_tree_entropy = 0
		sampler = Sampler(self)
		insidec = inside.InsideComputation(self)
		for i in range(n):
			tree = sampler.sample_tree()
			lp1 = self.log_probability_derivation(tree)
			sentence = collect_yield(tree)
			lp2 = insidec.inside_bracketed_log_probability(tree)
			lp3 = insidec.inside_log_probability(sentence)
			string_entropy -= lp1
			unlabeled_tree_entropy -= lp2
			labeled_tree_entropy -= lp3
		return string_entropy/n, unlabeled_tree_entropy/n, labeled_tree_entropy/n

	def partition_nonterminals(self):
		"""
		Partition the sets of nonterminals into sets of mutually recursive nonterminals.
		"""
		graph = defaultdict(list)
		for prod in self.productions:
			if len(prod) == 3:
				for i in [1,2]:
					graph[prod[0]].append(prod[i])
		return strongly_connected_components(graph)

	def renormalise(self):
		"""
		renormalise so that it is consistent.
		destructuve
		"""
		rn = self.compute_partition_function_fast()
		for prod in self.productions:

			if len(prod) == 3:
				a,b,c = prod
				self.parameters[prod] *=  (rn[b] * rn[c])/rn[a] 
			else:
				self.parameters[prod] *= 1.0/rn[prod[0]]

		self.normalise()
		




	def compute_partition_function_fast(self):
		"""
		Solve the quadratic equations using Newton method.
		"""
		ntindex = { nt:i for i,nt in enumerate(list(self.nonterminals))}
		n = len(ntindex)
		alpha = defaultdict(float)
		beta = np.zeros(n)
		for prod in self.productions:
			p = self.parameters[prod]
			if len(prod) == 2:
				beta[ ntindex[prod[0]]] += p
			else:

				alpha[ (ntindex[prod[0]],ntindex[prod[1]],ntindex[prod[2]])]+= p
		x = np.zeros(n)

		def f(y):
			## evaluate f at this point.
			fy = beta - y
			for i,j,k in alpha:
				a = alpha[(i,j,k)]
				#i is lhs
				fy[i] += a * y[j] * y[k]
			return fy

		def J(y):
			# evalate Jacobian
			J = -1 * np.eye(n)
			for i,j,k in alpha:
				a = alpha[(i,j,k)]
				J[i,j] += a * y[k]
				J[i,k] += a * y[j]
			return J

		epsilon = 1e-9
		max_iterations = 100
		for i in range(max_iterations):
			#print(x)
			y = f(x)
			#print(y)
			x1 =  x - np.dot(np.linalg.inv(J(x)),y)
			if numpy.linalg.norm(x - x1, 1) < epsilon:
				return { nt : x[ntindex[nt]] for nt in self.nonterminals}
			x = x1
		raise ValueError("failed to converge")
		




	def compute_partition_function_fp(self):
		"""
		Return a dict mapping each nonterminal to the prob that a string terminates from that.
		Use the naive fixed point algorithm.
		"""
		bprods = [ prod for prod in self.productions if len(prod) == 3]
		lprodmap = defaultdict(float)
		for prod in self.productions:
			if len(prod) == 2:
				lprodmap[prod[0]] += self.parameters[prod]
		z = defaultdict(float)
		for i in range(100):
			z = self._compute_one_step_partition(z, bprods,lprodmap)
		return z


	def production_expectations(self):
		nte = self.nonterminal_expectations()
		return { prod: (self.parameters[prod] * nte[prod[0]]) for prod in self.productions }


	def terminal_expectations(self):
		answer = defaultdict(float)
		pe = self.production_expectations()
		for prod in pe:
			if len(prod) == 2:				
				alpha = pe[prod]
				answer[prod[1]] += alpha
		return answer


	def expected_length(self):
		pe = self.production_expectations()
		return sum([ pe[prod] for prod in pe if len(prod) == 2 ])

	def nonterminal_expectations(self):
		"""
		Compute the expected number of times each nonterminal will be used in a given derivation.
		return a dict mapping nonterminals to non-negative reals.
		"""
		n = len(self.nonterminals)
		transitionMatrix = np.zeros([n,n])
		#outputMatrix = np.zeros(n)
		ntlist = list(self.nonterminals)
		#print(ntlist)
		index = { nt:i for i,nt in enumerate(ntlist)}
		for prod in self.productions:
			alpha = self.parameters[prod]
			lhs = index[prod[0]]
			if len(prod) == 3:
				transitionMatrix[lhs,index[prod[1]]] += alpha
				transitionMatrix[lhs,index[prod[2]]] += alpha
		
		#print(transitionMatrix)
		r2 = numpy.linalg.inv(np.eye(n) - transitionMatrix)
		#print(r2)
		result = np.dot(numpy.linalg.inv(np.eye(n) - transitionMatrix),transitionMatrix)
		si = index[self.start]
		resultD = { nt : result[si, index[nt]] for nt in self.nonterminals}
		resultD[self.start] += 1
		return resultD

	def expected_lengths(self):
		"""
		Compute the expected length of a string generated by each nonterminal.
		Assume that the grammar is consistent.
		"""
		n = len(self.nonterminals)
		m = len(self.terminals)
		ntlist = list(self.nonterminals)
		transitionMatrix = np.zeros([n,n])
		outputMatrix = np.zeros([n,m])
		index = {nt : i for i,nt in enumerate(ntlist)}
		terminalIndex = { a: i for i,a in enumerate(list(self.terminals))}
		# each element stores the expected number of times that 
		# nonterminal will generate another nonterminal in a single derivation.
		for prod in self.productions:
			alpha = self.parameters[prod]
			lhs = index[prod[0]]
			if len(prod) == 2:
				outputMatrix[lhs,terminalIndex[prod[1]]] += alpha
			else:
				transitionMatrix[lhs,index[prod[1]]] += alpha
				transitionMatrix[lhs,index[prod[2]]] += alpha

		
		# n = o * (1- t)^-1
		result = np.dot(numpy.linalg.inv(np.eye(n) - transitionMatrix),outputMatrix)
		return result
		
	def _compute_one_step_partition(self, z, bprods,lprodmap):

		newz = defaultdict(float) 
		for production in bprods:
			score = self.parameters[production] * z[production[1]] * z[production[2]]
			newz[production[0]] += score
		for nt in lprodmap:
			newz[nt] += lprodmap[nt]
		return newz

def load_pcfg_from_file(filename, normalise=True, discard_zero=True):
	nonterminals = set()
	## Nonterminals are things that appear on the lhs of a production.
	## Start symbol is S
	## Um thats it ?
	prods = []
	with open(filename) as infile:
		for line in infile:
			if line.startswith("#"):
				continue
			tks = line.split()
			if len(tks) == 0:
				continue
			assert len(tks) >= 4
			assert tks[2] == RIGHT_ARROW
			p = float(tks[0])
			assert p >= 0
			lhs = tks[1]
			nonterminals.add(lhs)

			if p == 0.0 and discard_zero:
				continue
			
			rhs = tuple(tks[3:])
			prods.append( (p,lhs,rhs))
	assert START_SYMBOL in nonterminals

	terminals = set()
	totals = { nt: 0.0 for nt in nonterminals }
	for p,lhs,rhs in prods:
		totals[lhs] += p
		for s in rhs:
			if not s in nonterminals:
				terminals.add(s)
	my_pcfg = PCFG()
	for p,lhs,rhs in prods:
		prod = (lhs,) + rhs
		my_pcfg.productions.append(prod)
		if normalise:
			p /= totals[lhs]
		my_pcfg.parameters[prod] = p
		if discard_zero:
			my_pcfg.log_parameters[prod] = math.log(p)
		
	my_pcfg.start =  START_SYMBOL
	my_pcfg.terminals = terminals
	my_pcfg.nonterminals = nonterminals
	return my_pcfg


class Multinomial:
	"""
	this represents a collection of productions with the same left hand side.
	"""

	def __init__(self, pcfg, nonterminal, cache_size=1000):
		self.cache_size = cache_size
		self.productions = [ prod for prod in pcfg.productions if prod[0] == nonterminal ]
		self.n = len(self.productions)
		self.nonterminal = nonterminal
		parameters = [ pcfg.parameters[prod] for prod in self.productions]
		self.p = np.array(parameters)/np.sum(parameters)
		self._sample()

	def _sample(self):
		self.cache = numpy.random.choice(range(self.n), size = self.cache_size,p = self.p)
		self.cache_index = 0

	def sample_production(self):
		"""
		return the rhs of a production as a tuple of length 1 or 2
		"""
		if self.cache_index >= self.cache_size:
			self._sample()
		result = self.productions[self.cache[self.cache_index]]
		self.cache_index += 1
		return result



class Sampler:
	"""
	This object is used to sample from a PCFG.
	"""
	def __init__(self, pcfg,cache_size=1000,max_depth = 1000):
		## construct indices for sampling
		self.multinomials = { nt: Multinomial(pcfg,nt, cache_size) for nt in pcfg.nonterminals}
		self.start = pcfg.start
		self.max_depth = max_depth


	def sample_production(self, lhs):
		return self.multinomials[lhs].sample_production()

	def sample_tree(self):
		return self._sample_tree(self.start, 0)

	def _sample_tree(self, nonterminal, current_depth):
		"""
		Sample a single tree from the pcfg, and throw an exception of max_depth is exceeded.
		"""
		if current_depth >= self.max_depth:
			raise ValueError("Too deep")

		prod = self.sample_production(nonterminal)
		if len(prod) == 2:
			# lexical rule
			return prod
		else:
			left_branch = self._sample_tree(prod[1],current_depth + 1)
			right_branch = self._sample_tree(prod[2],current_depth + 1)
			return (nonterminal, left_branch,right_branch)


