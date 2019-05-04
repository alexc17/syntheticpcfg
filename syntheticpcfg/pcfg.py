
## Python 3 

import logging
import math
import numpy as np
import numpy.linalg
import numpy.random
from collections import defaultdict
from collections import Counter

from utility import *
import inside


RIGHT_ARROW = "->"
START_SYMBOL = "S"
UNARY_SYMBOL = "<A>"

SAMPLE_MAX_DEPTH=100
SAMPLE_CACHE_SIZE=1000
PARTITION_FUNCTION_MAX_ITERATIONS=100
PARTITION_FUNCTION_EPSILON=1e-9


class PCFG:
	"""
	This class stores a PCFG where the underlying CFG is in CNF (more or less).
	"""
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
	

	def make_unary(self):
		""" 
		return a new grammar which has the same distribution over lengths and only one terminal symbol.
		"""
		upcfg = PCFG()
		upcfg.terminals.add(UNARY_SYMBOL)
		upcfg.nonterminals = set(self.nonterminals)
		upcfg.start = self.start
		
		for prod in self.productions:
			p = self.parameters[prod]
			if len(prod) == 3:
				# binary
				upcfg.productions.append(prod)
				upcfg.parameters[prod] = p
			else:
				# lexical
				nt,a = prod
				newprod = (nt,UNARY_SYMBOL)
				if newprod in upcfg.parameters:
					upcfg.parameters[newprod] += p 
				else:
					upcfg.productions.append(newprod)
					upcfg.parameters[newprod] = p
		# Why normalise?
		upcfg.set_log_parameters()
		return upcfg

	def store(self, filename,header=[]):
		"""
		Store this to a file.
		"""
		with open(filename,'w') as fhandle:
			if len(header) > 0:
				for line in header:
					fhandle.write('#' + line + "\n")
			for prod in self.productions:
				p = self.parameters[prod]
				if len(prod) == 2:	
					fhandle.write( "%e %s %s %s \n" % ( p, prod[0], RIGHT_ARROW,  prod[1] ))
				else:
					fhandle.write( "%e %s %s %s %s \n" % ( p, prod[0], RIGHT_ARROW, prod[1],prod[2]))

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
	
	def rhs_index(self, terminal):
		"""
		Utility function that returns a list of all nonterminals that have a production
		with this terminal on its right hand side.
		"""
		return [ prod[0] for prod in self.productions if terminal in prod]

	def copy(self):
		"""
		return a new copy of this pcfg.
		"""
		copypcfg = PCFG()
		copypcfg.nonterminals = set(self.nonterminals)
		copypcfg.terminals = set(self.terminals)
		copypcfg.start = self.start
		copypcfg.productions = list(self.productions)
		copypcfg.parameters = dict(self.parameters)
		copypcfg.log_parameters = dict(self.log_parameters)
		return copypcfg

	def trim_zeros(self, threshold = 0.0):
		"""
		destructively remove all zero productions, zero nonterminals and zero terminals.
		"""
		self.productions = [ prod for prod in self.productions if self.parameters[prod] > threshold]
		self.nonterminals = set( [ prod[0] for prod in self.productions])
		self.terminals = set( [ prod[1] for prod in self.productions if len(prod) == 2])
		self.parameters = { prod : self.parameters[prod] for prod in self.productions}
	
	def set_log_parameters(self):
		self.log_parameters = {}
		for prod in self.productions:
			self.log_parameters[prod] = math.log(self.parameters[prod])

	def normalise(self):
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		for prod in self.productions:
			p = self.parameters[prod]
			if p == 0.0:
				raise ValueError("zero parameter",prod)
			param = p/ totals[prod[0]]
			self.parameters[prod] = param
			self.log_parameters[prod] = math.log(param)

	def check_normalisation(self):
		totals = defaultdict(float)
		for prod in self.productions:
			totals[prod[0]] += self.parameters[prod]
		return totals

	def is_normalised(self, epsilon= 1e-5):
		totals = self.check_normalisation()
		for a in totals:
			if abs(1.0 - totals[a]) > epsilon:
				return False
		return True

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
	def useful_information(self):
		return [ 'Statistics of the final grammar',
				'Actual expected length %f' % self.expected_length(),
				'Derivational entropy %f' % self.derivational_entropy(),
				'Derivational entropy (split) binary %f lexical %f ' % self.derivational_entropy_split(),
				
				'Lexical Ambiguity (entropy) %f' % self.compute_lexical_ambiguity()
				]

	def per_word_entropies(self):
		"""
		dict mapping each word to the entropy of the posterior distribution of nonterminals given 
		terminals.
		"""
		result = defaultdict(float)
		te = self.terminal_expectations()
		pe = self.production_expectations()
		for prod, e in pe.items():
			if len(prod) == 2:
				a = prod[1]
				# posterior 
				p = e/te[a]
				result[a] -= p * math.log(p)
		return result

	def entropy_unigram(self):
		"""
		Entropy of the unigram distribution: compute exactly.
		"""
		te = self.terminal_expectations()
		L = self.expected_length()
		e = 0.0
		for a in te.values():
			p = a/L
			e -= p * math.log(p)
		return e

	def entropy_preterminal(self):
		"""
		Entropy of the preterminal distribution.
		"""
		prode = self.production_expectations()
		preterminalexpectations = defaultdict(float)
		for prod,e in prode.items():
			if len(prod) == 2:
				preterminalexpectations[prod[0]] += e
		l = self.expected_length()
		e = 0.0
		for a in preterminalexpectations.values():
			p = a/l
			e -= p * math.log(p)
		return e

	def entropy_conditional_nonterminals(self):
		# entropy of the conditional productions.
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
		nt_entropies = self.entropy_conditional_nonterminals()
		nt_expectations = self.nonterminal_expectations()
		return sum([ nt_entropies[nt] * nt_expectations[nt] for nt in self.nonterminals])
 
	def derivational_entropy_split(self):
		"""
		Retun binary and lexical entropies which sum to the derivational entropy.
		"""
		lexical_totals = self.sum_lexical_probs()
		
		bentropies = 0.0
		lentropies = 0.0
		nt_expectations = self.nonterminal_expectations()
		for nt, p in lexical_totals.items():
			bentropies -= nt_expectations[nt] *p  * math.log(p)
		for prod, alpha in self.parameters.items():
			if len(prod) == 2:
				p = alpha / lexical_totals[prod[0]]
				lentropies -= nt_expectations[prod[0]] * alpha * math.log(p)
			else:
				bentropies -= nt_expectations[prod[0]] * alpha * math.log(alpha)
		return (bentropies,lentropies)

	def sum_lexical_probs(self):
		sums = defaultdict(float)
		for prod, alpha in self.parameters.items():
			if len(prod) == 2:
				sums[prod[0]] += alpha
		return sums


	def monte_carlo_entropy(self, n, sampler = None):
		"""
		Use a Monte Carlo approximation; return string entropy, unlabeled entropy and derivation entropy.
		"""
		string_entropy=0
		unlabeled_tree_entropy = 0
		labeled_tree_entropy = 0
		if sampler == None:
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


	def compute_lexical_ambiguity(self):
		"""
		Conditional entropy in nats of the preterminal given the terminal.
		Note that even if the grammar is unambiguous, this can be greater than zero.

		"""
		l = self.expected_length()
		te = self.terminal_expectations()
		pe = self.production_expectations()
		ce = 0.0
		for prod, e in pe.items():
			if len(prod) ==2:
				le = te[prod[1]]
				ce -= (e/l) * math.log( e/le)
		return ce

	def estimate_ambiguity(self, samples = 1000, max_length=100, sampler = None):
		"""
		Monte Carlo estimate of the conditional entropy H(tree|string)
		"""
		if sampler==None:
			mysampler = Sampler(self)
		else:
			mysampler = sampler
		insider = inside.InsideComputation(self)
		total = 0.0
		n = 0.0
		for i in range(samples):
			tree = mysampler.sample_tree()
			s = collect_yield(tree)
			if len(s) > max_length:
				continue
			lp = insider.inside_log_probability(s)
			lpd = self.log_probability_derivation(tree)
			total += lp - lpd
			n += 1
		return total/n
	
	def estimate_communicability(self, samples = 1000, max_length=100, sampler = None):
		"""
		Returns two estimates of the communicability; The second one is I think better in most cases.
		"""
		if sampler==None:
			mysampler = Sampler(self)
		else:
			mysampler = sampler
		insider = inside.InsideComputation(self)
		same = 0.0
		ratio = 0.0
		n = 0
		for i in range(samples):
			t = sampler.sample_tree()
			s =collect_yield(t)
			if len(s) <= max_length:
				n += 1
				mapt = insider.viterbi_parse(s)
				if t == mapt:
					same += 1
				lpd = self.log_probability_derivation(mapt)
				lps = insider.inside_log_probability(s)
				ratio += math.exp(lpd - lps)
		return ( same/n, ratio/n)

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

		for i in range(PARTITION_FUNCTION_MAX_ITERATIONS):
			#print(x)
			y = f(x)
			#print(y)
			x1 =  x - np.dot(np.linalg.inv(J(x)),y)
			if numpy.linalg.norm(x - x1, 1) < PARTITION_FUNCTION_EPSILON:
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
		for i in range(PARTITION_FUNCTION_MAX_ITERATIONS):
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
		#r2 = numpy.linalg.inv(np.eye(n) - transitionMatrix)
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

	def oracle_fkp1(self):
		"""
		Return a list of terminals that characterise the nonterminals,
		picking the most likely one.

		Start with "S"
		"""
		lhs_counter = defaultdict(list)
		answer = []
		for prod in self.productions:
			if len(prod) == 2:
				lhs_counter[prod[1]].append(prod[0])
		nts = [ self.start ]
		for nt in self.nonterminals:
			if nt != self.start:
				nts.append(nt)
		for nt in nts:
			candidates = [ a for a in self.terminals if lhs_counter[a] == [nt] ]
			if len(candidates) == 0:
				return ()
			else:
				answer.append( max(candidates, key = lambda a : self.parameters[ (nt,a)]))
		return answer

	def oracle_kernel(self):
		"""
		Return a map from nonterminals to anchors.
		picking the most likely one.
		Return  empty map if it does not satisfy the condition.
		"""
		lhs_counter = defaultdict(list)
		answer = {}
		for prod in self.productions:
			if len(prod) == 2:
				lhs_counter[prod[1]].append(prod[0])
		
		for nt in self.nonterminals:
			candidates = [ a for a in self.terminals if lhs_counter[a] == [nt] ]
			if len(candidates) == 0:
				return {}
			else:
				answer[nt] =  max(candidates, key = lambda a : self.parameters[ (nt,a)])
		return answer

	def renormalise_convergent_wcfg(self):
		"""
		Scale all of the productions with "S" to get a globally normalised pcfg.
		"""
		scale = self.compute_partition_function_fast()[self.start]
		for prod in self.parameters:
			if prod[0]== self.start:
				self.parameters[prod] /= scale
		self.set_log_parameters()

	def renormalise_divergent_wcfg(self):
		"""
		Algorithm from Smith and Johnson (1999) 
		Weighted and Probabilistic Context-Free
		Grammars Are Equally Expressive.

		This gives us a wcfg which is convergent, then we can convert to a PCFG.
		
		"""
		sigma = len(self.terminals)
		beta = max ( [ self.parameters[prod] for prod in self.productions if len(prod) == 3])
		nu = max ( [ self.parameters[prod] for prod in self.productions if len(prod) == 2])

		factor = 1.0 / ( 8 * sigma * beta * nu)

		pcfg1 = self.copy()
		for prod in self.productions:
			if len(prod) == 2:
				pcfg1.parameters[prod] *= factor
		pcfg1.set_log_parameters()
		return pcfg1

	def convert_parameters_pi2xi(self):
		nte = self.nonterminal_expectations()
		xi = {}
		for prod in self.productions:
			if len(prod) == 3:
				a,b,c = prod
				param = self.parameters[prod]
				xib = param * nte[a]/ (nte[b] * nte[c])
				xi[prod] = xib
			else:
				a,b = prod
				param = self.parameters[prod]
				xib = param * nte[a]
			xi[prod] = xib
		xipcfg = self.copy()
		xipcfg.parameters = xi
		xipcfg.set_log_parameters()
		return xipcfg
		
	def convert_parameters_xi2pi(self):
		"""
		Assume pcfg1 has parameters in xi format.
		Convert these to pi
		"""
		pcfg1 = self.copy()
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
		pcfg1.set_log_parameters()
		return pcfg1



def estimate_pcfg_from_treebank(filename):
	"""
	Take ML estimate from a treebank and retuen the pcfg that results.
	"""
	counts = Counter()
	ml = PCFG()
	with open(filename) as inf:
		for line in inf:
			
			tokens = line.split()
			i = 0
			# skip some floats
			while tokens[i].startswith('-') or tokens[i].startswith('0'):
				i += 1

			tree = string_to_tree(" ".join(tokens[i:]))
			ml.start = tree[0]
			count_productions(tree,counts)
	
	for production in counts:
		ml.productions.append(production)
		ml.nonterminals.add(production[0])
		if len(production) == 2:
			ml.terminals.add(production[1])
		ml.parameters[production] = float(counts[production])
	ml.normalise()
	return ml

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

	def __init__(self, pcfg, nonterminal, cache_size=SAMPLE_CACHE_SIZE,random=random):
		self.cache_size = cache_size
		self.productions = [ prod for prod in pcfg.productions if prod[0] == nonterminal ]
		self.n = len(self.productions)
		self.nonterminal = nonterminal
		parameters = [ pcfg.parameters[prod] for prod in self.productions]
		self.p = np.array(parameters)/np.sum(parameters)
		self.rng = random
		self._sample()

	def _sample(self):
		self.cache = self.rng.choice(range(self.n), self.cache_size,True,self.p)
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
	def __init__(self, pcfg,cache_size=SAMPLE_CACHE_SIZE,max_depth = SAMPLE_MAX_DEPTH,random=None):
		## construct indices for sampling
		if random == None:
			random = numpy.random.RandomState()
		assert pcfg.is_normalised()
		self.multinomials = { nt: Multinomial(pcfg,nt, cache_size,random) for nt in pcfg.nonterminals}
		self.start = pcfg.start
		self.max_depth = max_depth
		self.insider = inside.InsideComputation(pcfg)
		self.mypcfg = pcfg

	def sample_production(self, lhs):
		return self.multinomials[lhs].sample_production()

	def sample_tree(self):
		return self._sample_tree(self.start, 0)

	def sample_string(self):
		return collect_yield(self.sample_tree())

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


	def estimate_string_entropy(self,samples,max_length=50,verbose=False):
		"""
		Estimate the string entropy and perplexity.
		"""
		total_length = 0.0
		total_samples = 0
		total_lps = 0.0
		total_lpt = 0.0
		total_lpb = 0.0

		for _ in range(samples):

			tree = self.sample_tree()
			s = collect_yield(tree)
			if True:
				total_samples += 1
				lps = self.insider.inside_log_probability(s)
				lpt = self.mypcfg.log_probability_derivation(tree)
				lpb = self.insider._bracketed_log_probability(tree)[self.start]
				total_length += len(s)
				if verbose: print(s,lpt, lpb,lps)
				total_lps += lps
				total_lpt += lpt
				total_lpb += lpb
		sentential_entropy = -total_lps/total_samples
		perplexity = math.exp(-total_lps/(total_length + total_samples))
		print("SentenceEntropy %f WordPerplexity %f " % (sentential_entropy, perplexity))
