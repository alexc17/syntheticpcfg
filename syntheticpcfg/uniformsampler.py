# Uniform sampler for a CNF grammar.

import collections
import numpy as np
import random


import cfg 
from utility import collect_yield

class UniformSampler:
	"""
	Stores data structures that contain counts and indices.
	"""

	def __init__(self, grammar, max_length):
		#print "Initialising uniform sampler"
		# map from nonterminals to vectors of the number of derivations of each length.
		self.index = {}
		self.grammar = grammar
		self.L = max_length + 1
		self.vocab = len(grammar.terminals)
		
		
		self.productions = [ prod for prod in grammar.productions if len(prod) == 3]
		
		
		self.prodindex = collections.defaultdict(list)
		for prod in self.productions:
			self.prodindex[prod[0]].append(prod)
		
		for nt in grammar.nonterminals:
			self.index[nt] = np.zeros(self.L)
		for prod in grammar.productions:
			self.index[prod] = np.zeros(self.L)
			if len(prod) == 2:
				self.index[prod][1] = 1
				self.index[prod[0]][1] += 1
		for i in range(2,self.L):
			for prod in self.productions:
				lhs,a,b = prod
				increment = 0
				for left in range(length):
					right = length - left
					increment += self.index[a][left] * self.index[b][right]
				self.index[prod][length] = increment
				self.index[lhs][length] += increment
		

	def density(self, length):
		""" 
		return the ratio between the number of derivations and
		the number of strings of the specified length. 
		If this is greater than 1 then there are more derivations than strings.
		
		Throws exception if grammar is empty.
		"""
		derivations = self.get_total(length)
		strings = self.vocab ** length
		return derivations/strings


	def string_density(self,length, samples):
		"""
		return an estimate of the proportion of strings of length n that are in the grammar.
		Do this by sampling uniformly from the derivations, 
		and computing the number of derivations for each such string, and dividing.
		"""
		derivations = self.get_total(length)

		strings = 1.0 * self.vocab ** length
		total = 0.0
		parser = FIXME
		inverse = 0.0
		for i in range(samples):

			tree = self.sample(length)
			w = collect_yield(tree)
			#print w
			#print w
			
			n = parser.count(w)
			#print n
			if n == 0:
				tree.dump()
				print "Bad number of parses", n
				raise ValueError(w)
			if n < 0:
				print "infinite number of trees for ", w
				self.grammar.dump()
				raise ValueError(w)
			total += n
			inverse += 1.0/n
		# mean derivations per string.
		mean = total / samples
		# Imean is thus 1/ the avergae number of derivations per string.
		imean = inverse /samples
		print "Alternative estimate of strings ", imean * derivations/strings, (derivations / strings) /  mean
		if mean <= 0.0:
			raise ValueError(mean)
	 	# the mean number of derivations per string.
	 	print "length ", length, "derivations ", derivations, "mean ", mean, " possible strings ", strings, "imean", imean
		return (derivations / strings) *  imean, derivations, strings, 1.0/imean


	def get_total(self,length):
		return self.index[self.grammar.start][length]


	def _get_shortest_length(self, nonterminal):
		for i in range(self.L):
			if self.index[nonterminal][i] > 0:
				return i
		return -1

	def get_shortest_yields(self):
		shortest_yield = {}}
		for nt in self.nonterminals:
			l = self.get_shortest_length(nonterminal)
			if l == -1:
				raise ValueError("No short yields")
			else:
				shortest_yield[nt] = self.sample_from_nonterminal(nonterminal,l).collectYield()
		return shortest_yield

	def get_a_shortest_yield(self,nonterminal):
		self._set_shortest_yields()
		return self.shortest_yield[nonterminal]

	def get(self, nonterminal, length):
		"""
		return how many derivations there are from this nonterminal  
		that generate a given length.
		"""
		return self.index[nonterminal][length]	
	



	def sample(self, length):
		"""
		Sample a tree that generates a string of this length,
		uniformly from all trees that generate a string of this length.
		"""
		# pick a start symbol.
		if self.index[self.grammar.start][length] == 0.0:
			raise ValueError("Sampling from empty")
		return self.sample_from_nonterminal(self.grammar.start,length)
		
	def sample_from_nonterminal(self, nonterminal, length):
		"""
		return a tree with length n, root is nonterminal.
		Pick a production and then sample from that.
		"""
		#print "sampling from ", nonterminal, "length ", length
		if nonterminal in self.grammar.terminals:
			if length == 1:
				return nonterminal
			else:
				raise ValueError("terminals are of length 1")
		total = self.index[nonterminal][length]
		if total == 0.0:
			raise ValueError("Sampling from empty " + nonterminal + " " + str(length))
		r = random.randrange(total)
		for prod in self.prodindex[nonterminal]:
			r -= self.index[prod][length]
			if r < 0.0:
				return self.sample_from_production(prod,length)
		#print "Residue %f" % r
		raise ValueError("normalisation error?" + str(r))


	
	def sample_from_production(self, production, length):
		"""
		return a tree that starts with this production.
		"""
		#print "sample prod", production, " length ", length
		if len(production) == 2:
			assert(length == 1)
			return production[1]

		lhs,one,two = production

		total = self.index[production][length]
		if total == 0.0:
			raise ValueError("Sampling from empty")
		r = random.randrange(total)
		for i in range(length ):
			j = length - i
			value = self.index[one][i] * self.index[two][j] 
			r -= value
			if r < 0:
				return (lhs, self.sample_from_nonterminal(one,i),self.sample_from_nonterminal(two,j))
		raise ValueError("normalisation error?")



	# def generateAll(self, length):
	# 	"""
	# 	generate all trees of a given width. Returns a list. Useful when there are only very few.
	# 	"""
	# 	result = []
	# 	for s in self.grammar.start_set:
	# 		result.extend(generateAllFrom(s,length))
	# 	return result

	# def generateAllFrom(self, nonterminal, length):
	# 	result = []
	# 	for prod in self.prodindex[nonterminal]:
	# 		result.extend(generateAllFromProduction(prod,length))
	# 	return result

	# def generateAllFromProduction(self, production, length):
	# 	rhs = production[1]
	# 	lhs = production[0]
	# 	l = len(rhs)
	# 	result = []
	# 	if l == 0:
	# 		result.append(tree.TreeNode(lhs))
	# 	if l == 1:
	# 		subtrees = self.generateAllFrom(rhs[0],length)
	# 		for subtreet in subtrees:
	# 			root = tree.TreeNode(lhs)
	# 			result.daughters.append(subtree)
	# 	if l == 2:
	# 		one = rhs[0]
	# 		two = rhs[1]
	# 		for i in xrange(length + 1):
	# 			value = self.index[one][i] * self.index[two][length -i] 
	# 			for subtree1 in self.generateAllFrom(one,i):
	# 				for subtree2 in self.generateAllFrom(two,length-i):
	# 					root = tree.TreeNode(lhs)
	# 					root.daughters.append(subtree1)
	# 					root.daughters.append(subtree2)
	# 					results.append(self.flatten(root))
	# 	return result


# class ContextSampler:
# 	"""
# 	Class for sampling from the contexts of nonterminals.
# 	Maintains an index from each nonterminal A
# 	of the number of derivation  of width w of S => lAr
# 	where |l| + |r| = w.

# 	Recursion:
# 	S has 1 derivation of length 0

# 	to compute derivations of length w of A
# 	look at each production that A is on the rhs of 
# 	B -> A X  
# 	then for each split of w into i+j 
# 	multiple contexts of length i of B times yields of X of length j
# 	and sum.

# 	Binarisation: 
# 	We need to binarise. So take the topological sort of the nonterminals. 
# 	Then binarise the productions, and get a sort of all nonterminal including the binarised ones.
# 	Then make an index of each of the bproductions.

# 	"""
# 	def __init__(self, grammar, uniformsampler, max_length):
# 		self.grammar = grammar
# 		self.us = uniformsampler
# 		self.max_length = max_length
# 		# index is a map from nonterminals to lists of floats
# 		# index[nt][l] gives number of derivations

# 		self.index = dict()
# 		# index for each nonterminal and rprod of a shortest context
# 		self.shortest_context = False
# 		# remember to binarise productions
# 		# remember to topologically sort the nonterminals
# 		# so the 
# 		self.eprodlist = []
# 		sortednts = grammar.topological_sort()
# 		# prodmap an index from occurrences of nonter
# 		prodmap = collections.defaultdict(list)
# 		for prod in grammar.productions:
# 			prodmap[prod[0]].append(prod)
# 		sortedntsfull = []
# 		rprodmap = collections.defaultdict(list)
# 		for scc in reversed(sortednts):
# 			if len(scc) > 1:
# 				raise ValueError("Unary cycle")
# 			nt = scc[0]
# 			sortedntsfull.append(nt)
# 			for prod in prodmap[nt]:
# 				for bprod in cfg.binarise_production(prod):
# 					lhs = bprod[0]
# 					rhs = bprod[1]
# 					if lhs != nt:
# 						sortedntsfull.append(lhs)
# 					for i,rhsnt in enumerate(rhs):
# 						rprod = (rhsnt, bprod,i)
# 						rprodmap[rhsnt].append(rprod)
# 						self.index[rprod] = [0.0] * (max_length+1)
# 		rprodlist = []
# 		for nt in sortedntsfull:
# 			self.index[nt] = [0.0] * (max_length+1)
# 			rprodlist.extend(rprodmap[nt])
# 		##now we have a good list of the rprods that we iterate through
# 		self.rproductions = rprodlist
# 		self.rprodmap = rprodmap
# 		# initialise
# 		for s in self.grammar.start_set:
# 			self.index[s][0] = 1
# 			# recurse
# 		for length in xrange(max_length+1):
# 			self._process(length)

# 	def _process(self, l):
# 		# do the things for length l
# 		for rprod in self.rproductions:
# 			rhsnt,prod, i = rprod
# 			lhs,rhs = prod
# 			assert len(rhs) == 1 or len(rhs) == 2
# 			increment = 0
# 			if len(rhs) == 1:
# 				increment = self.index[lhs][l]
# 			if len(rhs) == 2:
# 				othernt = rhs[1-i]
# 				#  check if other nt is a terminal for efficiency
# 				if othernt in self.grammar.terminals:
# 					# if it is then it only has one derivation and that is of length 1
# 					increment = self.index[lhs][l-1]
# 				else:
# 					for root in xrange(l+1):
# 						other = l - root
# 						# the number of contexts of the parent * the number of derivations from the ither nonterminal
# 						# of length other
# 						increment += self.index[lhs][root] * self.us.index[othernt][other]
# 			self.index[rprod][l] += increment
# 			self.index[rhsnt][l] += increment
	
			
# 	def sample_context(self, nonterminal, length):
# 		total = self.index[nonterminal][length]
# 		if total == 0:
# 			raise ValueError("sampling context where there are none.")
# 		alpha = random.randrange(total)
# 		for rprod in self.rprodmap[nonterminal]:
# 			alpha -= self.index[rprod][length]
# 			if alpha < 0:
# 				context = self.sample_context_rprod(rprod,length)
# 				assert len(context[0]) + len(context[1]) == length
# 				return context
# 		if length == 0 and nonterminal in self.grammar.start_set:
# 			return ((),())
# 		raise ValueError("Normalisation error " + str(alpha))


# 	def _set_shortest_contexts(self):
# 		if self.shortest_context:
# 			return True
# 		else:
# 			self.shortest_context = dict()
# 			for nt in self.grammar.nonterminals:
# 				#print "Computing ", nt
# 				shortest = False
# 				for rprod in self.rprodmap[nt]:
# 					l = self.find_shortest_length(rprod)
# 					c = self.sample_context_rprod(rprod,l)
# 					#print c
# 					self.shortest_context[rprod] = c
# 					if not shortest:
# 						shortest = c
# 					elif len(c[0]) + len(c[1]) < len(shortest[0]) + len(shortest[1]):
# 						shortest = c
# 				if nt in self.grammar.start_set:
# 					shortest = ((),())
# 				assert shortest
# 				self.shortest_context[nt] = shortest

# 	def sample_shortest_context(self, nonterminal):
# 		"""
# 		Return a context of minimal length for a nonterminal or rprod.
# 		"""
# 		self._set_shortest_contexts()
# 		return self.shortest_context[nonterminal]

# 	def find_shortest_length(self, nonterminal):	
# 		"""
# 		for nonterminals and rprods.
# 		"""
# 		for i in xrange(self.max_length):
# 			if self.index[nonterminal][i] > 0:
# 				return i
# 		return -1

# 	def collect_shortest_contexts(self, nonterminal):
# 		result = []
# 		for rprod in self.rprodmap[nonterminal]:
# 			result.append(self.sample_shortest_context(rprod))
# 		return result
		

# 	def sample_context_rprod(self,rprod, length):
# 		#print "sampling context ", rprod, "length ", length
# 		rhsnt,prod, i = rprod
# 		lhs,rhs = prod
# 		if len(rhs) == 1:
# 			return self.sample_context(lhs,length)
# 		else:
# 			othernt = rhs[1-i]
# 			#  check if other nt is a terminal for efficiency
# 			if othernt in self.grammar.terminals:
# 				subtree = (othernt,)
# 				context = self.sample_context(lhs,length-1)
# 				assert len(context[0]) + len(context[1]) == length-1
# 			else:
# 				# we need to sample a length
# 				l = self._pick_length(rprod, length)
# 				subtree = tuple(self.us.sample_from_nonterminal(othernt,length-l).collectYield())
# 				assert len(subtree) == length -l
# 				context = self.sample_context(lhs,l)
# 				assert len(context[0]) + len(context[1]) == l
# 			if i == 0:
# 				return (context[0], subtree + context[1])
# 			else:
# 				return (context[0] + subtree, context[1])

# 	def get(self, nonterminal, length):
# 		"""
# 		return how many context derivations there are from this nonterminal  
# 		that generate a given length.
# 		"""
# 		return self.index[nonterminal][length]	
	

# 	def _pick_length(self, rprod, length):
# 		"""
# 		Pick how to split the context of this.
# 		return the length of the root context
# 		"""
# 		rhsnt,prod, i = rprod
# 		lhs,rhs = prod
# 		total = self.index[rprod][length]
# 		if total == 0:
# 			raise ValueError("sampling context where there are none.")
# 		alpha = random.randrange(total)
# 		for root in xrange(length+1):
# 			otherl = length - root
# 			othernt = rhs[1-i]
# 			increment = self.index[lhs][root] * self.us.index[othernt][otherl]
# 			alpha -= increment
# 			if alpha < 0:
# 				return root
# 		raise ValueError("Normalisation error in pick-length")