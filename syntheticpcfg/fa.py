# Code for interescting a PCFG with a FA
from collections import defaultdict
import pcfg

class FiniteAutomaton:


	def __init__(self):
		# (i,a,j) tuples
		self.transitions = set()
		self.start = None
		self.end = None
		self.states = set()
		self.alphabet = set()


	def is_deterministic(self):
		nextset = set()
		for i,a,j in self.transitions:
			if (i,a) in nextset:
				return False
			else:
				nextset.add(i,a)
		return True

	def intersect(self, old_pcfg):
		"""
		Intersect and return a grammar in CFG normalform.
		"""
		transition_index = defaultdict(list)
		transition_index2 = defaultdict(list)
		for i,a,j in self.transitions:
			transition_index[a].append((i,a,j))
			transition_index2[i].append((i,a,j))
		new_states = set()
		new_productions = set()
		new_parameters = {}
		result = pcfg.PCFG()

		def get_nt(q0, a, q1):
			state = (q0,a,q1)
			new_states.add(state)
			return state
		new_start = get_nt(self.start, old_pcfg.start, self.end)
		
		for prod in old_pcfg.productions:
			alpha = old_pcfg.parameters[prod]
			if len(prod) == 2:
				A,a = prod
				for i,a,j in transition_index[a]:
					nt = get_nt(i,A,j)
					new_productions.add((nt,a))
					new_parameters[(nt,a)] = alpha
			if len(prod) == 3:
				A,B,C = prod
				for i,a,j in self.transitions:
					for j,b,k in transition_index2[j]:
						ntA = get_nt(i,A,k)
						ntB = get_nt(i,B,j)
						ntC = get_nt(j,C,k)
						new_productions.add( (ntA,ntB,ntC))
						new_parameters[(ntA,ntB,ntC)] = alpha
		result.start = new_start
		result.nonterminals = new_states
		result.terminals = set(old_pcfg.terminals)
		result.productions = new_productions
		result.parameters = new_parameters
		return result

def make_prefix(a, terminals):
	my_fa = FiniteAutomaton()
	my_fa.states.add(0)
	my_fa.states.add(1)
	my_fa.start = 0
	my_fa.end = 1
	my_fa.transitions.add( (0,a,1))
	for b in terminals:
		my_fa.transitions.add( (1,b,1))
	return my_fa



def make_ngram( sequence, terminals):
	"""
	deterministic.
	"""
	my_fa = FiniteAutomaton()
	n = length(sequence)
	my_fa.states.add(0)
	my_fa.start = 0
	for i,a in enumerate(sequence):
		my_fa.states.add(i+1)
		my_fa.transitions.add( (i,a,i+1))
		for b in terminals:
			if a != b:
				my_fa.transitions.add((i,b,0))
	my_fa.end=n
	for a in terminals:
		my_fa.transitions.add((n,a,n))









