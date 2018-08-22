import collections

def convert_pcfg_to_cfg(pcfg, discard_zero = True):
	my_cfg = CFG()
	my_cfg.start = pcfg.start
	my_cfg.nonterminals = set(pcfg.nonterminals)
	my_cfg.terminals = set(pcfg.terminals)
	my_cfg.productions = set()
	assert my_cfg.start in my_cfg.nonterminals
	for p in pcfg.productions:
		if pcfg.parameters[p] > 0 or not discard_zero:
			my_cfg.productions.add(p)
		assert len(p) == 2 or len(p) == 3
		if len(p) == 2:
			A,a = p
			assert A in my_cfg.nonterminals
			assert a in my_cfg.terminals
		if len(p) == 3:
			A,B,C = p
			assert A in my_cfg.nonterminals
			assert B in my_cfg.nonterminals
			assert C in my_cfg.nonterminals
class CFG:

	"""
	a CFG in CNF form
	"""
	def __init__(self):
		self.start = None
		self.nonterminals = set()
		self.productions = set()
		self.terminals = set()
	


	def compute_coreachable_set(self):
		"""
		return the set of all nonterminals that can generate a string.
		"""
		coreachable = set()
		iteration = 0
		
		prodmap = collections.defaultdict(list)
		for prod in self.productions:
			prodmap[prod[0]].append(prod)
		remaining = set(self.nonterminals)
		def prod_reachable(prod):
			for symbol in prod[1:]:
				if (not symbol in self.terminals) and (not symbol in coreachable):
					return False
			return True

		done_this_loop = 0
		while iteration == 0 or done_this_loop > 0:
			iteration += 1
			done_this_loop = 0
			for nt in remaining:
				for prod in prodmap[nt]:
					if prod_reachable(prod):
						done_this_loop += 1
						coreachable.add(nt)
						break
			remaining = remaining - coreachable


		return coreachable

	def compute_coreachable_productions(self, coreachable_nonterminals):
		"""
		Compute productions that can be used to generate something.
		"""
		good_productions = set()
		for prod in self.productions:
			for symbol in prod[1:]:
				if not symbol in coreachable_nonterminals and not symbol in self.terminals:
					break
			else:
				# we got to the end so its ok.
				good_productions.add(prod)
		return good_productions

	def compute_usable_productions(self, trim_set):
		"""
		return a list of the productions that are usable.
		"""
		tp = []
		for prod in self.productions:
			if prod[0] in trim_set:
				if len(prod) == 2:
					tp.append(prod)
				if prod[1] in trim_set and prod[2] in trim_set:
					tp.append(prod)
		return tp

	def compute_trim_set(self):
		"""
		return the set of all nonterminals A that can both generate a string 
		and have a context.
		"""
		#print "computing coreachable"
		coreachable = self.compute_coreachable_set()
		#print "coreachable", coreachable
		trim = set()
		good_productions = self.compute_coreachable_productions(coreachable)
		
		if self.start in coreachable:
			trim.add(self.start)
		done = len(trim)
		#print("start ", done)
		while done > 0:
			done = 0
			for prod in good_productions:
				if prod[0] in trim:
					#print(prod)
					for symbol in prod[1:]:
						if symbol in self.nonterminals and not symbol in trim:
							done += 1
							trim.add(symbol)
		#print "Trim set", trim
		return trim





