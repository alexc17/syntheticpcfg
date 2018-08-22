#cfgfactory.py
import utility
import numpy.random
import cfg


class CFGFactory:

	def __init__(self):
		self.number_terminals = 100
		self.number_nonterminals = 5
		self.binary_rules = 40
		self.lexical_rules = 100


	def generate_nonterminals(self):
		nonterminals = [ 'S']
		for i in range(1,self.number_nonterminals):
			nonterminals.append("NT" + str(i))
		return nonterminals


	def sample_trim(self):
		"""
		Sample one and then trim it.
		return the trim one.
		If empty, raise an exception.
		"""
		my_cfg = self.sample_raw()
		ts = my_cfg.compute_trim_set()
		if len(ts) == 0:
			# empty language
			raise ValueError("Empty language")

		prods = my_cfg.compute_usable_productions(ts)
		terminals = set()
		for prod in prods:
			if len(prod) == 2:
				terminals.add(prod[1])
		tcfg = cfg.CFG()
		tcfg.start = my_cfg.start
		tcfg.terminals = terminals
		tcfg.nonterminals = ts
		tcfg.productions = set(prods)
		return tcfg





	def sample_raw(self):
		"""
		return a CFG
		"""
		lexicon = list(utility.generate_lexicon(self.number_terminals))
		nonterminals = self.generate_nonterminals()
		lprods = set()
		bprods= set()
		while len(lprods) < self.lexical_rules:
			lhs = numpy.random.choice(nonterminals)
			rhs = lexicon[numpy.random.choice(range(len(lexicon)))]
			lprods.add( (lhs,rhs))
		while len(bprods) < self.binary_rules:
			a,b,c = numpy.random.choice(nonterminals,size=3)
			
			bprods.add( (a,b,c))	
		my_cfg = cfg.CFG()
		my_cfg.start = nonterminals[0]
		my_cfg.nonterminals = set(nonterminals)
		my_cfg.terminals = set(lexicon)
		my_cfg.productions = lprods | bprods
		return my_cfg



