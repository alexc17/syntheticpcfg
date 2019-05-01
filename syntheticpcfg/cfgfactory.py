#cfgfactory.py
import utility
import numpy.random
import cfg
import logging

class CFGFactory:

	def __init__(self):
		self.number_terminals = 100
		self.number_nonterminals = 5
		self.binary_rules = 40
		self.lexical_rules = 100
		self.strict_cnf = True


	def generate_nonterminals(self):
		nonterminals = [ 'S']
		for i in range(1,self.number_nonterminals):
			nonterminals.append("NT" + str(i))
		return nonterminals

	def sample_full(self):
		lexicon = list(utility.generate_lexicon(self.number_terminals))
		nonterminals = self.generate_nonterminals()
		lprods = set()
		bprods= set()
		for a in nonterminals:
			for b in lexicon:
				lprods.add((a,b))
		for a in nonterminals:
			for b in nonterminals[1:]:
				for c in nonterminals[1:]:
					bprods.add((a,b,c))
		my_cfg = cfg.CFG()
		my_cfg.start = nonterminals[0]
		my_cfg.nonterminals = set(nonterminals)
		my_cfg.terminals = set(lexicon)
		my_cfg.productions = lprods | bprods
		return my_cfg

	def sample_trim(self):
		"""
		Sample one and then trim it.
		return the trim one.
		If empty, raise an exception.
		"""
		my_cfg = self.sample_raw()
		logging.info("CFG nominally has %d nonterminals, %d terminals, %d binary_rules and %d lexical rules", self.number_nonterminals,self.number_terminals,self.binary_rules,self.lexical_rules)
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
		logging.info("Final CFG has %d nonterminals, %d terminals, %d binary_rules and %d lexical rules",
			len(tcfg.nonterminals), len(tcfg.terminals), 
			len([prod for prod in tcfg.productions if len(prod) == 3]),
			len([prod for prod in tcfg.productions if len(prod) == 2]))
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
			if self.strict_cnf:
				a =  numpy.random.choice(nonterminals)
				b,c = numpy.random.choice(nonterminals[1:],size=2)
			else:
				a,b,c = numpy.random.choice(nonterminals,size=3)
			bprods.add( (a,b,c))	
		my_cfg = cfg.CFG()
		my_cfg.start = nonterminals[0]
		my_cfg.nonterminals = set(nonterminals)
		my_cfg.terminals = set(lexicon)
		my_cfg.productions = lprods | bprods
		return my_cfg



