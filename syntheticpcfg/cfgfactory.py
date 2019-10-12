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


	def sample_uniform(self, lp=0.5,bp = 0.5):
		"""
		Sample all productions Bernoulli for lexical and binary. Default 0.5.
		"""
		lexicon = list(utility.generate_lexicon(self.number_terminals))
		#print("Lexicon",lexicon,self.number_terminals)
		nonterminals = self.generate_nonterminals()
		productions = []
		for a in nonterminals:
			for b in lexicon:
				if numpy.random.random() < lp:
					productions.append((a,b))
		for a in nonterminals:
			for b in nonterminals[1:]:
				for c in nonterminals[1:]:
					if numpy.random.random() < bp:
						productions.append((a,b,c))
		my_cfg = cfg.CFG()
		my_cfg.start = nonterminals[0]
		my_cfg.nonterminals = set(nonterminals)
		my_cfg.terminals = set(lexicon)
		my_cfg.productions = productions
		return my_cfg

	def sample_full(self):
		lexicon = list(utility.generate_lexicon(self.number_terminals))
		#print("Lexicon",lexicon,self.number_terminals)
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
		#print(my_cfg.terminals)
		return my_cfg

	def sample_trim(self):
		"""
		Sample one and then trim it.
		return the trim one.
		If empty, raise an exception.
		"""
		my_cfg = self.sample_raw()
		#print([ prod for prod in my_cfg.productions if len(prod) == 2 and prod[0] == 'S'])
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
		#DEBUGGING
		lexicon.sort()
		print(lexicon[0],lexicon[-1])
		nonterminals = self.generate_nonterminals()
		lprods = set()
		bprods= set()
		lexicon_size = len(lexicon)
		while len(lprods) < self.lexical_rules:
			lhs = numpy.random.choice(nonterminals)
			rhs = lexicon[numpy.random.choice(range(lexicon_size))]
			
			lprods.add( (lhs,rhs))
		print(lhs,rhs)
		while len(bprods) < self.binary_rules:
			if self.strict_cnf:
				a =  numpy.random.choice(nonterminals)
				b,c = numpy.random.choice(nonterminals[1:],size=2)
			else:
				a,b,c = numpy.random.choice(nonterminals,size=3)
			
			bprods.add( (a,b,c))	
		print(a,b,c)
		my_cfg = cfg.CFG()
		my_cfg.start = nonterminals[0]
		my_cfg.nonterminals = set(nonterminals)
		my_cfg.terminals = set(lexicon)
		my_cfg.productions = lprods | bprods
		return my_cfg



