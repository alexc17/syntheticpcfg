# Code for performing various types of strong and weak evaluations on 
# synthetic PCFGs


import pcfg
import inside
import utility
import logging
import math
from collections import Counter
from sklearn.metrics.cluster import v_measure_score






def string_kld(target, hypothesis, samples= 1000):
	### sample n trees from target. 
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		lp = inside_target.inside_log_probability(s)
		lq = inside_hypothesis.inside_log_probability(s)
		total += lp - lq
	return total/samples


def bracketed_kld(target, hypothesis, samples= 1000):
	### sample n trees from target. 
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		
		lp = inside_target.inside_bracketed_log_probability(t)
		lq = inside_hypothesis.inside_bracketed_log_probability(t)
		total += lp - lq
	return total/samples

def preterminal_vi(target, hypothesis, samples = 1000):
	
	ct = preterminal_contingency_table(target, hypothesis, samples)
	return utility.variation_of_information(ct)

def preterminal_contingency_table(target, hypothesis, samples = 1000):
	counter = Counter()
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	for i in range(samples):
		t = sampler.sample_tree()
		tut = utility.tree_to_preterminals(t)
		s = utility.collect_yield(t)
		try:
			th = inside_hypothesis.viterbi_parse(s)
		except utility.ParseFailureException as e:
			logging.warning("Parse failure", s)
			continue
		tpt = utility.tree_to_preterminals(th)
		for a,b in zip(tut,tpt):
			counter[(a,b)] += 1
	return counter


def nonterminal_variation_of_information(target, hypothesis, samples = 1000):
	ct = nonterminal_contingency_table(target, hypothesis, samples)
	return utility.variation_of_information(ct)

def nonterminal_contingency_table(target, hypothesis, samples = 1000, robust=False):
	counter = Counter()
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	
	def gather_pairs(tree1,tree2,counter):
		assert len(tree1) == len(tree2)
		counter[ (tree1[0],tree2[0])] += 1
		if len(tree1) == 3:
			gather_pairs(tree1[1],tree2[1], counter)
			gather_pairs(tree1[2],tree2[2], counter)
	for i in range(samples):
		t = sampler.sample_tree()
		try:
			th = inside_hypothesis.bracketed_viterbi_parse(t) 
			gather_pairs(t,th, counter)
		except utility.ParseFailureException as e:
			if robust:
				logging.info("Parse failure while doing the bracketed parse.")
			else:
				raise e
	return counter



def best_nonterminal_map(g1,g2, samples = 1000):
	ctable = nonterminal_contingency_table(g1,g2,samples)
	map12 = {}
	for nt in g1.nonterminals:
		# avoiding the double list comprehension for clarity
		map12[nt] = max( g2.nonterminals, key = lambda x : ctable[(nt,x)] ) 
	return map12

def best_map_preterminals(g1, g2, samples = 1000):
	ctable = preterminal_contingency_table(g1, g2, samples)
	map12 = {}
	for nt in g1.nonterminals:
		# avoiding the double list comprehension for clarity
		map12[nt] = max( g2.nonterminals, key = lambda x : ctable[(nt,x)] ) 
	return map12

def best_rmap_preterminals(g1, g2, samples = 1000):
	"""
	return a map g2 -> g1, but by sampling from g1
	"""

	ctable = preterminal_contingency_table(g1, g2, samples)
	map21 = {}
	for nt in g2.nonterminals:
		# avoiding the double list comprehension for clarity
		map21[nt] = max( g1.nonterminals, key = lambda x : ctable[(x,nt)] ) 
	return map21

def bracketed_exact_match(target, hypothesis, samples = 1000):
	"""
	Propprtion of trees whose viterbi parse has the same shape as the original.
	"""
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		tut = utility.tree_to_unlabeled_tree(t)
		s = utility.collect_yield(t)
		try:
			th = inside_hypothesis.viterbi_parse(s)
			thut = utility.tree_to_unlabeled_tree(th)
			if thut == tut:
				total += 1
		except utility.ParseFailureException as e:
			logging.warning("Parse failure", s)
		
	return total/samples

def labeled_exact_match(target, hypothesis, samples = 1000):
	"""
	Propprtion of trees whose viterbi parse has the same shape as the original.
	"""
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	ntmap = best_rmap_preterminals(target, hypothesis, samples)
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		try:
			th = inside_hypothesis.viterbi_parse(s)
			relabeled_tree = utility.relabel_tree(th, ntmap)
			if relabeled_tree == t:
				total += 1
		except utility.ParseFailureException as e:
			logging.warning("Parse failure", s)
	return total/samples


def labeled_kld_exact(target, hypothesis, injection):
	"""
	injection is a mapping from target to hypothesis.

	We can compute this in closed form.
	"""
	# check it is an injection
	assert len(target.nonterminals) >= len(hypothesis.nonterminals), "Hypothesis is too small (not enough nonterminals)"
	assert len(target.nonterminals) == len(set(injection.values())), "Map is not an injection"
	pe = target.production_expectations()
	kld = 0.0
	for prod in pe:
		e = pe[prod]
		alpha = target.parameters[prod]
		if len(prod) == 2:
			newprod = (injection[prod[0]], prod[1])
		else:
			newprod = (injection[prod[0]], injection[prod[1]],injection[prod[2]])
		if not newprod in hypothesis.parameters:
			raise utility.ParseFailureException()
		else:
			beta = hypothesis.parameters[newprod]
			kld += e * math.log(alpha/beta)
	return kld


def test_subgrammar(target, hypothesis, samples = 1000):
	"""
	return true if the hypothesis has a CFG subgrammar isomorphic to the target.
	"""
	try:
		ntmap = best_nonterminal_map(target, hypothesis, samples)
	except utility.ParseFailureException as e:
		# We know that we failed to parse a tree so we can't have a subgrammar.
		return False
	
	if len(ntmap)!= len(target.nonterminals):
		return False
	if len(set(ntmap.values())) != len(target.nonterminals):
		return False
	for prod in target.productions:
		if len(prod) == 2:
			newprod = (ntmap[prod[0]], prod[1])
		else:
			newprod = (ntmap[prod[0]], ntmap[prod[1]],ntmap[prod[2]])
		if not newprod in hypothesis.parameters:
			return False
	return True

def test_coverage(target,hypothesis, samples = 1000):
	"""
	Sample n strings from target and see if they are parsed by hypothesis.

	optimisation: parse bracketed string first. 
	"""
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for _ in range(samples):
		t = sampler.sample_tree()
		try:
			vp = inside_hypothesis.bracketed_viterbi_parse(t)
			total += 1
		except utility.ParseFailureException as e:
			try:
				s = utility.collect_yield(t)
				vp = inside_hypothesis.viterbi_parse(s)
				total+=1
			except utility.ParseFailureException as e:
				pass
	return total/samples
		
