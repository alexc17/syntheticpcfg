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
	
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	x = []
	y = []
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		tut = utility.tree_to_preterminals(t)
		x.append(tut)
		s = utility.collect_yield(t)
		th = inside_hypothesis.viterbi_parse(s)
		tpt = utility.tree_to_preterminals(th)
		y.append(tpt)
	return v_measure_score(x,y)

def preterminal_contingency_table(target, hypothesis, samples = 1000):
	counter = Counter()
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		tut = utility.tree_to_preterminals(t)
		s = utility.collect_yield(t)
		try:
			th = inside_hypothesis.viterbi_parse(s)
		except utility.ParseFaiureException as e:
			logging.warning("Parse failure", s)
			continue
		tpt = utility.tree_to_preterminals(th)
		for a,b in zip(tut,tpt):
			counter[(a,b)] += 1
	return counter

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
		except utility.ParseFaiureException as e:
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
		except utility.ParseFaiureException as e:
			logging.warning("Parse failure", s)
	return total/samples


def labeled_kld_exact(target, hypothesis, bijection):
	"""
	Bijection is a mapping from target to hypothesis.
	"""
	pe = target.production_expectations()
	kld = 0.0
	for prod in pe:
		e = pe[prod]
		alpha = target.parameters[prod]
		if len(prod) == 2:
			newprod = (bijection[prod[0]], prod[1])
		else:
			newprod = (bijection[prod[0]], bijection[prod[1]],bijection[prod[2]])
		if not newprod in hypothesis.parameters:
			raise utility.ParseFaiureException()
		else:
			beta = hypothesis.parameters[newprod]
			kld += e * math.log(alpha/beta)
	return kld
