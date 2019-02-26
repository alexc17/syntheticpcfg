# Code for performing various types of strong and weak evaluations on 
# synthetic PCFGs


import pcfg
import inside
import utility
import logging
import math
from collections import Counter
from sklearn.metrics.cluster import v_measure_score




## Weak probabilistic

def string_kld(target, hypothesis, samples= 1000, verbose=False):
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
		if verbose:
			logging.info("Sample %d %s, target %f, hypothesis %f", i,t,lp,lq)
		total += lp - lq
	return total/samples

# Strong probabilistic
def bracketed_kld(target, hypothesis, samples= 1000, verbose=False):
	### sample n trees from target. FAST
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		
		lp = inside_target.inside_bracketed_log_probability(t)
		lq = inside_hypothesis.inside_bracketed_log_probability(t)
		if verbose:
			logging.info("Sample %d %s, target %f, hypothesis %f", i,t,lp,lq)
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

def best_nonterminal_rmap(g1,g2, samples = 1000):
	ctable = nonterminal_contingency_table(g1,g2,samples)
	map21 = {}
	for nt in g2.nonterminals:
		# avoiding the double list comprehension for clarity
		map21[nt] = max( g1.nonterminals, key = lambda x : ctable[(x,nt)] ) 
	return map21

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






def bracketed_match(target, hypothesis, test_viterbi = False, samples = 1000, verbose=False, exact_match=True):
	"""
	Proportion of trees whose viterbi parse has the same shape as the original.
	test viterbi option means that it will test against the viterbi parse wrt the true grammar not the original tree
	"""
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	ttotal = 0.0
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if test_viterbi:
			t = inside_target.viterbi_parse(s)
		try:
			th = inside_hypothesis.viterbi_parse(s)
			if exact_match:
				num,denom = utility.zero_one_unlabeled(t,th)
			else:
				num,denom = utility.microaveraged_unlabeled(t,th)
			total += num
			ttotal += denom
			if verbose and num < denom:
				logging.info("Mismatch (%d / %d) with string %s",num,denom, s)
		except utility.ParseFailureException as e:
			logging.warning("Parse failure", s)
			
	return total/ttotal

def labeled_exact_match(target, hypothesis, samples = 1000, test_viterbi = False, verbose=False):
	"""
	Proportion of trees whose viterbi parse is the same up to a relabeling of the hypothesis tree.

	SLOW
	"""
	if test_viterbi:
		inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	ntmap = best_nonterminal_rmap(target, hypothesis, samples)
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)
		if test_viterbi:
			t = inside_target.viterbi_parse(s)
		try:
			th = inside_hypothesis.viterbi_parse(s)
			relabeled_tree = utility.relabel_tree(th, ntmap)
			if relabeled_tree == t:
				total += 1
			elif verbose:
				logging.info("Mismatch in trees with parse of %s", s)
				print(relabeled_tree)
				print(t)
		except utility.ParseFailureException as e:
			logging.warning("Parse failure", s)
	return total/samples


def labeled_kld_exact(target, hypothesis, injection=None):
	"""
	injection is a mapping from target to hypothesis.

	We can compute this in closed form.
	"""
	# check it is an injection
	assert len(target.nonterminals) >= len(hypothesis.nonterminals), "Hypothesis is too small (not enough nonterminals)"
	if not injection:
		injection = best_nonterminal_map(target,hypothesis)
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

# Strong non probabilistic

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


def test_cfg_morphism(target, hypothesis, bijection):
	"""
	return true if bijection is a morphism from target to hypothesis.

	"""
	for prod in target.productions:
		if len(prod) == 2:
			newprod = (bijection[prod[0]], prod[1])
		else:
			newprod = (bijection[prod[0]], bijection[prod[1]],bijection[prod[2]])
		if not newprod in hypothesis.parameters:
			return False
	return True

def test_cfg_isomorphism(target, hypothesis, bijection):
	"""
	return true if bijection is an isomorphism from target to hypothesis.

	"""
	return (test_cfg_morphism(target, hypothesis, bijection) and 
		test_cfg_morphism(hypothesis, target, { bijection[a]: a for a in bijection}))

	

# Weak non probabilistic

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
		
#
# Strong probabilistic 
def conditional_kld(target, hypothesis, samples = 1000,verbose=False):
	"""
	Estimate the kld between the conditional probability distributions.

	for a given string $w$ D( P(tree|w) | Q(tree|w)).

	difference between string KLD and tree KLD.
	"""
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	ntmap = best_nonterminal_map(target, hypothesis, samples)
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)

		ptree = target.log_probability_derivation(t)
		pstring = inside_target.inside_log_probability(s)

		qt = utility.relabel_tree(t, ntmap)
		qtree = hypothesis.log_probability_derivation(qt)
		qstring = inside_hypothesis.inside_log_probability(s)

		total += (ptree - qtree) - (pstring - qstring)
		if verbose:
			logging.info("%s p(t) = %f, p(w) = %f, q(t) = %f, q(w) = %f", s, ptree,pstring,qtree,qstring)
	return total/samples	


def conditional_unlabeled_kld(target, hypothesis, samples = 1000, verbose=False):
	"""
	Estimate the kld between the conditional probability distributions.

	for a given string $w$ D( P(unlabeled tree|w) | Q(tree|w)).

	difference between string KLD and tree KLD.
	"""
	inside_target = inside.InsideComputation(target)
	inside_hypothesis = inside.InsideComputation(hypothesis)
	sampler = pcfg.Sampler(target)
	total = 0.0
	
	for i in range(samples):
		t = sampler.sample_tree()
		s = utility.collect_yield(t)

		ptree = inside_target.inside_bracketed_log_probability(t)
		pstring = inside_target.inside_log_probability(s)

		
		qtree = inside_hypothesis.inside_bracketed_log_probability(t)
		qstring = inside_hypothesis.inside_log_probability(s)

		total += (ptree - qtree) - (pstring - qstring)
		if verbose:
			logging.info("%s p(t) = %f, p(w) = %f, q(t) = %f, q(w) = %f", s, ptree,pstring,qtree,qstring)
	return total/samples	
