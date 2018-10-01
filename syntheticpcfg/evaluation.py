# Code for performing various types of strong and weak evaluations on 
# synthetic PCFGs
import pcfg
import inside
import utility

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
		lq = inside_target.inside_log_probability(s)
		total += lp - lq
	return total/samples

