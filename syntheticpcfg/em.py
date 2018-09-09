#em.py
from collections import defaultdict
import inside
from utility import *
import logging

MAX_LENGTH = 20
MAX_EM_ITERATIONS = 100
EPSILON = 1e-8

class Trainer:
	"""
	class for training a PCFG on a corpus of *strings.*
	"""

	def __init__(self,training_file):
		## No point in using a counter here.
		if training_file:
			self.training = self._load(training_file)
		self.max_length = MAX_LENGTH
		


	def _load(self, corpus):
		result = []
		with open(corpus,'r') as inf:
			for line in inf:
				if not lines.startswith('#'):
					s = tuple(line.split())
					if len(s) > 0:
						result.append(s)
		return result

	def train(self,mypcfg):
		lps = []
		mypcfg = mypcfg.copy()
		for i in range(MAX_EM_ITERATIONS):
			lp, mypcfg = self.train_once(mypcfg)
			logging.info("Iteration %d, log prob %f", i, lp)
			if i > 0:
				if abs(lp - lps[-1]) > EPSILON * len(self.training):
					lps.append(lp)
				else:
					logging.info("Converged : lp difference is %f", abs(lp - lps[-1]))
					return mypcfg,lps
			else:
				lps.append(lp)

	def train_once(self, mypcfg, skip_errors = False):
		insider = inside.InsideComputation(mypcfg)
		posteriors = defaultdict(float)
		total_lp = 0.0
		too_long = 0
		for s in self.training:
			if len(s) <= self.max_length:
				try:
					total_lp += insider.add_posteriors(s, posteriors, 1.0)
				except ParseFailureException as e:
					if skip_errors:
						logging.error("Parse failure on ", s)
					else:
						raise e
			else:
				too_long += 1

		logging.info("Skipped %d sentences which were > %d long", too_long, self.max_length)
				
		mypcfg.parameters = posteriors
		mypcfg.trim_zeros()
		mypcfg.normalise()
		return total_lp, mypcfg
