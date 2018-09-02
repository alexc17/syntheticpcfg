
import collections
import numpy as np
import math

from utility import *

class InsideComputation:

	def __init__(self, pcfg):
		## store all the probabilities in some useful form.
		self.lprod_index = collections.defaultdict(list)
		self.bprod_index = collections.defaultdict(list)
		self.rprod_index = collections.defaultdict(list)
		self.log_parameters = pcfg.log_parameters.copy()
		self.start = pcfg.start
		for prod in pcfg.productions:
			if len(prod) == 2:
				self.lprod_index[prod[1]].append( (prod, pcfg.log_parameters[prod]))
			if len(prod) == 3:
				self.bprod_index[prod[1]].append( (prod, pcfg.log_parameters[prod]))
				self.rprod_index[prod[2]].append( (prod, pcfg.log_parameters[prod]))


	def _bracketed_log_probability(self, tree):
		"""
		Compute log prob of all derivations with this bracketed tree.
		Return a dict mapping nontermials to lof probs
		"""
		if len(tree) == 2:
			return { prod[0]: lp for prod,lp in self.lprod_index[tree[1]]}
		else:
			left = self._bracketed_log_probability(tree[1])
			right = self._bracketed_log_probability(tree[2])
			answer = {}
			for leftcat in left:
				for prod, prod_lp in self.bprod_index[leftcat]:
					a,b,c = prod
					if c in right:
						score = prod_lp + left[leftcat] + right[c]
						if a in answer:
							answer[a] = np.logaddexp(score, answer[a])
						else:
							answer[a] = score
			return answer 



	def inside_log_probability(self, sentence):
		table,_ = self._compute_inside_table(sentence)
		return table[(0,self.start,len(sentence))]

	def inside_bracketed_log_probability(self, tree):
		table = self._bracketed_log_probability(tree)
		#print(table)
		return table[self.start]

	def _compute_inside_table(self,sentence,mode=0):
		"""
		mode 0 logprobs
		mode 1 max log prob
		mode 2 counts
		"""
	
		## simple cky but sparse
		l = len(sentence)
		# take the ;eft hand one and then loop through all productions _ -> A _
		## table mapping items to lp
		table = {}
		## index mapping start end to list of category,lp pairs
		table2 = collections.defaultdict(list)
		def _add_to_chart( start, category, end, score):
			table[ (start,category,end)] = score
			table2[(start,end)].append((category,score))

		for i,a in enumerate(sentence):
			for prod, lp in self.lprod_index[a]:
				if mode == 0:
					score = lp
				elif mode == 1:
					score = (lp,)
				else:
					score= 1
				_add_to_chart(i,prod[0],i+1,score)
					
		#print("Lexical items", table)
		for width in range(2,l+1):
			for start in range(0, l+1 -width):
				end = start + width
				scores = {}
				for middle in range(start+1, end):
					#print(start,middle,end)
					#print("table2 ", table2[(start,middle)])
					for leftnt,left_score in table2[(start,middle)]:
						#print("table2",leftnt,lp1)
						
						#print("index of binary rules", leftnt, self.bprod_index[leftnt] )
						for prod,prod_lp in self.bprod_index[leftnt]:
							#print("trying",prod,lp2)
							lhs, left, right = prod
							assert left == leftnt
							# now see if we have a (middle,right,end ) in the chart
							try:
								right_score = table[(middle,right,end)]
								if mode == 0:
									score = left_score + prod_lp + right_score
								elif mode == 1:
									score = left_score[0] + prod_lp + right_score[0]
								else:
									# counts 
									score = left_score * right_score
								#print(lp1,lp2,lp3,score)

								if lhs in scores:
									if mode == 0:
										scores[lhs]= np.logaddexp(score, scores[lhs])
									elif mode == 1:
										# Viterbi logprob
										if score > scores[lhs][0]:

											scores[lhs]= (score, middle, prod)

									elif mode == 2:
										# counts
										scores[lhs] += score
								else:
									if mode == 1:
										scores[lhs]= (score, middle, prod)
									else:
										scores[lhs] = score
							except KeyError:
								#print("no entry on the rright", middle, right,end)
								#print(table)
								pass
				for lhs in scores:
					_add_to_chart(start,lhs,end,scores[lhs])
		#print(table)
		#print(table2)
		return table, table2
	
	def _compute_outside_table(self, sentence, table, table2):
		# Compute the outside probs
		l = len(sentence)
		otable = {}
		otable[ (0, self.start,l)] = 0.0
		for width in reversed(range(1,l)):
			for start in range(0, l+1-width):
				scores = {}
				end = start+width
				#print("Starting",start,end)
				# we compute item of the form (start, cat, end)
				for leftpos in range(0,start):
					# case where this is the right branch of a binary rule.
					for leftcat, lp1 in table2[(leftpos, start)]:
						#print("Left", leftcat, leftpos)
						for prod,lp2 in self.bprod_index[leftcat]:
							#print("Production", prod)
							lhs, left, right = prod
							assert leftcat == left
							try:
								lp3 = otable[(leftpos,lhs,end)]
								#print("outside entry for ", leftpos,lhs,end,lp3)
								score = lp1 + lp2 + lp3
								if right in scores:
									scores[right]= np.logaddexp(score, scores[right])
								else:
									scores[right] = score
							except KeyError:
								#print("No outside entry for ", leftpos,lhs,end)
								pass
				for rightpos in range(end+1, l+1):

					# case where this is the left branch of a binary rule.
					for rightcat, lp1 in table2[(end, rightpos)]:
						#print("Right", rightcat, rightpos)
						for prod,lp2 in self.rprod_index[rightcat]:
							#print("Production", prod)
							lhs, left, right = prod
							assert rightcat == right
							try:
								lp3 = otable[(start,lhs,rightpos)]
								score = lp1 + lp2 + lp3
								if left in scores:
									scores[left]= np.logaddexp(score, scores[left])
								else:
									scores[left] = score
							except KeyError:
								# there is no 
								pass
				# completed for start end
				#print(start,end, scores)
				for cat in scores:
					otable[ (start, cat, end)] = scores[cat]
		return otable

	def add_posteriors(self, sentence, posteriors, weight=1.0):
		l = len(sentence)
		table, table2 = self._compute_inside_table(sentence)
		if not (0, self.start,l) in table:
			raise ValueError("Can't parse," , sentence)
		total_lp = table[(0, self.start,l)]
		otable = self._compute_outside_table(sentence, table, table2)
		
		for start,cat,end in otable:
			olp = otable[ (start,cat,end)]
			if end == start + 1:
				a = sentence[start]
				prod = (cat,a)
				if prod in self.log_parameters:
					rule_lp = self.log_parameters[prod]
					posterior = math.exp(rule_lp + olp - total_lp)
					posteriors[prod] += weight * posterior
			else:
				for middle in range(start+1,end):
					for lcat, ilp1 in table2[(start,middle)]:
						for rcat, ilp2 in table2[(middle,end)]:
							prod = (cat,lcat,rcat)
							if prod in self.log_parameters:
								rule_lp = self.log_parameters[prod]
								posterior = math.exp(olp + ilp1 + ilp2 + rule_lp - total_lp)
								posteriors[prod] += weight * posterior


	def viterbi_parse(self,sentence):
		table, table2 = self._compute_inside_table(sentence,mode=1)
		def extract_tree(start,lhs,end):
			if end == start + 1:
				return (lhs, sentence[start])
			else:
				if not (start,lhs,end) in table:
					raise ParseFailureException()
				score, middle, prod = table[(start,lhs,end)]
				lhs,rhs1,rhs2 = prod
				left_subtree = extract_tree(start,rhs1,middle)
				right_subtree = extract_tree(middle,rhs2,end)
				return (lhs, left_subtree,right_subtree)
		return extract_tree(0, self.start, len(sentence))

	def count_parses(self,sentence):
		table, table2 = self._compute_inside_table(sentence,mode=2)
		item = (0,self.start,len(sentence))
		if not item in table:
			return 0
		return table[item]


