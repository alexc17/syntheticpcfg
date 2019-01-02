#dpda.py
from __future__ import print_function
import cfg
import numpy.random
import uniformsampler
from utility import collect_yield


class DPDA:
	"""
	A deterministic push down automaton.
	"""


	def __init__(self):
		"""
		Default rceognizes a LOGDCFL complete language.
		"""

		self.terminals = set(['a1', 'a2', 'b1', 'b2', 'x', 'y', 'z'])
		self.stack_symbols = set(['z0', '1', '2'])
		self.states = set(['I', 'B', 'R1','S1', 'G1', 'G2', 'S2', 'R2', 'F'])
		self.start_state = 'I'
		self.final_states = set(['F'])
		self.initial_stack = 'z0'
		# accet by empty stack.
		## Map from 
		self.transitions = {
			("I", 'a1', 'z0') : ("I", ('z0','1')),
			("I", 'a1', '1') : ("I", ('1','1')),
			("I", 'a1', '2') : ("I", ('2','1')),
			("I", 'a2', 'z0') : ("I", ('z0','2')),
			("I", 'a2', '1') : ("I", ('1','2')),
			("I", 'a2', '2') : ("I", ('2','2')),
			("I", 'b1', '1') : ("I", ()),
			("I", 'b2', '2') : ("I", ()),
#			("I", 'x', 'z0') : ("B", ('z0',)),
			("I", 'x', '1') : ("B", ('1',)),
			("I", 'x', '2') : ("B", ('2',)),
			# B
			("B", 'b1', '1') : ("R1", ()),
			("B", 'b1', '2') : ("G2", ('2',)),

			# R1 -- is 
			("R1", 'a1', 'z0') : ("R1", ('z0','1')),
			("R1", 'a1', '1') : ("R1", ('1','1')),
			("R1", 'a1', '2') : ("R1", ('2','1')),
			("R1", 'a2', 'z0') : ("R1", ('z0','2')),
			("R1", 'a2', '1') : ("R1", ('1','2')),
			("R1", 'a2', '2') : ("R1", ('2','2')),
			("R1", 'b1', '1') : ("R1", ()),
			("R1", 'b2', '2') : ("R1", ()),

			# stack unchanged
			("R1", 'y', 'z0') : ("S1", ('z0',)),
			("R1", 'y', '1') : ("S1", ('1',)),
			("R1", 'y', '2') : ("S1", ('2',)),
			## and then a b2 which we discare
			("S1", 'b2', 'z0') : ("G1", ('z0',)),
			("S1", 'b2', '1') : ("G1", ('1',)),
			("S1", 'b2', '2') : ("G1", ('2',)),
			# "G1 just discards \Sigma"
			("G1", 'a1', 'z0') : ("G1", ('z0',)),
			("G1", 'a2', 'z0') : ("G1", ('z0',)),
			("G1", 'b1', 'z0') : ("G1", ('z0',)),
			("G1", 'b2', 'z0') : ("G1", ('z0',)),
			("G1", 'a1', '1') : ("G1", ('1',)),
			("G1", 'a2', '1') : ("G1", ('1',)),
			("G1", 'b1', '1') : ("G1", ('1',)),
			("G1", 'b2', '1') : ("G1", ('1',)),
			("G1", 'a1', '2') : ("G1", ('2',)),
			("G1", 'a2', '2') : ("G1", ('2',)),
			("G1", 'b1', '2') : ("G1", ('2',)),
			("G1", 'b2', '2') : ("G1", ('2',)),	
			# and then goes to F when we have a z  and empty stack

			("G1", 'z', 'z0') : ("F", ('z0',)),
			# otherwise 
			("G1", 'z', '1') : ("I", ('1',)),
			("G1", 'z', '2') : ("I", ('2',)),

			# similarly 
			# "G2 just discards \Sigma"
			("G2", 'a1', 'z0') : ("G2", ('z0',)),
			("G2", 'a2', 'z0') : ("G2", ('z0',)),
			("G2", 'b1', 'z0') : ("G2", ('z0',)),
			("G2", 'b2', 'z0') : ("G2", ('z0',)),
			("G2", 'a1', '1') : ("G2", ('1',)),
			("G2", 'a2', '1') : ("G2", ('1',)),
			("G2", 'b1', '1') : ("G2", ('1',)),
			("G2", 'b2', '1') : ("G2", ('1',)),
			("G2", 'a1', '2') : ("G2", ('2',)),
			("G2", 'a2', '2') : ("G2", ('2',)),
			("G2", 'b1', '2') : ("G2", ('2',)),
			("G2", 'b2', '2') : ("G2", ('2',)),	
			# then goses to s2 when we have a y
			# stack unchanged
			("G2", 'y', 'z0') : ("S2", ('zo',)),
			("G2", 'y', '1') : ("S2", ('1',)),
			("G2", 'y', '2') : ("S2", ('2',)),	
			# then a b2 which we pop
			("S2", 'b2', '2') : ("R2", ()),	
			# the r2 just does that dyck thing
			# R1 -- is 
			("R2", 'a1', 'z0') : ("R2", ('z0','1')),
			("R2", 'a1', '1') : ("R2", ('1','1')),
			("R2", 'a1', '2') : ("R2", ('2','1')),
			("R2", 'a2', 'z0') : ("R2", ('z0','2')),
			("R2", 'a2', '1') : ("R2", ('1','2')),
			("R2", 'a2', '2') : ("R2", ('2','2')),
			("R2", 'b1', '1') : ("R2", ()),
			("R2", 'b2', '2') : ("R2", ()),
			# 

			("R2", 'z', 'z0') : ("F", ('z0',)),
			# otherwise 
			("R2", 'z', '1') : ("I", ('1',)),
			("R2", 'z', '2') : ("I", ('2',)),



		}



	def test(self, input_tuple,verbose=False):
		"""
		Return true or false.
		"""
		stack = [self.initial_stack]
		current_state = self.start_state
		while (current_state, stack[-1]) in self.transitions:
			current_state, ss = self.transitions[(current_state, stack.pop())]
			for s in ss:
				stack.append(s)
		for input_symbol in input_tuple:

			if verbose: print (current_state, input_symbol, stack)
			idx = (current_state, input_symbol, stack.pop())
			if idx in self.transitions:
				current_state, ss = self.transitions[idx]
				if verbose: print("-->", current_state, ss)
				for s in ss:
					stack.append(s)
			else:
				#print "NO transition", idx
				return False
			while (current_state, stack[-1]) in self.transitions:
				current_state, ss = self.transitions[(current_state, stack.pop())]
				for s in ss:
					stack.append(s)
		return current_state in self.final_states



positive = [
	
	"a1 x b1 y b2 z",	
	"a2 x b1 y b2 z",	
	"a1 b1 a2 x b1 y b2 z",	
	"a1 a2 b2 x b1 y b2 z",	
	"a2 x b1 y b2 z",	

	"a1 a2 x b1 y b2 z x b1 y b2 z",
	"a1 a2 x b1 y b2 b1 z",
	
]

negative = [
	"b2 b2",
	"x",
	"y",
	"x y z"
	"a1 a2 b2 b1  a2 b2 b1",
	"a1 a2 b2 b1",
	"a1 a2 b2 b1 a1 a2 b2 b1",
	"a1 a2 x b1 y b2  z",
	"a1 a2 x b1 y b2 b2 z",
	'a1 a2 b2 a2 b2 b1 a1 x b1 y a2 b1 a2 a2 a2 a1 z'

]


import numpy.random

class Sampler:

	def __init__(self, max_length):
		self.cfg = cfg.CFG()
		self.cfg.start = "S"
		self.cfg.nonterminals = set(["S", "S1", "S2", "A1", "A2", "B1","B2"])
		self.cfg.terminals = set(["a1","a2","b1","b2"])
		self.cfg.productions = set([ 
			("S","A1","B1"),
			("S","A2","B2"),
			("S","A1","S1"),
			("S","A2","S2"),
			("S1","S","B1"),
			("S2","S","B2"),
			("S","S","S"),
			("A1","a1"),	
			("A2","a2"),	
			("B1","b1"),	
			("B2","b2"),
			])
		self.sampler = uniformsampler.UniformSampler(self.cfg, max_length)

	def sample_string(self, mu = 3):
		sigma = ["a1","a2","b1","b2"]
		l = numpy.random.poisson(mu)
		idx = numpy.random.choice(range(len(sigma)), l)
		return [   sigma[i] for i in idx ]

	def sample_dyck(self):
		n = numpy.random.poisson(2) * 2
		if n == 0:
			return ()
		tree = self.sampler.sample(n)
		d = collect_yield(tree)	
		return d

	def sample(self,n,verbose=False):
		tree = self.sampler.sample(n)
		d = collect_yield(tree)
		if verbose: print(d)
		output = []
		state = "INITIAL"
		for i in d:
			if verbose: print(i)
			if i == "a1" or i == "a2":
				output.append(i)
			else:
				if numpy.random.randint(0,2) == 0:
					output.append(i)
				else:
					if verbose: print("SPLITTING")
					if state == "INITIAL":
						output.append("x")
					if state == "FIRST":
						output.append("y")
						output.append("b2")
						output.extend(self.sample_string())
						output.append("z")
						output.append("x")
					if state == "SECOND":
						output.append("z")
						output.append("x")
					if i =="b1":
						state = "FIRST"
						output.append("b1")
					else:
						state = "SECOND"
						output.append("b1")
						output.extend(self.sample_string())
						output.append("y")
						output.append("b2")
					if verbose: print(output)
		## Now all done.
		if state == "INITIAL":
			##we got to the end without picking one 
			if verbose: print("COMPLETE Dyck so adding")
			if numpy.random.randint(0,2) == 0:
				output.extend(("a1","x","b1"))
				output.extend(self.sample_dyck())
				output.extend(("y","b2"))
				output.extend(self.sample_string())
			else:
				output.extend(("a2","x","b1"))
				output.extend(self.sample_string())
				output.extend(("y","b2"))
				output.extend(self.sample_dyck())


		if state == "FIRST":
			output.append("y")
			output.append("b2")
			output.extend(self.sample_string())
		
		output.append("z")
		return output






if __name__ == "__main__":
	mydpda = DPDA()
	print("Positive")
	for s in positive:
		print(s,mydpda.test(s.split()))

	print("Negative")
	for s in negative:
		print(s,mydpda.test(s.split()))
	mysampler = Sampler(50)
	for i in range(1000):
		l = 2 * numpy.random.poisson(3) + 2
		if l > 48:
			l = 48
		dyckword = mysampler.sample(l,verbose=False)
		print(dyckword)
		if not mydpda.test(dyckword):
			print(dyckword,"FAIL")
			exit()


	## Sample





