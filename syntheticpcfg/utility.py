#utility.py
import random
import math
import matplotlib.pyplot as plt

class ParseFailureException(Exception):
    pass

def collect_yield(tree):
	return _append_yield(tree, [])

def _append_yield(tree, prefix):
	if len(tree) == 2:
		prefix.append(tree[1])
	else:
		prefix = _append_yield(tree[1], prefix)
		prefix = _append_yield(tree[2], prefix)
	return prefix
		
def tree_to_string(tree):
	if len(tree) == 2:
		return "(" + tree[0] + " " + tree[1] + ")"
	else:
		return "(" + tree[0] + " " + tree_to_string(tree[1]) + " " +tree_to_string(tree[2]) +  ")"



def string_to_tree(tree):
	return _string_to_tree(tree,0)[0]

def _string_to_tree(s, index):
	# starting at a ( or beginning of a token.
	# returns index one after 
	if s[index] == '(':
		label,index = read_token(s,index+1)
		left, index = _string_to_tree(s,index)
		if s[index]== ')':
			return (label,left), index+1
		while s[index].isspace():
			index += 1
		right,index = _string_to_tree(s,index)
		while s[index].isspace():
			index += 1
		assert s[index] == ')'
		return (label,left,right),index+1
	else:
		return read_token(s,index)

def read_token(s,index):
	start = index
	while s[index].isalnum():
		index+=1
	end = index
	while s[index].isspace():
		index += 1
	return s[start:end], index
	



def count_productions(tree, counter):
	if len(tree) == 2:
		counter[tree] += 1
	else:
		father,left,right = tree
		counter[(father,left[0],right[0])] += 1
		count_productions(left,counter)
		count_productions(right,counter)

def knuth_tree(tree):
    i = 0
    layout = {}
    def knuth_layout(tree, depth):
        nonlocal i
        nonlocal layout
        if len(tree) == 3:
            knuth_layout(tree[1], depth+1)
            layout[tree] = (i,depth)
            i+= 1
            knuth_layout(tree[2], depth+1)
        else:
            layout[tree] = (i,depth)
            i += 1
    knuth_layout(tree,0)
    return layout

def plot_tree_with_layout(tree, layout):
    
    x,y = layout[tree]
    label = tree[0]
    diff = len(label) *0.15
    plt.text(x-diff,-y,label)
    if len(tree) == 3:
        for subtree in tree[1:]:
            x2,y2 = layout[subtree]
            plt.plot( [x,x2],[-y-0.1,-y2+0.4],'b')
            plot_tree_with_layout(subtree,layout)
    else:
        plt.plot( [x,x],[-y-0.1,-y-0.5],'r' )
        leaf = tree[1]
        diff = len(leaf) *0.15
        plt.text(x-diff,-y-1.0,tree[1])
    
def plot_tree(tree):
	plt.axis('off')
	layout = knuth_tree(tree)
	plot_tree_with_layout(tree, layout)
	plt.show()

def generateRandomString(n):
	"""generate a random string of lower case letters of length n"""
	myString = ""
	for i in range(n):
		a = random.randint(0,25)
		letter = chr(ord('a') + a)
		myString += letter
	return myString

	
def strongly_connected_components(graph):
	"""
	Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
	for finding the strongly connected components of a graph.
	
	Graph is a dict mapping nodes to a list of their succesors.
	Returns a list of tuples
	
	Downloaded from http://www.logarithmic.net/pfh/blog/01208083168
	Allegedly by Dries Verdegem

	Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
	"""

	index_counter = [0]
	stack = []
	lowlinks = {}
	index = {}
	result = []
	
	def strongconnect(node):
		#print "calling strong connect", node
		# set the depth index for this node to the smallest unused index
		index[node] = index_counter[0]
		lowlinks[node] = index_counter[0]
		index_counter[0] += 1
		stack.append(node)
	
		# Consider successors of `node`
		try:
			successors = graph[node]
		except:
			successors = []
		for successor in successors:
			if successor not in lowlinks:
				# Successor has not yet been visited; recurse on it
				strongconnect(successor)
				lowlinks[node] = min(lowlinks[node],lowlinks[successor])
			elif successor in stack:
				# the successor is in the stack and hence in the current strongly connected component (SCC)
				lowlinks[node] = min(lowlinks[node],index[successor])
		
		# If `node` is a root node, pop the stack and generate an SCC
		if lowlinks[node] == index[node]:
			connected_component = []
			
			while True:
				successor = stack.pop()
				connected_component.append(successor)
				if successor == node: 
					break
			component = tuple(connected_component)
			# storing the result
			result.append(component)
	for node in graph:
		if node not in lowlinks:
			strongconnect(node)
	return result

def generate_lexicon(n):
	""" Generate a list of n tokens that can be used to be 
	words in a synthetic example"""
	
	length = int(math.ceil(math.log10(n)))
	dictionary = set()
	count = 0
	while len(dictionary) < n :
		newWord = generateRandomString(length)
		count = count + 1
		if not newWord in dictionary:
			dictionary.add(newWord)
	return dictionary