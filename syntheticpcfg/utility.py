 #utility.py
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict

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


def zero_one_unlabeled(tree1, tree2):
	u1 = tree_to_unlabeled_tree(tree1)	
	u2 = tree_to_unlabeled_tree(tree2)
	if u1 == u2:
		return (1,1)
	else:
		return (0,1)

def microaveraged_unlabeled(tree1, tree2):
	u1 = collect_unlabeled_spans(tree1)	
	u2 = set(collect_unlabeled_spans(tree2))
	assert len(u1) == len(u2)
	n = len(u1)
	x = 0
	for span in u1:
		if span in u2:
			x += 1
	return (x,n)
	
def collect_unlabeled_spans(tree):
	spans = []
	_collect_unlabeled_spans(0, tree, spans)
	return spans[:-1]

def _collect_unlabeled_spans(start, tree, spans):
	if len(tree) == 2:
		end = start+1
	else:
		end = _collect_unlabeled_spans(start, tree[1], spans)
		end = _collect_unlabeled_spans(end, tree[2], spans)
		spans.append((start, end))
	return end

def collect_labeled_spans(tree):
	spans = []
	_collect_spans(0, tree, spans)
	return spans

def _collect_spans(start, tree, spans):
	if len(tree) == 2:
		end = start+1
	else:
		end = _collect_spans(start, tree[1], spans)
		end = _collect_spans(end, tree[2], spans)
	spans.append((tree[0],start, end))
	return end
		
def tree_to_string(tree):
	if len(tree) == 2:
		return "(" + tree[0] + " " + tree[1] + ")"
	else:
		return "(" + tree[0] + " " + tree_to_string(tree[1]) + " " +tree_to_string(tree[2]) +  ")"

def tree_to_unlabeled_tree(tree):
	if len(tree) == 2:
		return tree[1]
	else:
		return (tree_to_unlabeled_tree(tree[1]), tree_to_unlabeled_tree(tree[2]))

def relabel_tree(tree, nt_map):
	if len(tree) == 2:
		return (nt_map[tree[0]],tree[1])
	else:
		return (nt_map[tree[0]],relabel_tree(tree[1],nt_map), relabel_tree(tree[2],nt_map))

def tree_to_preterminals(tree):
	"""
	return a list of the preterminals in the tree.
	"""
	if len(tree) == 2:
		return (tree[0],)
	else:
		# inefficient but it doesnt matter.
		return tree_to_preterminals(tree[1]) + tree_to_preterminals(tree[2])

def tree_depth(tree):
	if len(tree) == 2:
		return 1
	else:
		return 1+max(tree_depth(tree[1]), tree_depth(tree[2]))

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
    
    def knuth_layout(tree, depth):
        nonlocal i
        

        if len(tree) == 3:
            left = knuth_layout(tree[1], depth+1)
            location = (i,depth)
            i+= 1
            right = knuth_layout(tree[2], depth+1)
            return (location, tree[0], left, right)
        else:
            location = (i,depth)
            i += 1
            return (location, tree[0], tree[1])
    return knuth_layout(tree,0)

def plot_tree_with_layout(tree,w):
    
    x,y = tree[0]
    label = tree[1]
    character_offset = w * 0.01

    diff = len(label) * character_offset
    plt.text(x-diff,-y,label)
    if len(tree) == 4:
        for subtree in tree[2:]:
            x2,y2 = subtree[0]
            plt.plot( [x,x2],[-y-0.2,-y2+0.4],'b')
            plot_tree_with_layout(subtree,w)
    else:

        plt.plot( [x,x],[-y-0.2,-y-0.6],'r' )
        leaf = tree[2]
        ldiff = len(leaf) * character_offset
        # diff = 0
        plt.text(x-ldiff,-y-1.0,leaf)
    
def plot_tree(tree):
	l = len(collect_yield(tree))
	d = tree_depth(tree)
	plt.figure(figsize=(l/2, d/2))
	plt.axis('off')
	layout = knuth_tree(tree)
	plot_tree_with_layout(layout,l+d-1)
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
	
	length = max(1,int(math.ceil(math.log10(n))))
	dictionary = set()
	count = 0
	while len(dictionary) < n :
		newWord = generateRandomString(length)
		count = count + 1
		if not newWord in dictionary:
			dictionary.add(newWord)
	return dictionary

def variation_of_information(counter):
	total = sum(counter.values())
	x = defaultdict(float)
	y = defaultdict(float)
	for (a,b) in counter:
		n = counter[(a,b)]
		x[a] += n
		y[b] += n
	vi = 0
	for (a,b) in counter:
		p = x[a]/total
		q = y[b]/total
		r = counter[(a,b)]/total
		vi += r * (math.log(r/p)  + math.log(r/q))
	return -vi

def catalan_numbers(n, cache = {}):
	if cache and n in cache:
		return cache[n]
	if n == 0:
		cache[0] = 1
		return 1
	else:
		cn = 0
		for i in range(n):
			cn += catalan_numbers(i, cache) * catalan_numbers(n-1-i, cache)
		cache[n] = cn
		return cn
