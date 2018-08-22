#pitmanyor.py
import numpy as np
import numpy.random as npr


def sample_twoparam_gem(k, d, alpha):
	"""
	Sample from a Two parameters GEM (Griffiths, Engen and McCloskey) distribution.
	Using a stick breaking method.
	d is discount parameter (big d means fatter tails)

	alpha is strength

	d= 0 gives Dirichlet
	This gives a power law with 
	with exponent 1/d. So d =1 or d= 0.95 is sensible.
	"""
	assert( 0 <= d and d < 1)
	assert(alpha > -d)
	prod = 1.0
	w = []
	for j in range(1,k+1):
		v_j = npr.beta(1-d, alpha + j * d)
		w_j = v_j * prod
		prod *= (1 - v_j)
		w.append(w_j)
	return w


def polya_urn(maxn,d,alpha):
	"""
	Use a Polya urn scheme.
	"""
	current_tables = 0
	table_counts = {}
	for n in range(1,maxn):
		r = npr.random()
		#print r
		for j in range(current_tables):
			nk = table_counts[j]
			#print nk
			p  = (nk - d)/ (n - 1 + alpha)
			#print "P", p, "J", j
			r -= p
			#print "r is now", r
			if r <= 0:
				## Put it in this table
				table_counts[j] += 1
				break
		else:
			# new table
			#print "NEw table, r " , r
			table_counts[current_tables] = 1
			current_tables += 1
	## sampling complete.
	return table_counts

def sample(k, d, alpha):
	"""
	Sample a k-dimensional distribution from Pitman Yor.
	"""
	w = np.array(sample_twoparam_gem(k, d, alpha))
	w /= np.sum(w)
	w= np.sort(w)
	w= np.flip(w,axis=0)
	return w
	
def sample2(k, d, alpha):
	"""
	Sample a k-dimensional distribution from Pitman Yor.
	Sample 10 times as many and just return the first k.
	This should give us a clean estimate of what we want.
	"""
	w = sample(10*k,d,alpha)
	w2 = w[:k]
	return w2/np.sum(w2)
