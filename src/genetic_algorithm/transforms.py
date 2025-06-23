'''
Logan Kelsch - 6/8/2025

This file contains the functions that will be used for the manipulation of features
during the development / evolution stage of a gene or population.

Please refer to the pdf for more detailed description. 

## Transformation functions and transformation trees

Transformation functions (Phi) of some feature(s) (x in X) will be used for the development of information observation by genes.
All raw features will be brought into log-space immediately, and all operations will be done in this space.
The transformations are as follows: Given some current time t, #note {} denotes variables needed

MAX {Delta}: takes highest value of x over past time window Delta
Min {Delta}: Takes lowest value of x over past time window Delta
AVG {Delta}: Takes average of x over past time window Delta
NEG {}: Takes negative value of x
DIF {alpha}: Takes difference of x and some variable alpha (ex: x-a)
VAR {alpha}: Takes squared difference of x and some variable alpha
RNG {Delta_xmin, Delta_xmax}: Ranges x in terms of the max x value and min x value over past time respective time windows
HKP {kappa}: Takes the self-excited linear-space feature x (brought from log-space) using decay constant kappa>0

These transformations are designed to be partially atomic and partially structured, all while being able to be applied in any order and to any extend desired.

---

Transformation trees will be the method of representing a gene's evolutionary development of observed data, 
for both visual interpretation and algorithmic data transforming.
'''

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import bottleneck as bn
from typing import Optional, Union
import random
from collections import deque
from dataclasses import dataclass, field
import genetic_algorithm.utility as utility


## NOTE BEGIN TRANSFORMATION FUNCTIONS NOTE ##

#func #0
'''
func 0 is reserved for the identity function on some feature x. 
This function is not programmed, of course as it is not needed.
'''

#func #1
def t_MAX(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info
	Takes highest value of x over past time window delta. <br>
	utilizing 'bottleneck' library for optimized c code.
	### optimized?
	- this is the most optimal version
	'''
	#a universal general maximum to data development size
	assert 0 < Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_max(X, window=Delta, axis=0, min_count=1)


#func #2
def t_MIN(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info
	Takes lowest value of x over past time window delta <br>
	utilizing 'bottleneck' library for optimized c code.
	### optimized?
	- this is the most optimal version
	'''
	#a universal general maximum to data development size
	assert 0 < Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_min(X, window=Delta, axis=0, min_count=1)


#func #3
def t_AVG(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info 
	takes average value of x over past time window Delta<br>
	utilizing 'bottleneck' library for optimized c code.
	### optimized?
	- this is the most optimal version
	'''
	#a universal general maximum to data development size
	assert 0 < Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_mean(X, window=Delta, axis=0, min_count=1)


#func #4
def t_NEG(
	X:np.ndarray
) -> np.ndarray:
	
	'''
	### info 
	swapps the sign of all values in x
	utlizing numpy library for optimized c code.
	### optimized?
	- this is the most optimal version
	'''
	
	return np.negative(X)


#func #5
def t_DIF(
	X:np.ndarray,
	alpha:np.ndarray,
	in_place:bool			=	False,
	out_arr:np.ndarray|None	=	None
) -> np.ndarray:
	
	'''
	### info
	Subtracting x from variable alpha<br>
	This is going to be done using numpy library for optimized c code.<br>
	<br>
	More specifically, subtracting alpha from x, either elementwise (X.shape==alpha.shape)<br>
	or braodcasting alpha as a vector across rows of x (Y.ndim==1)
	### special params:
	- out:
	- - this function gives the user an opportunity to send in an output buffer for optimization, if this becomes a useful option.
	- - this will minimize malloc calls and will allow for no extra copy
	- in-place:
	- - allows user to modify the x array directly to avoid the use of out arrays
	'''
	
	#gives the user the opportunity to modify the x array directory
	if(in_place):

		X -= alpha
		return X
	
	#if the user provides an out buffer then it will be used
	#this if statement is true when one is not created
	if(out_arr is None):

		out_arr = np.empty_like(X)

	#subtracting with broadcasting and return out buffer
	np.subtract(X, alpha, out=out_arr)
	return out_arr


#func #6
def t_VAR(
	X:np.ndarray,
	alpha:np.ndarray,
	out_arr:np.ndarray|None	=	None,
	in_place:bool			=	False
) -> np.ndarray:
	
	'''
	### info
	computing the squared difference (x - a)**2 from x to variable alpha<br>
	This is going to be done using numpy library for optimized c code.<br><br>
	just as the dif function above, this will be done either elementwise (X.shape==alpha.shape)<br>
	or braodcasting alpha as a vector across rows of x (Y.ndim==1)
	### special params:
	- out:
	- - this function gives the user an opportunity to send in an output buffer for optimization, if this becomes a useful option.
	- - this will minimize malloc calls and will allow for no extra copy
	- in-place:
	- - allows user to modify the x array directly to avoid the use of out arrays
	'''

	#gives the user the opportunity to modify the x array directory
	if(in_place):

		#first get the difference between the two and rewrite into x for memory saving
		np.subtract(X, alpha, out=X)
		
		#then square the difference at the value just by multiplying in place with numpy
		np.multiply(X, X, out=X)
		return X
	
	#if this is not being done in place, make sure out buffer is made
	if(out_arr is None):

		out = np.empty_like(X)

	#using numpy to subtract and bring the values to the out array
	np.subtract(X, alpha, out=out_arr)

	#then using numpy multiply to square the out array in place
	np.multiply(out_arr, out_arr, out=out_arr)

	return out_arr


#func #7
def t_RNG(
	X:np.ndarray,
	Delta_xmin:int	=	-1,
	Delta_xmax:int	=	-1,
	out_arr:np.ndarray|None	=	None
) -> np.ndarray:
	
	'''
	### info 
	this function scores x within the min and max values of x within Delta xmin and xmax respective time windows<br>
	This function utlizes numpy
	'''
	#a universal general maximum to data development size
	assert (Delta_xmin < 240 & Delta_xmax < 240), "t_RNG: Deltas must be below 240."

	if(out_arr is None):
		out_arr = np.empty_like(X)

	#num rows we can compute
	m = X.shape[0] - max(Delta_xmax, Delta_xmin)

	#slices for vectorized diff
	current = X[Delta_xmax:]
	vec1 = X[Delta_xmax - Delta_xmin : (Delta_xmax - Delta_xmin) + m]
	vec2 = X[:m]

	#complete vectorized subtraction 
	numer = current - vec1
	denom = vec2	- vec1

	#suppress div warning if den==0
	#complete division
	with np.errstate(divide='ignore', invalid='ignore'):
		out_arr[Delta_xmax:] = numer / denom - 0.5

	return out_arr


#func #8
def t_HKP(
	X:np.ndarray,
	out_arr:np.ndarray|None	=	None,
	kappa:int   =   -1
) -> np.ndarray:
	
	'''
	
	'''

	#sanity check on decay value since decay coefficient is e^-kappa
	assert kappa > 0, "t_HKP: kappa must be over 0 to avoid inverse decay."

	k = np.exp(-kappa)

	if(out_arr is None):
		out_arr = np.empty_like(X)

	#init first time step
	out_arr[0] = X[0]

	#iterate through featureset
	for t in range(1, X.shape[0]):

		#vectorize across features
		out_arr[t] = k*out_arr[t-1] + X[t]

	return out_arr


## NOTE END TRANSFORMATION FUNCTIONS NOTE ##


## NOTE BEGIN beta NUMEXPR STRING CREATION NOTE ##

def t_NEG_str():
	str = " ( -tmpX ) "
	return str

def t_DIF_str():
	str = " ( tmpX - tmpA ) "
	return str

def t_VAR_str():
	str = " ( tmpX - tmpA ) ** 2 "
	return str

def t_RNG_str():
	str = " ( tmpX1 - tmpX2 ) / ( tmpX3 - tmpX2 ) - 0.5 "
	return str

## NOTE END beta NUMEXPR STRING CREATION NOTE ##


## NOTE BEGIN transformation tree / node logic NOTE ##

'''
ultimately, I will need some kind of map, maybe multiple.

I will need some kind of map for computation decyphering.
This is because I will need to be able to have some given transformation tree
and be able to turn it into a string of operations from leaf to root
in some sort of tuple form 
'''
t_map = {}

#relevant raw features map
#this allows mutation to move in a healthy manner
#new trees or branches can go to volume or h,l,c group
#h,l,c group can move freely within itself
rrf_map = {
	0	:	[1, 2, 3, 4],
	1	:	[1, 2, 3],
	2	:	[1, 2, 3],
	3	:	[1, 2, 3],
	4	:	[4]
}
#NOTE NOTE NOTE DEV NOTE may want to consider option to dissolve rrf map during generation


class T_node:
	'''
		# Transformation Tree Node #
		### info: ###
		This class holds the nesting variable that will construct some given<br>
		transformation tree on some set of raw feature(s).<br><br>
		an attr 'type' value of zero symbolizes a leaf node.
		### params: ###
		- t-type:
		- - transformation type (0-8) ex: avg, idy, hkp
		- x:
		- - index value for raw feature, OR is child node.
		- - default is 3 since that is the index for close price.
		- kappa:
		- - used if this node is hkp function (8) k>0
		- alpha:
		- - used if this node is dif or var (5, 6)
		- - can be a constant (float), a raw feature index (int) or can be a child node.
		- delta:
		- - used if this node is max, min, avg, or rng (1, 2, 3, 7)
		- delta2:
		- - used if this function is rng (7) 
		- random:
		- - parameter used if an entirely new entirely random node should be generated

	function numbers:
	0- identity
	1- max 
	2- min
	3- avg
	4- neg
	5- dif
	6- var
	7- rng
	8- hkp
	'''
	def __init__(
		self,
		type	:	int				=	0,
		x		:	Union[int,"T_node"]	=	0,
		kappa	:	float			=	None,
		alpha	:	Union[float,"T_node"]	=	None,
		delta	:	int				=	None,
		delta2	:	int				=	None,
		random	:	bool	=	False
	) -> None:
		
		'''
		When this node is initiated, it could either be:
		- pre-made and ready for use.
		- an identity node
		- 
		'''
		
		
		self._type	= type
		self._x		= x
		self._kappa = kappa
		self._alpha = alpha
		self._delta = delta
		self._delta2= delta2
		#check to see if this node came in as total random
		if(random):
			self.random()
		
		
	
	def mutate_tree(
		self
	):
		'''
		Considering all parameter mutations are done in the<br>
		local (intra-dim) optimization step, the only things that are mutatable are:
		- THIS node's transformation type IF it is a leaf or parent of a leaf.
		- A parameter x (or alpha) into a new node with param "random" as true.
		This function is called on the root node, and passes down the mutation function until it is not on a leaf node.
		'''

		#check to see if this node is NOT a leaf node
		if(self._type!=0):

			#this node is not a leaf node, must mutate children instead
			#check first to see if there can be multiple children from this node
			#at the same time, if the first two conditions are true,
			#then for the whole statement to be true, a coin flip is made.
			#coin flip is for code minimization to see if alpha should be mutated instead of x
			#if it returns false then we can proceed with one X mutation
			if(self._type==5 or self._type==6):

				#are there two branches that can be mutated? (two children, x and alpha)
				if(isinstance(self._alpha, T_node)):

					#if there are two branches, coin flip!
					if(random.random()>0.5):

						#go to alpha child node and mutate alpha branch
						self._alpha.mutate_tree()

					else:

						#go to x child node and mutate x branch
						self._x.mutate_tree()

				#if alpha can be mutated into a transformation
				else:

					#give it a (1-1/e)% chance of being mutated
					if(np.log(np.random.rand())>-1):
						#in this case, generate new random and relevant alpha transformation

						#starting with making alpha an empty node
						self._alpha = T_node()

						#since this node is a leaf, _x is safe and is an int.
						self._alpha.random(rrf=self._x.get_rrf())

			
			#if this is reached, that means the only mutatable branch is going to be x
			else:

				#go to x child node and mutate x
				self._x.mutate_tree()

		#if this else is entered, that means that the current node is a leaf node and will be mutated!
		else:

			#get the number of mutations that will be done
			#keeping this between 1 and 2 using inverse transform sampling
			#then differentiating between whether or not the mutation will be done ALSO on a possible child node
			#depending on if the generated sample is over or under -1.5 (rounding)
			child_mutation = True if np.log(np.random.rand()) < -1.5 else False

			#when this is reached, we have a type of 0 and an x of SOMETHING!
			#so now we randomly pick a transformation type, shift x, then pick random relevant parameters

			#pick random transformation function
			self._type = random.choice(list(range(1, 9)))
			
			#pick rrf that will be used
			rrf = random.choice(rrf_map[self._x])

			#pick random possibly shifted relevant raw feature to observe
			self._x = T_node()

			#pass the observed raw feature to the child node 
			self._x._x = rrf

			match(self._type):

				case 1|2|3|7:

					#using inverse transformation sampling and scaling by 26 for relevance according to 15m candles, may make this dynamic
					self._delta = int(np.clip(np.ceil(-26 * np.log(np.random.rand())), 1, 239))

					if(self._type==7):

						#doing the same exact thing as above, but for the other delta for ranging function
						self._delta2 = int(np.clip(np.ceil(-26 * np.log(np.random.rand())), 1, 239))

				case 5|6:

					#until I come up with some better way to go again and formulate this, 
					#it must remain zero
					self._alpha = int(0)

				case 8:

					#pick kappa value at random using inverse transform sampling
					self._kappa = round(-np.log(np.random.rand()), 3)

		#if a mutation is being passed to the child, then go mutate child!
		if(child_mutation):
			self._x.mutate_tree()

	def get_tlist(
		self,
		tlist	:	list
	):
		'''
		### info
		This function is recursive and is called at each node for external function get_oplist()
		### format
		Each item in the oplist will be a list of 3 items, each item depicting:
		- transformation type (sign represents inversion of sources)
		- flags, used for pushing or popping of xptr in/out of vstk
		- variables needed in the operation (ex: delta, delta2, kappa)
		'''

		#this function was originally crafted during my table server shift on 6/16/2025

		#x is tnode? (meaning node._type is nonzero)
		if(isinstance(self._x, T_node)):

			#go to x child node
			tlist = self._x.get_tlist(tlist=tlist)

		#x is index (of some raw feature)? (meaning node._type is zero)
		if(isinstance(self._x, int)):

			#push new item into the tlist (type, index, params(none needed))
			#this action (0) represents: go and put raw feature of index ._x into an ndarray
			# action zero can also represent utility functionality (vstk pop/push) per flags
			tlist.append([0, self._x, ()])
			return tlist
		
		#NOTE to reach this point in recursion, x branch bias has been fully explored NOTE#

		#does this node possibly have two children (alpha)?
		#I am using 'is not' syntax out of fear over auto-overwridden __eq__
		if(self._alpha is not None):

			#if this exists, that means we need to utilze a variable stack (vstk) to hold
			#the value/array (xptr) waiting for operation

			#dev note: the pop and push is essential. this is a byproduct of trying to 
			#	flatten a tree into operations and is simplest as far as my brainstorming is concerned.

			#This flag (-1) represents: take the variable in xptr and push onto vstk!
			#there is no directed needing for operation (None) and no variables either ()
			tlist.append([0, -1, ()])

			#under all cases, we need to operate ._x as from top of vstk
			#this will be denoted as a negative transformation variable indicating flipped var sources
			#we will split the operation into whether or not alpha is a float or array(node)

			#alpha is tnode?
			if(isinstance(self._alpha, T_node)):

				#go to alpha child node
				tlist = self._alpha.get_tlist(tlist=tlist)

				#negative type suggests x is from top of vstk
				#zero flag value suggests alpha is from xptr
				tlist.append([-1*self._type, 0, ()])

			else:

				#in this case we will be operating on alpha as a float value

				#negative type suggests x is from top of vstk
				#(1) flag value suggests alpha never made it to xptr and is passed in
				tlist.append([-1*self._type, 1, (self._alpha,)])

			#now we also need to pop the used variable from vstk
			#we will represent this with a flag of (-2)
			tlist.append([0, -2, ()])

			#by here we have completed all actions for any path with alpha
			return tlist

		#NOTE to reach this point in recursion, we have a prepared x, as well as no alpha var NOTE#
				
		#we can operate with prepared variables!
		#we will throw the variables into the third list param according to protocol
		match(self._type):

			#our unseen transformations so far are 1, 2, 3, 4, 7, 8

			case 1|2|3:

				#this case consists of the use of delta
				tlist.append([self._type, 0, (self._delta,)])

			case 4:

				#this case consists of no variable use
				tlist.append([self._type, 0, ()])

			case 7:

				#this case consists of the use of delta1,2
				tlist.append([self._type, 0, (self._delta, self._delta2)])
			
			case 8:

				#this case consists of the use of kappa
				tlist.append([self._type, 0, (self._kappa,)])

			case _:
				raise ValueError(f"FATAL: match-case failed checking T_node trans_type, got {self._type}")
			
		#all possible operations have been added to tlist
		return tlist
				
	
	def random(
		self,
		rrf	:	int	=	0
	):
		'''
		Usage of this function destroys and child nodes and forces random generation.
		'''

		#force transformation type to be set to 'identity'
		self._type = 0

		#get a random new raw feature index that will be explored in the random tree
		self._x = int(random.choice(rrf_map[rrf]))

		#will anyone ever see this comment? <3 from Logan 6/15/2025 11:35pm
		self.mutate_tree()


	def get_rrf(
		self
	):
		'''
		This function looks down the tree and grabs the relevant raw feature on x branch bias.
		'''
		if(isinstance(self._x, T_node)):
			return self._x.get_rrf()
		else:
			return self._x
		
	@property
	def alpha(self):
		return self._alpha
	
	@alpha.setter
	def alpha(self, new:any):
		self._alpha

	@property
	def delta(self):
		return self._delta
	
	@delta.setter
	def delta(self, new:any):
		self._delta

	@property
	def delta2(self):
		return self._delta2
	
	@delta2.setter
	def delta2(self, new:any):
		self._delta2

	@property
	def kappa(self):
		return self._kappa
	
	@kappa.setter
	def kappa(self, new:any):
		self._kappa

	@property
	def x(self):
		return self._x
	
	@x.setter
	def x(self, new:any):
		self._x

	@property
	def type(self):
		return self._type
	
	@type.setter
	def type(self, new:any):
		self._type
		
def get_oplist(
	root	:	T_node
) -> list:
	'''
	This function is the recursive intiator for a transformation tree to flatten into an operational list.<br>
	### format
		Each item in the oplist will be a list of 3 items, each item depicting:
		- transformation type (sign represents inversion of sources)
		- flags, used for pushing or popping of xptr in/out of vstk
		- variables needed in the operation (ex: delta, delta2, kappa)
	'''

	#initiate the operation list
	tlist = []

	#return the results from the root node
	return root.get_tlist(tlist=tlist)

def oplist2tstack(
	oplist	:	list
)	->	list:
	
	'''
	This function takes some given oplist and 
	creates more legible single int list for shortest common supersequence.
	'''

	#initialize transformation stack
	tstack = []

	#go through each operation in the stack
	for op in oplist:

		#if this is true, this means that 
		#a regular operation is being completed, sign can be neglected.
		if(op[0]!=0):

			#put the transformation integer in the tstack
			tstack.append(abs(op[0]))

		#for 0 operations, can only parallelize true identity transformation
		#identity transformation is used for turning feature index into ndarray.
		#break up 0 type into flags
		else:

			#if the flag is negative, can neglect!
			#those operations cannot be parallel,
			#and are not considered in the scs search
			if(op[1]>=0):

				#the only case that is added here is identity (feature grabbing)
				tstack.append(0)
	
	return tstack

		
			


def forest2features(
	population	:	list,
	x_raw		:	np.ndarray
)	-> np.ndarray:
	
	'''
	### info:
	This function will take an original set of transformation trees<br>
	and use them to generate a corresponding parallel featureset
	### possible oplist values:
	- o[0, >=0, ()] - get feature o[1] from x -> xptr
	- o[0, -1, ()] - push xptr -> vstk[-0]
	- o[0, -2, ()] - pop top of vstk -> [delete it]
	- o[n<0, 0, ()] - operate t[-o[n]] with x from vstk[-1] and alpha in xptr -> xptr
	- o[n<0, 1, (1 item)] - operate t[-o[n]] with x from vstk[-1] and alpha from o[2] -> xptr
	- 0[n>0, 0, (1-2 items)] - operate t[o[n]] on xptr using corresponding o[2] parameters
	'''

	#initiate xptr holding dynamic data parallel with population
	xptr = np.empty((x_raw.shape[0],len(population)))

	#initiate list of oplists parallel with population
	oplists = np.empty(len(population), dtype=list)

	#initiate list of tstack, parallel with population
	tstacks = np.empty(len(population), dtype=list)

	#initiate list of variable stacks parallel with population
	vstk = np.empty(len(population), dtype=list)

	for i in range(vstk.shape[0]):
		vstk[i] = []

	#collect oplists for each member of population
	for i, this_tree in enumerate(population):

		#for each tree, get the op list
		oplists[i] = get_oplist(root=this_tree)

		#collect transformation stacks, for legibility within
		#the shortest common supersequence algorithm
		tstacks[i] = oplist2tstack(oplists[i])

	#get the shortest common supersequence from all tstacks in one go
	#NOTE this function will result in a final string of operations for full transforming NOTE#
	#t, i is transformation ID and stack indices for correlating transformation
	t_supseq, i_supseq = utility.shortest_common_supersequence(seqs=tstacks)

	#create hotloop for operating on oplist patterns
	for op, (t_ss, i_ss) in enumerate(zip(t_supseq, i_supseq), 1):

		#NOTE beginning vstk pop and push interaction section NOTE#
		#check always to see if any vstk operations need completed first!
		
		#print([oplists[i] for i, _ in enumerate(oplists)])

		#get indices of all push instances
		vstk_push_indices = [
			i for i, lst in enumerate(oplists)
			if lst and lst[0][0]==0 and lst[0][1]==-1
		]

		#get indices of all pop instances
		vstk_pop_indices = [
			i for i, lst in enumerate(oplists)
			if lst and lst[0][0]==0 and lst[0][1]==-2
		]

		#push all requested instances of vstk from collected instances
		for i in vstk_push_indices:
			vstk[i].append(xptr[:,i])

		#pop all requested instances of vstk from collected instances
		for i in vstk_pop_indices:
			vstk[i].pop()

		#then, for all oplists, we can combine and remove all vstk push/pop interactions
		vstk_poppush_indices = list(set(vstk_pop_indices)|set(vstk_push_indices))

		#remove pop/push operations in oplists
		for i in vstk_poppush_indices:
			#operation lists are moving front to back, so oplist pop is always zero
			oplists[i].pop(0)

		#NOTE ending vstk pop / push interaction section NOTE#

		#here we need to add transformation based interpretation
		
		#we will be taking i_ss and using that as a mask
		# as in using the where= parameter in the actual transformation functions and use out= too!
		#call the transformation function using match case and t_ss as id
		#then pop all items with i_ss from oplist

		#turn destination indices into numpy array for numpy optimization
		op_idx  = np.asarray(i_ss,  dtype=int)
		
		match(t_ss):

			#this case is entered for the identity function, 
			#which holds place for fetching raw feature data
			case 0:

				#go get the raw feature indices from the flag of each first op in the oplist
				rf_idx = [oplists[i][0][1] for i in i_ss]

				#turn retrieving index list into numpy array
				flag_arr = np.asarray(rf_idx, dtype=int)

				#print(xptr.shape, x_raw.shape)

				#this does, for each k:
				#xptr[:, op_idx[k]] = raw_features[:, flag_arr[k]]
				xptr[:, op_idx] = x_raw[:, flag_arr]

			#this case is entered for the function max(x, delta)
			case 1:

				#this is using in place rolling max with per-column window sizes to 
				#accomodate to inequal delta values for different sources

				#go get the deltas and turn them into numpy array
				#deltas are located in the first slot of parameter section of oplist
				#print(oplists[i_ss[0]], oplists[i_ss[0]][0], oplists[i_ss[0]][0][2])

				deltas = [oplists[i][0][2][0] for i in i_ss]
				
				#turn delta in to numpy array
				deltas = np.asarray(deltas, dtype=int)

				T = xptr.shape[0]

				#we can do matching deltas at one time, 
				#so we will see if any match to speed it up
				unique_deltas = np.unique(deltas)

				#loop over each delta value
				for delta in unique_deltas:

					#go and get columns with the current delta value
					cols = op_idx[deltas == delta]

					#print(cols.shape)

					#view of xptr for just those columns
					block = xptr[:, cols]

					#repeat first row (delta-1) times for padding
					#this is going to have shape (delta-1) by k
					pad = np.repeat(block[0:1, :], delta-1, axis=0)

					#vstack to form the padded array
					#this now has shape (T+delta-1) by k
					ap = np.vstack((pad, block))

					#look at the given window
					win = sliding_window_view(ap, window_shape=delta, axis=0)

					#allocate some memory for the outputs
					#out_buf = np.empty((T, cols.size), dtype=xptr.dtype)

					#compute the max over each window and put it into the output buffer
					#np.maximum.reduce(win, axis=1, out=out_buf)

					out_buf = win.max(axis=2)

					#print(win.shape)

					#write the results back and go again through the loop
					xptr[:, cols] = out_buf

					#had a lot of crashing errors in my old code, 
					#hopefully excessively calling delete will keep the code running fine
					del cols, block, pad, ap, win, out_buf

				#delete all long term holders once these operations are done
				del deltas, unique_deltas


			#this case is entered for the function min(x, delta)
			case 2:
				
				#this is using in place rolling min with per-column window sizes to 
				#accomodate to inequal delta values for different sources

				#go get the deltas and turn them into numpy array
				#deltas are located in the first slot of parameter section of oplist
				deltas = [oplists[i][0][2][0] for i in i_ss]
				
				#turn delta in to numpy array
				deltas = np.asarray(deltas, dtype=int)

				T = xptr.shape[0]

				#we can do matching deltas at one time, 
				#so we will see if any match to speed it up
				unique_deltas = np.unique(deltas)

				#loop over each delta value
				for delta in unique_deltas:

					#go and get columns with the current delta value
					cols = op_idx[deltas==delta]

					#view of xptr for just those columns
					block = xptr[:, cols]

					#repeat first row (delta-1) times for padding
					#this is going to have shape (delta-1) by k
					pad = np.repeat(block[0:1, :], delta-1, axis=0)

					#vstack to form the padded array
					#this now has shape (T+delta-1) by k
					ap = np.vstack((pad, block))

					#look at the given window
					win = sliding_window_view(ap, window_shape=delta, axis=0)

					#allocate some memory for the outputs
					#out_buf = np.empty((T, cols.size), dtype=xptr.dtype)

					#compute the min over each window and put it into the output buffer
					#np.minimum.reduce(win, axis=1, out=out_buf)

					out_buf = win.min(axis=2)

					#write the results back and go again through the loop
					xptr[:, cols] = out_buf

					#had a lot of crashing errors in my old code, 
					#hopefully excessively calling delete will keep the code running fine
					del cols, block, pad, ap, win, out_buf

				#delete all long term holders once these operations are done
				del deltas, unique_deltas

			#this case is entered for the function avg(x, delta)
			case 3:

				#go get the deltas and turn them into numpy array
				#deltas are located in the first slot of parameter section of oplist
				deltas = [oplists[i][0][2][0] for i in i_ss]

				#going to convert delta values into a numpy array to ensure numpy smoothness
				deltas = np.asarray(deltas, dtype=int)
				
				T = xptr.shape[0]

				#get each distinct delta value to run similar values in parallel
				unique_deltas = np.unique(deltas)
				
				#here we are going to precompute counts (1, 2, .., min(delta, T)) for each delta
				counts = {
					delta: np.arange(1, min(delta, T) + 1, dtype=xptr.dtype)
					for delta in unique_deltas
				}

				#loop over each column and its window size
				for col, delta in zip(op_idx, deltas):
					
					#go get the colums that can be done in parallel
					x = xptr[:, col]
					
					#sum up all values with cumsum this will make the computation way way easier
					csum = x.cumsum()
					
					#allocate some memory for output
					out = np.empty(T, dtype=xptr.dtype)

					#number of growing window steps where we have not reached sample delta
					m = counts[delta].size
					
					#compute avg for partial windows if needed
					out[:m] = csum[:m] / counts[delta]
					
					if T > delta:

						#compute the avg for full delta window for remaining entries
						out[delta:] = (csum[delta:] - csum[:-delta]) / delta

					#write the results back into the array
					xptr[:, col] = out
					
					#delete the temp variables for free memory immediately
					#hoping this actually helps this run indef.
					del x, csum, out

				#clean up the helper arrays for this function now that the operations are complete
				del deltas, unique_deltas, counts

			#this case is entered for the function neg(x)
			case 4:
				
				#really almost nothing is needed here
				#multiply the columns by -1
				xptr[:, op_idx] *= -1

			#this case is entered for the function dif(x, alpha)
			case 5:
				
				#first we need to use opidx to find the arrays in xptr that need
				#filled in (fully with the provided constant values)
				missing = [oplists[i][0][1] for i in op_idx]
				
				#and convert it to numpy array for smoothy smoothy smoothness? 
				#no idea if it actually helps much
				missing = np.asarray(missing, dtype=int)

				#num rows and columns to operate on 
				T = xptr.shape[0]
				k = op_idx.size

				#first we have to build x (in the function definition of "x - alpha")
				#we will do this by stacking vstk[i][-1] for each i in op_idx
				x = np.stack([vstk[i][-1] for i in op_idx], axis=1)

				#print([vstk[i][-1] for i in op_idx])
			
				#second we need to build alpha (in the function definition of "x - alpha")
				#this happens to be partially in xptr depending where alpha values are coming from
				#if alpha is still a constant then it is not in xptr and we need to bring it in
				#if alpha is some given result of a branch from the transformation tree, 
				#then it will already be in xptr and we need to do nothing until the actual dif operation

				#use missing boolean mask to find which indices to fill
				fill_cols = op_idx[missing==1]

				#print(x.shape)

				#get the constant values only from missing indices
				#these will be used for column filling
				consts = np.array([oplists[col][0][2][0] for col in fill_cols], dtype=xptr.dtype)

				#fill in the 'empty' columns with the constant alphas 
				xptr[:, fill_cols] = consts

				#now here is the actual difference operation
				#single c loop
				np.subtract(x, xptr[:, op_idx], out=xptr[:, op_idx])

				#this entire 2d array is no longer needed and the vstk value will be popped in the
				#subsequent operation in oplist of those operation stacks
				del x

			#this case is entered for the function var(x, alpha)
			case 6:
				
				#first we need to use opidx to find the arrays in xptr that need
				#filled in (fully with the provided constant values)
				missing = [oplists[i][0][1] for i in op_idx]
				
				#and convert it to numpy array for smoothy smoothy smoothness? 
				#no idea if it actually helps much
				missing = np.asarray(missing, dtype=int)

				#num rows and columns to operate on 
				T = xptr.shape[0]
				k = op_idx.size

				#first we have to build x (in the function definition of "x - alpha")
				#we will do this by stacking vstk[i][-1] for each i in op_idx
				x = np.stack([vstk[i][-1] for i in op_idx], axis=1)
			
				#print(vstk.shape, "-----", vstk[op_idx[0]].shape)

				#second we need to build alpha (in the function definition of "x - alpha")
				#this happens to be partially in xptr depending where alpha values are coming from
				#if alpha is still a constant then it is not in xptr and we need to bring it in
				#if alpha is some given result of a branch from the transformation tree, 
				#then it will already be in xptr and we need to do nothing until the actual dif operation

				#use missing boolean mask to find which indices to fill
				fill_cols = op_idx[missing==1]

				#get the constant values only from missing indices
				#these will be used for column filling
				consts = np.array([oplists[col][0][2][0] for col in fill_cols], dtype=xptr.dtype)

				#fill in the 'empty' columns with the constant alphas 
				xptr[:, fill_cols] = consts

				#now here is the actual difference operation
				#single c loop
				np.subtract(x, xptr[:, op_idx], out=xptr[:, op_idx])

				#the only difference is here we are squaring the result
				#this is a loop in C and also in place
				xptr[:, op_idx] **= 2

				#this entire 2d array is no longer needed and the vstk value will be popped in the
				#subsequent operation in oplist of those operation stacks
				del x

			#this case is entered for the function rng(x, delta, delta2)
			case 7:

				#NOTE begin creation of max matrix NOTE#

				#this is using in place rolling max with per-column window sizes to 
				#accomodate to inequal delta values for different sources

				#go get the deltas and turn them into numpy array
				#deltas are located in the first slot of parameter section of oplist
				deltas = [oplists[i][0][2][1] for i in i_ss]
				
				#turn delta in to numpy array
				deltas = np.asarray(deltas, dtype=int)

				T = xptr.shape[0]

				#we can do matching deltas at one time, 
				#so we will see if any match to speed it up
				unique_deltas = np.unique(deltas)

				x_max = np.empty_like(xptr)

				#loop over each delta value
				for delta in unique_deltas:

					#go and get columns with the current delta value
					cols = op_idx[deltas==delta]

					#view of xptr for just those columns
					block = xptr[:, cols]

					#repeat first row (delta-1) times for padding
					#this is going to have shape (delta-1) by k
					pad = np.repeat(block[0:1, :], delta-1, axis=0)

					#vstack to form the padded array
					#this now has shape (T+delta-1) by k
					ap = np.vstack((pad, block))

					#look at the given window
					win = sliding_window_view(ap, window_shape=delta, axis=0)

					#allocate some memory for the outputs
					#out_buf = np.empty((T, cols.size), dtype=xptr.dtype)

					#compute the max over each window and put it into the output buffer
					#np.maximum.reduce(win, axis=1, out=out_buf)

					out_buf = win.max(axis=2)

					#write the results back and go again through the loop
					x_max[:, cols] = out_buf

					#had a lot of crashing errors in my old code, 
					#hopefully excessively calling delete will keep the code running fine
					del cols, block, pad, ap, win, out_buf

				#delete all long term holders once these operations are done
				del deltas, unique_deltas

				#NOTE end creation of max matrix NOTE#

				#NOTE begin creation of min matrix NOTE#

				#this is using in place rolling min with per-column window sizes to 
				#accomodate to inequal delta values for different sources

				#go get the deltas and turn them into numpy array
				#deltas are located in the first slot of parameter section of oplist
				deltas = [oplists[i][0][2][0] for i in i_ss]
				
				#turn delta in to numpy array
				deltas = np.asarray(deltas, dtype=int)

				T = xptr.shape[0]

				#we can do matching deltas at one time, 
				#so we will see if any match to speed it up
				unique_deltas = np.unique(deltas)

				x_min = np.empty_like(xptr)

				#loop over each delta value
				for delta in unique_deltas:

					#go and get columns with the current delta value
					cols = op_idx[deltas==delta]

					#view of xptr for just those columns
					block = xptr[:, cols]

					#repeat first row (delta-1) times for padding
					#this is going to have shape (delta-1) by k
					pad = np.repeat(block[0:1, :], delta-1, axis=0)

					#vstack to form the padded array
					#this now has shape (T+delta-1) by k
					ap = np.vstack((pad, block))

					#look at the given window
					win = sliding_window_view(ap, window_shape=delta, axis=0)

					#allocate some memory for the outputs
					#out_buf = np.empty((T, cols.size), dtype=xptr.dtype)

					#compute the min over each window and put it into the output buffer
					#np.minimum.reduce(win, axis=1, out=out_buf)

					out_buf = win.min(axis=2)

					#write the results back and go again through the loop
					x_min[:, cols] = out_buf

					#had a lot of crashing errors in my old code, 
					#hopefully excessively calling delete will keep the code running fine
					del cols, block, pad, ap, win, out_buf

				#delete all long term holders once these operations are done
				del deltas, unique_deltas

				#NOTE end creation of min matrix NOTE#

				#now that we have all four matricies needed
				#   x - x_min
				#     over
				# x_max - x_min

				#we can proceed with a more memory efficient approach
				
				#form the denominator first
				x_denom = np.empty_like(xptr)

				#do subtraction in denominator
				np.subtract(x_max[:, op_idx], x_min[:, op_idx], out=x_denom[:, op_idx])

				#we now no longer need x_max
				del x_max

				#do subtraction in numerator
				xptr[:, op_idx] -= x_min[:,op_idx]
				#np.subtract(xptr[:, op_idx], x_min[: op_idx], out=xptr[:, op_idx])

				#we no longer need x_min
				del x_min

				#now do division operation
				np.divide(xptr[:, op_idx], x_denom[:, op_idx], out=xptr[:, op_idx])

				#we no longer need the denominator
				del x_denom

				#now we can subtract the .5 in place
				#this operation makes sense when pulling from log space
				#so that exciting functionality can work even on ranged values!
				#although only drawback is it will be bias to excitation on the positive value end
				#of whatever is being analyzed
				xptr[:, op_idx] -= 0.5

			#this case is entered for the function hkp(x, kappa)
			case 8:

				#go get the kappas and turn them into numpy array
				#kappas are located in the first slot of parameter section of oplist
				kappas = [oplists[i][0][2][0] for i in i_ss]
				
				#turn delta in to numpy array
				kappas = np.asarray(kappas, dtype=np.float32)

				#simplify coefficient to a single variable alpha
				alphas = np.exp(-kappas)

				#run recursion in a single loop, not sure if there is a simpler way to
				#go about this or anything in parallel with summations,
				#but I honestly did not feel like trying to figure it out this 
				#should be fast enough
				T = xptr.shape[0]
				for t in range(1, T):

					#vectorized update across selected columns in one loop
					#this is in C
					xptr[t, op_idx] += alphas * xptr[t-1, op_idx]

			case _:
				raise ValueError(f"t_ss value is not valid in forest2feature. ({t_ss})")
			
		#pop all completed operations in used oplists 
		for i in i_ss:
			oplists[i].pop(0)

	#delete variables, they honestly should not have anything 
	#not sure if residual memory holds or anything
	del vstk, tstacks

	#once we have reached here, all operations are completed and we can return completed xptr array
	return xptr