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
import bottleneck as bn
from typing import Optional, Union
import random
from dataclasses import dataclass, field


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


class T_node():
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

	'''

	'''
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
		type	:	int					=	0,
		x		:	int|"T_node"		=	0,
		kappa	:	float				=	0,
		alpha	:	float|int|"T_node"	=	0,
		delta	:	int					=	0,
		delta2	:	int					=	0,
		random	:	bool	=	False
	) -> None:
		
		'''
		When this node is initiated, it could either be:
		- pre-made and ready for use.
		- an identity node
		- 
		'''
		
		#check to see if this node came in as total random
		if(random):
			self.random()
		
		#if the node is not coming in as random, assign attr per parameters
		else:
			self._type	= type
			self._x		= x
			self._kappa = kappa
			self._alpha = alpha
			self._delta = delta
			self._delta2= delta2

		
		return
	
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

					#until I come up with some better way to go again and formulate this, it must remain zero
					self._alpha = 0

				case 8:

					#pick kappa value at random using inverse transform sampling
					self._kappa = round(-np.log(np.random.rand()), 3)

		#if a mutation is being passed to the child, then go mutate child!
		if(child_mutation):
			self._x.mutate_tree()


	
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