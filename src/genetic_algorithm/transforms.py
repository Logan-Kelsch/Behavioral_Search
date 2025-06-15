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


## NOTE BEGIN TRANSFORMATION FUNCTIONS NOTE ##


def t_MAX(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info
	Takes highest value of x over past time window delta. <br>
	utilizing 'bottleneck' library for optimized c code.
	'''
	#a universal general maximum to data development size
	assert Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_max(X, window=Delta, axis=0, min_count=1)



def t_MIN(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info
	Takes lowest value of x over past time window delta <br>
	utilizing 'bottleneck' library for optimized c code.
	'''
	#a universal general maximum to data development size
	assert Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_min(X, window=Delta, axis=0, min_count=1)



def t_AVG(
	X:np.ndarray,
	Delta:int   =   -1
) -> np.ndarray:
	
	'''
	### info 
	takes average value of x over past time window Delta<br>
	utilizing 'bottleneck' library for optimized c code.
	'''
	#a universal general maximum to data development size
	assert Delta < 240, "t_MAX: Delta must be below 240."

	return bn.move_mean(X, window=Delta, axis=0, min_count=1)



def t_NEG(
	X:np.ndarray
) -> np.ndarray:
	
	'''
	### info 
	swapps the sign of all values in x
	utlizing numpy library for optimized c code.
	'''
	
	return np.negative(X)



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



def t_RNG(
	X:np.ndarray,
	Delta_xmin:int	=	-1,
	Delta_xmax:int	=	-1
) -> np.ndarray:
	
	'''
	### info 
	this function scores x within the min and max values of x within Delta xmin and xmax respective time windows<br>
	This function utlizes numpy
	'''
	#a universal general maximum to data development size
	assert (Delta_xmin < 240 & Delta_xmax < 240), "t_RNG: Deltas must be below 240."

	#prepare output buffer of nans
	out = np.full((X.shape[0], X.shape[1]), np.nan, dtype=np.float32)

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
		out[Delta_xmax:] = numer / denom - 0.5

	return out



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

