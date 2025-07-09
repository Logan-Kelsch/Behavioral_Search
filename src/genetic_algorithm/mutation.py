'''

'''

import genetic_algorithm.transforms as transforms
import genetic_algorithm.population as population
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.visualization as visualization

import random
import copy
from typing import Literal
import numpy as np
import math
import pandas as pd
from IPython.display import Image, display

def branch_ntimes(
	tree	:	transforms.T_node,
	branches:	int	
)	->	transforms.T_node:
	
	for iter in range(branches):
		tree.mutate_tree()

	return tree


def quarter_forest_evolution(
	init_forest	:	np.ndarray,
	x_raw		:	np.ndarray,
	tree_optim	:	bool	=	True,
	limit_const_optim	:	bool = False,	
	const_optim_maxiter	:	int	=	10,
	iterations	:	int	=	5
):
	
	'''
	This function is a beginning stage evolution function, where some
	population of trees (forest) is brought in. for each iteration,
	the forest is evaluated. all dead (<=0 score) trees are removed
	from the forest. for all surviving trees, the worst trees are removed
	down until only 25% of the forest remains, this being the best performing trees.
	'''

	print('entering preloop')
	iter_forest:list = init_forest
	chld_forest:list = []


	f_size = len(iter_forest)
	q_size = int(np.floor(f_size/4))


	for i in range(iterations):

		if(tree_optim):
			it = -1
			if(limit_const_optim):
				it = const_optim_maxiter
			chld_forest = optimize_constants(population=chld_forest, iterations=it)

		#bring offspring over to iterating forest
		for tree in chld_forest:
			iter_forest.append(tree)
		chld_forest = []

		p_gen = transforms.forest2features(iter_forest, x_raw)
		p = pd.DataFrame(p_gen)

		visualization.visualize_all_distributions(x=p)

		p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest(p_gen,x_raw[:,3], n_bins=300,lag_range=(2,4))

		#best_oplist = transforms.get_oplist(iter_forest[p_treelist[0]])
		img = visualization.visualize_tree(iter_forest[p_treelist[0]])
		display(img)

		ordered_forest, ordered_scores = population.sort_forest(iter_forest, p_treelist, p_scorelist)
		iter_forest = ordered_forest
		prll_scores = ordered_scores

		#identify trees to be kept
		while(len(iter_forest) > q_size):
			iter_forest.pop()

		print('popped all lists, adding all children')
		if(i<iterations-1):
			#make 3 children for each surviving tree
			for j in range(q_size):

				for three_offspring in [0,1,2]:
					chld_forest.append(
						branch_ntimes(
							copy.deepcopy(iter_forest[j]), 
							int(-np.ceil(np.log(random.random())))
						)
				)

	return iter_forest
		

def optimize_constants(
	population	:	list,
	iterations	:	int	=	-1
)	->	list:
	






	return

def mutate_constant(
	type	:	Literal['delta', 'kappa','U01']	=	'U01',
	val		:	float	=	0,
	dev		:	float	=	0
):
	'''
	This function is going to take some given constant's value
	VAL
	and return a new value picked from a random uniform range of
	DEV standard deviations according to TYPE distribution
	'''


	pin_der = math.erf(dev / math.sqrt(2)) / 2

	u_under, u_over = random.uniform(0, pin_der), random.uniform(0, pin_der)

	area = np.exp(-val)

	area_under= max(area - u_under,0.0)
	area_over = min(area + u_over, 1.0)



	match(type):

		case 'kappa':

			delta_under = -np.log(area_under) if area_under > 0 else float("-inf")
			delta_over	= -np.log(area_over) if area_over > 0 else float("-inf")

			under = int(np.clip(val-delta_under, 1, 239))
			over  = int(np.clip(val+delta_over, 1, 239))

			return under, over
		
		case 'delta':

			delta_under = -np.log(area_under) if area_under > 0 else float("-inf")
			delta_over	= -np.log(area_over) if area_over > 0 else float("-inf")

			under = round(np.clip(val-delta_under, 1, 239))
			over  = round(np.clip(val+delta_over, 1, 239))

			return under, over
		
		case 'alpha':
			raise NotImplementedError(f"Alpha has not been implemented into mutate_constants.")
		
	#function is done here